import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn.cluster import KMeans
import json
from datetime import datetime

def make_unique_id(train, test, card, user):
    # 'user_id' と 'card_id' を文字列に変換して連結
    train['unique_id'] = train['user_id'].astype(str) + '_' + train['card_id'].astype(str)
    test['unique_id'] = test['user_id'].astype(str) + '_' + test['card_id'].astype(str)
    card['unique_id'] = card['user_id'].astype(str) + '_' + card['card_id'].astype(str)

    # 混同しないように一旦列を削除
    train = train.drop(['user_id', 'card_id'], axis=1)
    test = test.drop(['user_id', 'card_id'], axis=1)
    card = card.drop(['user_id', 'card_id'], axis=1)

    # 分割できるようにフラグ列を作成
    train['is_train'] = 1
    test['is_train'] = 0

    # trainとtestを結合
    train_test = pd.concat([train, test])

    # cardデータフレームとマージ
    merged_with_card = pd.merge(train_test, card, on='unique_id', how='left')

    # 'unique_id'から'user_id'を抽出
    merged_with_card['user_id'] = merged_with_card['unique_id'].apply(lambda x: int(x.split('_')[0]))

    # userデータフレームとマージ
    final_merged = pd.merge(merged_with_card, user, on='user_id', how='left')
    return final_merged

def removre_dollar_mark(df):
    """ドルマークを外して数値型にする関数"""
    df['amount'] = df['amount'].replace({'\$': '', ',': ''}, regex=True).astype(float)
    df['credit_limit'] = df['credit_limit'].replace({'\$': '', ',': ''}, regex=True).astype(float)
    df['per_capita_income_zipcode'] = df['per_capita_income_zipcode'].replace({'\$': '', ',': ''}, regex=True).astype(float)
    df['yearly_income_person'] = df['yearly_income_person'].replace({'\$': '', ',': ''}, regex=True).astype(float)
    df['total_debt'] = df['total_debt'].replace({'\$': '', ',': ''}, regex=True).astype(float)
    return df

def fill_zipnumber(df):
    """ONLINE以外の利用で、欠損しているzipを埋める関数"""
    # merchant_cityとmerchant_stateに基づいて最も頻繁に出現するzipコードを取得
    merchant_zip_map = df.dropna(subset=['zip']).groupby(['merchant_city', 'merchant_state'])['zip'].agg(lambda x: x.mode()[0] if not x.mode().empty else None).to_dict()

    # cityとstateに基づいて最も頻繁に出現するzipcodeを取得
    user_zip_map = df.dropna(subset=['zipcode']).groupby(['city', 'state'])['zipcode'].agg(lambda x: x.mode()[0] if not x.mode().empty else None).to_dict()

    # 欠損値を埋める関数
    def fill_missing_zip(row):
        if pd.isna(row['zip']):
            # "ONLINE"という特殊なケースを処理
            if row['merchant_city'] == 'ONLINE' and pd.isna(row['merchant_state']):
                return row['zip']
            merchant_zip = merchant_zip_map.get((row['merchant_city'], row['merchant_state']))
            user_zip = user_zip_map.get((row['city'], row['state']))
            return merchant_zip if merchant_zip is not None else (user_zip if user_zip is not None else None)
        return row['zip']

    # apply関数を用いてzip列の欠損値を埋める
    df['zip'] = df.apply(fill_missing_zip, axis=1)
    return df

def zip_to_lonlat(df):
    """zipから緯度経度を算出する関数"""
    with open('lat_lng_cache.json', 'r') as f:
        lat_lng_dict = json.load(f)

    # キーを数値に変換
    lat_lng_dict = {float(k): v for k, v in lat_lng_dict.items()}

    df['zip'] = df['zip'].astype(float)
    df['shop_lat'] = df['zip'].apply(lambda x: lat_lng_dict.get(x, {}).get('lat', None))
    df['shop_lng'] = df['zip'].apply(lambda x: lat_lng_dict.get(x, {}).get('lng', None))
    return df

def make_distance(df):
    def calculate_distance(row):
        if pd.isna(row['shop_lat']) or pd.isna(row['shop_lng']):
            return -1

        shop_coords = (row['shop_lat'], row['shop_lng'])
        customer_coords = (row['latitude'], row['longitude'])

        # geodesic関数で距離を計算（km単位）
        distance = geodesic(shop_coords, customer_coords).kilometers
        return distance

    # 新しい列に距離を格納
    df['distance_km'] = df.apply(calculate_distance, axis=1)
    return df

def calculate_statistics(df):
    # unique_idとuser_idごとに統計量を計算
    grouped_df = df.groupby(['unique_id', 'user_id']).agg({
        'amount': ['mean', 'max', 'min', 'var'],
        'errors?': lambda x: (x == 'ERROR').sum()
    }).reset_index()

    # カラム名を調整
    grouped_df.columns = ['unique_id', 'user_id', 'amount_mean', 'amount_max', 'amount_min', 'amount_var', 'error_count']

    # 元のDataFrameと統計量のDataFrameをマージ
    merged_df = pd.merge(df, grouped_df, on=['unique_id', 'user_id'], how='left')
    return merged_df

def make_date_feature(df):
    # 'expires'列を日付形式に変換
    df['expires_date'] = pd.to_datetime(df['expires'], format='%m/%Y')
    # 'acct_open_date'列を日付形式に変換
    df['acct_open'] = pd.to_datetime(df['acct_open_date'], format='%m/%Y')

    # 現在の年月を取得（日は1日として扱う）
    now = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    # 年月の差分を計算
    df['expires_diff'] = ((df['expires_date'].dt.year - now.year) * 12) + df['expires_date'].dt.month - now.month
    df['acct_open_diff'] = ((df['acct_open'].dt.year - now.year) * 12) + df['acct_open'].dt.month - now.month

    # リタイアしているかどうかの差分を計算
    df['retirement_diff'] = df['current_age'] - df['retirement_age']

    # birth_monthを2πに正規化
    df['birth_month_rad'] = df['birth_month'] * (2 * np.pi / 12)

    # sinとcosを計算
    df['birth_month_sin'] = np.sin(df['birth_month_rad'])
    df['birth_month_cos'] = np.cos(df['birth_month_rad'])

    # 不要な列（birth_monthとbirth_month_rad）を削除
    df.drop(['birth_month', 'birth_month_rad'], axis=1, inplace=True)
    return df

def binning_score(df):
    # FICOスコアに基づいて新しい列を作成
    bins = [300, 579, 669, 739, 799, 850]
    labels = [1, 2, 3, 4, 5]
    df['fico_label'] = pd.cut(df['fico_score'], bins=bins, labels=labels, right=True)
    # fico_label列を整数型に変換
    df['fico_label'] = df['fico_label'].astype('int')
    return df


def clastering_geometry(df):
    # NaN（ONLINE）を除去してクラスタリング
    df_not_online = df[df['shop_lng'].notna()]
    kmeans = KMeans(n_clusters=3)
    df_not_online['cluster_id'] = kmeans.fit_predict(df_not_online[['shop_lng', 'shop_lat']])

    # 元のDataFrameにクラスタIDをマージ
    df = pd.merge(df, df_not_online[['index', 'cluster_id']], on='index', how='left')

    # ONLINEに対するクラスタIDを設定
    df['cluster_id'].fillna(-1, inplace=True)
    return df

def one_hot_encode_object_columns(df):
    # DataFrameのobject型の列名を取得
    object_columns = df.select_dtypes(include=['object']).columns

    # ワンホットエンコーディングを適用
    df_one_hot = pd.get_dummies(df, columns=object_columns, drop_first=True)

    return df_one_hot

def apply_feature_engineering(train, test, card, user):
    df = make_unique_id(train, test, card, user)
    df = removre_dollar_mark(df)
    df = fill_zipnumber(df)
    df = zip_to_lonlat(df)
    df = make_distance(df)
    df = calculate_statistics(df)
    df = make_date_feature(df)
    df = binning_score(df)
    df = df.drop(['zip','merchant_state','unique_id','expires','acct_open_date','acct_open','user_id','address','city','state','expires_date','merchant_city'],axis=1)
    df = one_hot_encode_object_columns(df)
    df = clastering_geometry(df)
    df = df.drop(['shop_lng', 'shop_lat'],axis=1)
    return df

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

def smote_optimize_and_apply(train_df, target_column, random_state):
    # ターゲットと特徴量を分割
    X = train_df.drop(target_column, axis=1)
    y = train_df[target_column]

    # パイプライン設定
    pipeline = Pipeline([
        ('smote', SMOTE()),
        ('classifier', RandomForestClassifier())
    ])

    # ハイパーパラメータの設定範囲
    param_grid = {
        'smote__k_neighbors': [3, 5, 7],
        'smote__sampling_strategy': ['auto', 0.5, 0.7],
    }

    # StratifiedKFoldのインスタンスを作成
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # グリッドサーチ
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=stratified_kfold, scoring='f1_macro')
    grid_search.fit(X, y)

    # 最適なパラメータでSMOTEを適用
    best_params = grid_search.best_params_
    smote = SMOTE(
        k_neighbors=best_params['smote__k_neighbors'],
        sampling_strategy=best_params['smote__sampling_strategy'],
        random_state=random_state
    )

    X_resampled, y_resampled = smote.fit_resample(X, y)

    # SMOTE適用後のデータフレームを生成
    train_df_SMOTE = pd.DataFrame(X_resampled, columns=X.columns)
    train_df_SMOTE[target_column] = y_resampled
    return train_df_SMOTE

# 使用例（target_columnは目的変数の列名）
# train_df_SMOTE, best_params = smote_optimize_and_apply(train_df, 'target_column')



# 緯度経度をGoogleMapから算出するコード
# from googlemaps import Client as GoogleMaps
# gmaps = GoogleMaps('XXXXXX')
# lat_lng_cache = {}
# def get_lat_lng(zip_code):
#     # キャッシュ内に結果があるか確認
#     if zip_code in lat_lng_cache:
#         return lat_lng_cache[zip_code]

#     try:
#         geocode_result = gmaps.geocode('{}, USA'.format(zip_code))
#         lat = geocode_result[0]["geometry"]["location"]["lat"]
#         lng = geocode_result[0]["geometry"]["location"]["lng"]
#         lat_lng_cache[zip_code] = (lat, lng)
#         return lat, lng
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return None, None

# unique_zips = all_data['zip'].unique()
# lat_lng_dict = {}

# for zip_code in tqdm(unique_zips):
#     lat, lng = get_lat_lng(zip_code)
#     if lat and lng:
#         lat_lng_dict[zip_code] = {'lat': lat, 'lng': lng}

# # 緯度・経度情報をJSONに保存
# import json
# with open('lat_lng_cache.json', 'w') as f:
#     json.dump(lat_lng_dict, f)

