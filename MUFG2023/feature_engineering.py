import pandas as pd
from googlemaps import Client as gmaps

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

    def fill_missing_zip(row):
        """実際に欠損を補完する関数"""
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
    # キャッシュ用の辞書
    lat_lng_cache = {}

    def get_lat_lng(zip_code):
        """緯度経度を引き出す関数"""
        # キャッシュ内に結果があるか確認
        if zip_code in lat_lng_cache:
            return lat_lng_cache[zip_code]

        geocode_result = gmaps.geocode('{}, USA'.format(zip_code))
        lat = geocode_result[0]["geometry"]["location"]["lat"]
        lng = geocode_result[0]["geometry"]["location"]["lng"]
        lat_lng_cache[zip_code] = (lat, lng)
        return lat, lng

    # DataFrameに新しい列として緯度と経度を追加
    df['shop_lat'], df['shop_lng'] = zip(*df['zip'].apply(get_lat_lng))
    return df

def apply_feature_engineering(df):
    df = removre_dollar_mark(df)
    df = fill_zipnumber(df)
    df = fill_zipnumber(df)
    # df = make_distance(df)
    # df = make_diffdate(df)
    # df = errors_count(df)
    # df = agg_unique_id(df)
    # df = flag_expires(df)
    # df = binnning_score(df)
    return df
