import pandas as pd
import numpy as np
import re

def process_periodic_features(df):
    df['match_date'] = pd.to_datetime(df['match_date'])
    df['year'] = df['match_date'].dt.year
    df['month'] = df['match_date'].dt.month
    df['day'] = df['match_date'].dt.day
    df['day_of_week'] = df['match_date'].dt.dayofweek

    df['kick_off_time'] = pd.to_datetime(df['kick_off_time'])
    df['hour'] = df['kick_off_time'].dt.hour

    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    df['section'] = df['section'].str.extract(r'(\d+)').astype(int)  # 節の数値部分のみを抽出し、int型に変換
    df['section_sin'] = np.sin(2 * np.pi * df['section'] / 38)
    df['section_cos'] = np.cos(2 * np.pi * df['section'] / 38)

    return df

def process_weather(weather):
    main_weathers = ['晴', '曇', '雨']  # 主要な天候
    return weather.apply(lambda x: x if x in main_weathers else 'その他')

def clean_data(df):
    df['round'] = df['round'].str.replace('第', '').str.replace('日', '').astype(int)
    return df

def apply_feature_engineering(df):
    # broadcastersを'/'で分割
    df['broadcasters'] = df['broadcasters'].str.split('/')

    # list形式を解除して全ての放送局を列挙
    broadcasters_list = df['broadcasters'].explode()

    # 最も頻度が高い放送局を抽出
    top_broadcasters = broadcasters_list.value_counts().index.tolist()

    # 例えば上位5つの放送局を新しい特徴として持つ
    for broadcaster in top_broadcasters[:20]:
        df[broadcaster] = df['broadcasters'].apply(lambda x: 1 if broadcaster in x else 0)

    # broadcasters列はもはや必要ないため削除
    df = df.drop('broadcasters', axis=1)

    df['holiday_flag'] = df['description'].notna().astype(int)

    # まず、祝日の前日と翌日が祝日であるかどうかをチェック
    df['prev_day_holiday'] = df['holiday_flag'].shift(fill_value=0)
    df['next_day_holiday'] = df['holiday_flag'].shift(-1, fill_value=0)

    # その日が祝日で、前日または翌日も祝日であれば、連休と見なす
    df['long_weekend_flag'] = ((df['holiday_flag'] == 1) & ((df['prev_day_holiday'] == 1) | (df['next_day_holiday'] == 1))).astype(int)

    # 不要な列を削除
    df.drop(['prev_day_holiday', 'next_day_holiday'], axis=1, inplace=True)
    df.drop(['holiday_date'], axis=1, inplace=True)

    df['weather'] = process_weather(df['weather'])

    df = clean_data(df)

    return df

def extract_prefecture(address):
    match = re.match(r"([^市区町村]+?[都道府県])", address)
    return match.group(1) if match else None

def add_geographical_features(df, venue_info_df):
    # 都道府県名とその都道府県が所属する地方のマッピングを作成します。
    pref_to_region = {
        '北海道': '北海道',
        '青森県': '東北', '岩手県': '東北', '秋田県': '東北', '宮城県': '東北', '山形県': '東北', '福島県': '東北',
        '茨城県': '関東', '栃木県': '関東', '群馬県': '関東', '埼玉県': '関東', '千葉県': '関東', '東京都': '関東', '神奈川県': '関東',
        '新潟県': '中部', '富山県': '中部', '石川県': '中部', '福井県': '中部', '山梨県': '中部', '長野県': '中部', '岐阜県': '中部', '静岡県': '中部', '愛知県': '中部',
        '三重県': '近畿', '滋賀県': '近畿', '奈良県': '近畿', '和歌山県': '近畿', '京都': '近畿', '大阪府': '近畿', '兵庫県': '近畿',
        '鳥取県': '中国', '島根県': '中国', '岡山県': '中国', '広島県': '中国', '山口県': '中国',
        '徳島県': '四国', '香川県': '四国', '愛媛県': '四国', '高知県': '四国',
        '福岡県': '九州', '佐賀県': '九州', '長崎県': '九州', '熊本県': '九州', '大分県': '九州', '宮崎県': '九州', '鹿児島県': '九州', '沖縄県': '九州'
    }

    team_prefecture = {
        '浦和': '埼玉県', '清水': '静岡県','大分': '大分県','福岡': '福岡県','C大阪': '大阪府','千葉': '千葉県','新潟': '新潟県','鹿島': '茨城県','京都': '京都','大宮': '埼玉県','川崎F': '神奈川県','横浜FM': '神奈川県','甲府': '山梨県','FC東京': '東京都','磐田': '静岡県','名古屋': '愛知県','広島': '広島県','G大阪': '大阪府','神戸': '兵庫県','横浜FC': '神奈川県','柏': '千葉県','札幌': '北海道','東京V': '東京都','山形': '山形県','仙台': '宮城県','湘南': '神奈川県','鳥栖': '佐賀県','徳島': '徳島県','松本': '長野県','長崎': '長崎県'
    }


    # スタジアムがどの都道府県に位置しているかを把握し、それを基に地方を定義します。
    venue_info_df['prefecture'] = venue_info_df['address'].apply(extract_prefecture)
    venue_info_df['region'] = venue_info_df['prefecture'].map(pref_to_region)

    # これらの情報を元に新たな特徴量を作成します。
    venue_to_prefecture = venue_info_df.set_index('venue')['prefecture'].to_dict()
    venue_to_region = venue_info_df.set_index('venue')['region'].to_dict()

    df['venue_prefecture'] = df['venue'].map(venue_to_prefecture)
    df['venue_region'] = df['venue'].map(venue_to_region)

    df['home_prefecture'] = df['home_team'].map(team_prefecture)
    df['home_region'] = df['home_prefecture'].map(pref_to_region)

    df['away_prefecture'] = df['away_team'].map(team_prefecture)
    df['away_region'] = df['away_prefecture'].map(pref_to_region)

    # 同じ都道府県内での試合か、同じ地方内での試合かを特徴量として追加します。
    df['home_same_prefecture'] = df['away_prefecture'] == df['venue_prefecture']
    df['home_same_region'] = df['home_region'] == df['venue_region']
    df['away_same_prefecture'] = df['away_prefecture'] == df['venue_prefecture']
    df['away_same_region'] = df['away_region'] == df['venue_region']

    df = df.drop(['home_prefecture', 'home_region', 'away_region'], axis=1)

    return df
