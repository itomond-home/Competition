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
