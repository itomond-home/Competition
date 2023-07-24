import pandas as pd
import numpy as np

def fill_missing_values(df):
    # 最頻値で欠損値を補完
    # df['fuel'] = df['fuel'].fillna(df['fuel'].mode()[0])

    # 定数値（'Unknown'）で欠損値を補完
    df['fuel'] = df['fuel'].fillna('Unknown')
    df['state'] = df['state'].fillna('Unknown')

    # 'Missing'という新たなカテゴリで欠損値を補完
    df['title_status'] = df['title_status'].fillna('Missing')
    df['type'] = df['type'].fillna('Missing')

    return df

def create_missing_flags(df):
    for column in ['title_status', 'type']:
        df[column + '_missing'] = df[column].isna().astype(int)
    return df

def extract_num_cylinders(df):
    # Extract numerical values and convert them to float
    df['num_cylinders'] = df['cylinders'].str.extract('(\d+)', expand=False).astype(float)
    # Replace 'other' with NaN
    df.loc[df['cylinders'] == 'other', 'num_cylinders'] = np.nan
    return df

def order_condition(df):
    condition_order = {'new': 5, 'like new': 4, 'excellent': 3, 'good': 2, 'fair': 1, 'salvage': 0}
    df['condition_order'] = df['condition'].map(condition_order)
    return df

def correct_year(df):
    df.loc[df['year'] > 3000, 'year'] = df.loc[df['year'] > 3000, 'year'] - 1000
    return df

def process_odometer(df):
    # 1. Create a missing value flag for -1
    df['odometer_is_missing'] = df['odometer'] == -1
    # 2. If the odometer value is less than -1, flip it to positive
    df.loc[df['odometer'] < -1, 'odometer'] = abs(df.loc[df['odometer'] < -1, 'odometer'])
    # 3. Create a flag for odometer values greater than 1 million
    df['odometer_over_1M'] = df['odometer'] > 1000000
    return df

def create_car_age_and_average_mileage(df):
    # Car age calculation
    df['car_age'] = pd.Timestamp.now().year - df['year']
    # Average mileage per year calculation
    df['average_mileage_per_year'] = df['odometer'] / df['car_age']
    df.loc[df['odometer'] == -1, 'average_mileage_per_year'] = -1
    return df


def apply_feature_engineering(df):
    df = fill_missing_values(df) # ほぼ必須
    df = process_odometer(df)
    df = correct_year(df)
    df = create_missing_flags(df)
    df = extract_num_cylinders(df)
    df = order_condition(df)
    df = create_car_age_and_average_mileage(df)

    return df
