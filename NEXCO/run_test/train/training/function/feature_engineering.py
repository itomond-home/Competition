import jpholiday
import pandas as pd
import numpy as np
import datetime as dt


def read_data(col):
    # road.csvを読み込んで不要な列を削除
    road = pd.read_csv("../road.csv").drop(["start_name", "end_name"], axis=1)
    # カテゴリ変数をダミー変数に変換
    road = pd.get_dummies(road, drop_first=True)

    # search_data.csvとsearch_unspec_data.csvを読み込んで必要な列をマージ
    search_data = pd.read_csv("../search_data.csv")
    search_data["datetime"] = pd.to_datetime(search_data["datetime"])
    search_data["date"] = search_data["datetime"].dt.date.astype(
        'datetime64[ns]')
    search_unspec_data = pd.read_csv("../search_unspec_data.csv")
    search_unspec_data["date"] = pd.to_datetime(search_unspec_data["date"])
    merged_search_data = search_data.merge(search_unspec_data[["date", "start_code", "end_code", "search_unspec_1d"]],
                                           on=["date", "start_code", "end_code"], how="left")

    # train.csvを読み込んで必要な列をマージ
    train = pd.read_csv("../train.csv")
    train["datetime"] = pd.to_datetime(train["datetime"])
    all_data = train.merge(road, on=["start_code", "end_code"], how="left")
    all_data = all_data.merge(merged_search_data, on=[
                              "datetime", "start_code", "end_code"], how="left")
    all_data = all_data.drop(["date"], axis=1)

    # 全車数が0の行を削除
    date_list = all_data.loc[all_data['allCars'].isin(
        [0])]['datetime'].dt.date.unique()
    all_data = all_data[~all_data['datetime'].dt.date.isin(date_list)]

    # start_codeとend_codeを連結して新たな変数を作成
    all_data["section"] = all_data["start_code"].astype(
        str) + all_data["end_code"].astype(str)

    all_data = all_data.drop(["is_congestion"], axis=1)

    all_data = reduce_mem_usage(all_data)

    all_data = TargetGenFunc(all_data, col)
    all_data = reduce_mem_usage(all_data)

    target_cols = ["speed_1h_old", "OCC_1h_old",
                   "search_1h_old", "allCars_1h_old"]
    all_data = DateFeatGen(all_data)
    all_data = reduce_mem_usage(all_data)

    window_size = [3, 6, 12]
    for group_column in ["section", "month"]:
        all_data = add_grouped_rolling_features(
            all_data, group_column, target_cols, window_size)
    all_data = reduce_mem_usage(all_data)

    shift_list = [1, 2, 3, 4, 5, 6, 12, 24]
    for group_column in ["section", "month"]:
        all_data = add_grouped_shift_diff_features(
            all_data, group_column, target_cols, shift_list)
    all_data = reduce_mem_usage(all_data)

    for group_column in ["section", "month"]:
        all_data = add_grouped_rank_features(
            all_data, group_column, target_cols)
    all_data = reduce_mem_usage(all_data)

    return all_data


def TargetGenFunc(data, col):
    num = -1
    if col == "speed":
        num = 1
    # 各説明変数の作成
    data = (data.groupby("start_code", group_keys=False).apply(
        lambda x: x.assign(speed_1h_old=x["speed"].shift(num))))
    data = (data.groupby("start_code", group_keys=False).apply(
        lambda x: x.assign(OCC_1h_old=x["OCC"].shift(num))))
    data = (data.groupby("start_code", group_keys=False).apply(
        lambda x: x.assign(search_1h_old=x["search_1h"].shift(num))))
    data = (data.groupby("start_code", group_keys=False).apply(
        lambda x: x.assign(allCars_1h_old=x["allCars"].shift(num))))
    return data


def DateFeatGen(data):
    # 日付などの処理
    data["month"] = data["datetime"].dt.month
    data["day"] = data["datetime"].dt.day
    data["dayofweek"] = data["datetime"].dt.dayofweek
    data["hour"] = data["datetime"].dt.hour
    data["is_holiday"] = data["datetime"].map(jpholiday.is_holiday).astype(int)
    return data


def add_grouped_shift_diff_features(df, group_column, target_cols, shift_vals):
    grouped_df = df.groupby(group_column)
    for target_col in target_cols:
        # shift_list内の数値ごとにシフト特徴量と差分特徴量を作成する
        for shift_val in shift_vals:
            shift_col_name = f"{target_col}_shift_{shift_val}"
            diff_col_name = f"{target_col}_diff_{shift_val}"
            df[shift_col_name] = grouped_df[target_col].shift(shift_val)
            df[diff_col_name] = grouped_df[target_col].diff(shift_val)
    return df


def add_grouped_rolling_features(df, group_column, target_cols, window_size):
    grouped_df = df.groupby(group_column, group_keys=False)
    for col in target_cols:
        # window_listに格納された窓幅の数分、以下の処理を繰り返す
        for window in window_size:
            rolling_col_name = col + "_rolling_mean_" + str(window)
            df[rolling_col_name] = grouped_df[col].rolling(
                window).mean().reset_index()[col].copy()
            rolling_col_name = col + "_rolling_std_" + str(window)
            df[rolling_col_name] = grouped_df[col].rolling(
                window).std().reset_index()[col].copy()
    return df


def add_grouped_rank_features(df, group_column, target_cols):
    for target_col in target_cols:
        grouped_df = df.groupby(group_column)
        # rank関数を使用して、グループ内での順位を計算する
        rank_df = grouped_df[target_col].rank(method='dense')
        df[f'{target_col}_{group_column}_rank'] = rank_df
    return df


def add_grouped_agg_features(df, group_column, target_col_pattern):
    # 変数group_columnを指定してグループ分けをする
    grouped_df = df.groupby(group_column)
    # dfの列名から、target_col_patternを含む列名を抽出する
    target_cols = [col for col in df.columns if target_col_pattern in col]
    # 対象となる変数に対して集約統計量を作成する
    agg_df = grouped_df[target_cols].agg(["mean", "max", "min", "std", "sum"])
    # 列名を変更する
    agg_df.rename(columns=lambda x: f"{x[0]}_{x[1]}", inplace=True)
    # 元のデータに結合する
    df = pd.concat([df, agg_df], axis=1)
    return df


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(
        100 * (start_mem - end_mem) / start_mem))

    return df
