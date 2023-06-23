import re
import pandas as pd
import numpy as np
import hdbscan
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

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
    # Define the basic ratings for the weathers
    basic_ratings = {
        '晴': 2,
        '曇': -1,
        '雨': -2,
        '雪': -3,
        '霧': -3,
        '雷': -3,
        '屋内': 3
    }

    def rate_weather(weather_string):
        rating = 0
        for w, r in basic_ratings.items():
            if w in weather_string:
                rating += r
        # Convert the rating to 1-5 scale
        if rating <= -4: return 1
        if rating <= -1: return 2
        if rating == 0: return 3
        if rating <= 2: return 4
        return 5

    return weather.apply(rate_weather)

def clean_data(df):
    df['round'] = df['round'].str.replace('第', '').str.replace('日', '').astype(int)
    return df

def calculate_discomfort_index(df):
    # Calculate discomfort index
    df['discomfort_index'] = 0.81 * df['temperature'] + 0.01 * df['humidity'] * (0.99 * df['temperature'] - 14.3) + 46.3
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

    # Apply the new functions
    df = calculate_discomfort_index(df)
    df = generate_team_performance(df)

    df = clean_data(df)

    # Define function to get the two most frequent players
    def most_freq_players(df, team):
        player_cols = [f'{team}_player{i}' for i in range(1, 12)]
        players = df[player_cols].values.flatten()
        return pd.Series(players).value_counts().nlargest(2).index.tolist()

    # Get the two most frequent players for each team
    most_freq_home_players = df.groupby('home_team').apply(most_freq_players, 'home_team')
    most_freq_away_players = df.groupby('away_team').apply(most_freq_players, 'away_team')

    # Convert these to dictionaries for faster lookup
    most_freq_home_players_dict = most_freq_home_players.to_dict()
    most_freq_away_players_dict = most_freq_away_players.to_dict()

    # Define function to flag if top 2 most frequent players are in the match
    def flag_most_freq_players(row, team, most_freq_players):
        flags = [0, 0]
        for i in range(1, 12):
            if row[f'{team}_player{i}'] == most_freq_players[row[team]][0]:
                flags[0] = 1
            elif row[f'{team}_player{i}'] == most_freq_players[row[team]][1]:
                flags[1] = 1
            if all(flags):  # if both flags are 1, no need to continue the loop
                break
        return flags

    # Create new columns to flag if the top 2 most frequent players are in the match
    df[['most_freq_home_player_in_match', 'second_most_freq_home_player_in_match']] = df.apply(lambda row: flag_most_freq_players(row, 'home_team', most_freq_home_players_dict), axis=1, result_type='expand')
    df[['most_freq_away_player_in_match', 'second_most_freq_away_player_in_match']] = df.apply(lambda row: flag_most_freq_players(row, 'away_team', most_freq_away_players_dict), axis=1, result_type='expand')

    for i in range(1, 12):
        df = df.drop([f'home_team_player{i}'],axis=1)
        df = df.drop([f'away_team_player{i}'],axis=1)

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

    df['distance_category'] = df.apply(lambda row: get_distance_category(row['home_region'], row['away_region']), axis=1)

    df = df.drop(['home_prefecture', 'home_region', 'away_region'], axis=1)

    return df

def get_distance_category(home_region, away_region):
    # 隣接地方を定義
    adjacent_regions = {
        '北海道': ['東北'],
        '東北': ['北海道', '関東'],
        '関東': ['東北', '中部'],
        '中部': ['関東', '近畿'],
        '近畿': ['中部', '中国', '四国'],
        '中国': ['近畿', '四国', '九州'],
        '四国': ['近畿', '中国', '九州'],
        '九州': ['中国', '四国']
    }

    # 遠距離地方を定義
    distant_regions = {
        '北海道': ['関東', '中部', '近畿', '中国', '四国', '九州'],
        '東北': ['中部', '近畿', '中国', '四国', '九州'],
        '関東': ['中国', '四国', '九州'],
        '中部': ['四国', '九州'],
        '近畿': ['九州'],
        '中国': [],
        '四国': [],
        '九州': []
    }

    if home_region == away_region:
        return 1
    elif away_region in adjacent_regions[home_region]:
        return 2
    elif away_region in distant_regions[home_region]:
        return 3
    else:
        return 4


def generate_team_performance(df):
    # Define point rules
    points = {"win": 3, "draw": 1, "lose": 0}

    # Prepare dictionaries to keep track of each team's points, ranks, scores, wins, and losses
    team_points = {}
    team_ranks = {}
    team_scores_for = {}
    team_scores_against = {}
    team_wins = {}
    team_losses = {}
    last_team_ranks = {}
    last_year_final_ranks = {}
    last_year = df['match_date'].dt.year.min() - 1  # initialize with a year before the minimum
    last_3_matches_points = {}
    last_5_matches_points = {}
    last_3_matches_scores_for = {}
    last_5_matches_scores_for = {}
    last_3_matches_scores_against = {}

    for idx, row in df.iterrows():
        # Get team ids
        home_team_id = row['home_team']
        away_team_id = row['away_team']

        # Check if a new year has started
        current_year = row['match_date'].year
        if current_year != last_year:
            # Store last year's final ranks
            if last_team_ranks:
                last_year_final_ranks = last_team_ranks.copy()
            # Reset team points, ranks, scores, wins, and losses for the new year
            team_points = {}
            team_ranks = {}
            team_scores_for = {}
            team_scores_against = {}
            team_wins = {}
            team_losses = {}
            last_year = current_year

        # Calculate result
        if row['home_team_score'] > row['away_team_score']:
            home_team_points = points["win"]
            away_team_points = points["lose"]
            team_wins[home_team_id] = team_wins.get(home_team_id, 0) + 1
            team_losses[away_team_id] = team_losses.get(away_team_id, 0) - 1  # losses are represented as negative
        elif row['home_team_score'] < row['away_team_score']:
            home_team_points = points["lose"]
            away_team_points = points["win"]
            team_wins[away_team_id] = team_wins.get(away_team_id, 0) + 1
            team_losses[home_team_id] = team_losses.get(home_team_id, 0) - 1  # losses are represented as negative
        else:
            home_team_points = points["draw"]
            away_team_points = points["draw"]

        # Update team points
        team_points[home_team_id] = team_points.get(home_team_id, 0) + home_team_points
        team_points[away_team_id] = team_points.get(away_team_id, 0) + away_team_points

        # Update last 3 and 5 matches points
        if home_team_id in last_3_matches_points:
            last_3_matches_points[home_team_id].append(home_team_points)
            if len(last_3_matches_points[home_team_id]) > 3:
                last_3_matches_points[home_team_id].pop(0)
        else:
            last_3_matches_points[home_team_id] = [home_team_points]

        if home_team_id in last_5_matches_points:
            last_5_matches_points[home_team_id].append(home_team_points)
            if len(last_5_matches_points[home_team_id]) > 5:
                last_5_matches_points[home_team_id].pop(0)
        else:
            last_5_matches_points[home_team_id] = [home_team_points]

        if away_team_id in last_3_matches_points:
            last_3_matches_points[away_team_id].append(away_team_points)
            if len(last_3_matches_points[away_team_id]) > 3:
                last_3_matches_points[away_team_id].pop(0)
        else:
            last_3_matches_points[away_team_id] = [away_team_points]

        if away_team_id in last_5_matches_points:
            last_5_matches_points[away_team_id].append(away_team_points)
            if len(last_5_matches_points[away_team_id]) > 5:
                last_5_matches_points[away_team_id].pop(0)
        else:
            last_5_matches_points[away_team_id] = [away_team_points]

        # Calculate team ranks
        team_ranks = {team: rank for rank, (team, _) in enumerate(sorted(team_points.items(), key=lambda item: item[1], reverse=True), 1)}

        # Update team ranks
        last_team_ranks = team_ranks.copy()

        # Add team ranks and last year's final ranks to the data
        df.loc[idx, 'home_team_rank'] = team_ranks[home_team_id]
        df.loc[idx, 'away_team_rank'] = team_ranks[away_team_id]
        df.loc[idx, 'home_team_last_year_rank'] = last_year_final_ranks.get(home_team_id, -1)
        df.loc[idx, 'away_team_last_year_rank'] = last_year_final_ranks.get(away_team_id, -1)

        # Add rank differences and their absolute values to the data
        df.loc[idx, 'rank_diff'] = team_ranks[home_team_id] - team_ranks[away_team_id]
        df.loc[idx, 'rank_diff_abs'] = abs(df.loc[idx, 'rank_diff'])
        df.loc[idx, 'last_year_rank_diff'] = last_year_final_ranks.get(home_team_id, 0) - last_year_final_ranks.get(away_team_id, 0)
        df.loc[idx, 'last_year_rank_diff_abs'] = abs(df.loc[idx, 'last_year_rank_diff'])

        df['diff_score'] = np.abs(df['away_team_score'] - df['home_team_score'])

        # Update team scores
        team_scores_for[home_team_id] = team_scores_for.get(home_team_id, 0) + row['home_team_score']
        team_scores_for[away_team_id] = team_scores_for.get(away_team_id, 0) + row['away_team_score']

        team_scores_against[home_team_id] = team_scores_against.get(home_team_id, 0) + row['away_team_score']
        team_scores_against[away_team_id] = team_scores_against.get(away_team_id, 0) + row['home_team_score']

        # Update last 3 and 5 matches scores
        if home_team_id in last_3_matches_scores_for:
            last_3_matches_scores_for[home_team_id].append(row['home_team_score'])
            if len(last_3_matches_scores_for[home_team_id]) > 3:
                last_3_matches_scores_for[home_team_id].pop(0)
        else:
            last_3_matches_scores_for[home_team_id] = [row['home_team_score']]

        if home_team_id in last_5_matches_scores_for:
            last_5_matches_scores_for[home_team_id].append(row['home_team_score'])
            if len(last_5_matches_scores_for[home_team_id]) > 5:
                last_5_matches_scores_for[home_team_id].pop(0)
        else:
            last_5_matches_scores_for[home_team_id] = [row['home_team_score']]

        if away_team_id in last_3_matches_scores_for:
            last_3_matches_scores_for[away_team_id].append(row['away_team_score'])
            if len(last_3_matches_scores_for[away_team_id]) > 3:
                last_3_matches_scores_for[away_team_id].pop(0)
        else:
            last_3_matches_scores_for[away_team_id] = [row['away_team_score']]

        if away_team_id in last_5_matches_scores_for:
            last_5_matches_scores_for[away_team_id].append(row['away_team_score'])
            if len(last_5_matches_scores_for[away_team_id]) > 5:
                last_5_matches_scores_for[away_team_id].pop(0)
        else:
            last_5_matches_scores_for[away_team_id] = [row['away_team_score']]

        # Update last 3 matches scores against
        if home_team_id in last_3_matches_scores_against:
            last_3_matches_scores_against[home_team_id].append(row['away_team_score'])
            if len(last_3_matches_scores_against[home_team_id]) > 3:
                last_3_matches_scores_against[home_team_id].pop(0)
        else:
            last_3_matches_scores_against[home_team_id] = [row['away_team_score']]

        if away_team_id in last_3_matches_scores_against:
            last_3_matches_scores_against[away_team_id].append(row['home_team_score'])
            if len(last_3_matches_scores_against[away_team_id]) > 3:
                last_3_matches_scores_against[away_team_id].pop(0)
        else:
            last_3_matches_scores_against[away_team_id] = [row['home_team_score']]

        # Add averages of last 3 matches against scores to the data
        df.loc[idx, 'home_team_avg_conceded_last_3'] = sum(last_3_matches_scores_against[home_team_id]) / len(last_3_matches_scores_against[home_team_id]) if len(last_3_matches_scores_against[home_team_id]) > 0 else 0
        df.loc[idx, 'away_team_avg_conceded_last_3'] = sum(last_3_matches_scores_against[away_team_id]) / len(last_3_matches_scores_against[away_team_id]) if len(last_3_matches_scores_against[away_team_id]) > 0 else 0

        # Add team scores to the data
        df.loc[idx, 'home_team_scored'] = team_scores_for[home_team_id]
        df.loc[idx, 'away_team_scored'] = team_scores_for[away_team_id]
        df.loc[idx, 'home_team_conceded'] = team_scores_against[home_team_id]
        df.loc[idx, 'away_team_conceded'] = team_scores_against[away_team_id]

        # Add averages of last 3 and 5 matches points and scores to the data
        df.loc[idx, 'home_team_avg_points_last_3'] = sum(last_3_matches_points[home_team_id]) / len(last_3_matches_points[home_team_id]) if len(last_3_matches_points[home_team_id]) > 0 else 0
        df.loc[idx, 'home_team_avg_points_last_5'] = sum(last_5_matches_points[home_team_id]) / len(last_5_matches_points[home_team_id]) if len(last_5_matches_points[home_team_id]) > 0 else 0
        df.loc[idx, 'away_team_avg_points_last_3'] = sum(last_3_matches_points[away_team_id]) / len(last_3_matches_points[away_team_id]) if len(last_3_matches_points[away_team_id]) > 0 else 0
        df.loc[idx, 'away_team_avg_points_last_5'] = sum(last_5_matches_points[away_team_id]) / len(last_5_matches_points[away_team_id]) if len(last_5_matches_points[away_team_id]) > 0 else 0

        df.loc[idx, 'home_team_avg_scored_last_3'] = sum(last_3_matches_scores_for[home_team_id]) / len(last_3_matches_scores_for[home_team_id]) if len(last_3_matches_scores_for[home_team_id]) > 0 else 0
        df.loc[idx, 'home_team_avg_scored_last_5'] = sum(last_5_matches_scores_for[home_team_id]) / len(last_5_matches_scores_for[home_team_id]) if len(last_5_matches_scores_for[home_team_id]) > 0 else 0
        df.loc[idx, 'away_team_avg_scored_last_3'] = sum(last_3_matches_scores_for[away_team_id]) / len(last_3_matches_scores_for[away_team_id]) if len(last_3_matches_scores_for[away_team_id]) > 0 else 0
        df.loc[idx, 'away_team_avg_scored_last_5'] = sum(last_5_matches_scores_for[away_team_id]) / len(last_5_matches_scores_for[away_team_id]) if len(last_5_matches_scores_for[away_team_id]) > 0 else 0

        # Add winning and losing streaks to the data
        df.loc[idx, 'home_team_winning_streak'] = team_wins.get(home_team_id, 0)
        df.loc[idx, 'away_team_winning_streak'] = team_wins.get(away_team_id, 0)
        df.loc[idx, 'home_team_losing_streak'] = team_losses.get(home_team_id, 0)
        df.loc[idx, 'away_team_losing_streak'] = team_losses.get(away_team_id, 0)

    return df

def add_grouped_statistics(df):
    columns_to_group = ['temperature', 'discomfort_index', 'humidity', 'home_team_score', 'away_team_score', 'capacity', 'home_team_rank', 'away_team_rank']

    statistics = ['max', 'min', 'var', 'mean', 'sum']

    for col in columns_to_group:
        for stat in statistics:
            df_grouped = df.groupby('venue')[col].agg(stat).reset_index().rename(columns={col: f'{col}_{stat}_by_venue'})
            df = pd.merge(df, df_grouped, on='venue', how='left')

    return df

def HDBSCAN_featuring(df):
    selected_features = ['temperature', 'discomfort_index', 'humidity', 'home_team_score', 'away_team_score', 'capacity', 'home_team_rank', 'away_team_rank']
    non_binary_features = selected_features.copy()

    # HDBSCAN-based features
    df_copy = df[non_binary_features].copy()
    df_copy.reset_index(drop=True, inplace=True)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
    df['cluster'] = clusterer.fit_predict(df_copy)

    return df

def compute_knn_features_and_preprocess(train_df, test_df, target_col, k=3, folds=5):
    # Reset the index of train_df and test_df
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Combine into a single dataframe
    all_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

    # Separate features and target
    train_index = all_df[all_df[target_col] != -1].index
    test_index = all_df[all_df[target_col] == -1].index

    # One-hot encode categorical variables
    enc = OneHotEncoder(handle_unknown='ignore')
    all_X = pd.DataFrame(enc.fit_transform(all_df.drop(columns=[target_col])).toarray())

    # Scale features
    scaler = StandardScaler()
    all_X = pd.DataFrame(scaler.fit_transform(all_X), index=all_X.index)

    train_X = all_X.loc[train_index]
    test_X = all_X.loc[test_index]

    # Initialize KFold
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)

    # Compute KNN features for training data
    for train_index, _ in kf.split(train_X):
        X_fold = train_X.iloc[train_index]
        for i in range(1, k + 1):
            knn = NearestNeighbors(n_neighbors=i)
            knn.fit(X_fold)
            dist, _ = knn.kneighbors(train_X)
            all_df.loc[train_X.index, f'knn_avg_dist_{i}'] = dist.mean(axis=1)

    # Compute KNN features for test data
    for i in range(1, k + 1):
        knn = NearestNeighbors(n_neighbors=i)
        knn.fit(train_X)
        dist, _ = knn.kneighbors(test_X)
        all_df.loc[test_X.index, f'knn_avg_dist_{i}'] = dist.mean(axis=1)

    return all_df


def perform_target_encoding(columns, all_df, target, n_folds=5, seed=42):
    # Create a copy to not modify original data
    all_df_encoded = all_df.copy()

    # Create a new target encoded feature for each feature in columns
    for column in columns:
        # Create a new column initialized with 0
        all_df_encoded[f'{column}_target_enc'] = 0

        # Perform out-of-fold target encoding
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for train_index, valid_index in kf.split(all_df_encoded):
            # Create splits
            train_df, valid_df = all_df_encoded.iloc[train_index], all_df_encoded.iloc[valid_index]
            # Calculate out-of-fold means and map them to the validation data
            out_of_fold_means = valid_df[column].map(train_df.groupby(column)[target].mean())
            all_df_encoded.loc[valid_index, f'{column}_target_enc'] = out_of_fold_means

        # Fill NaNs with global mean
        all_df_encoded[f'{column}_target_enc'].fillna(all_df_encoded[target].mean(), inplace=True)
    return all_df_encoded
