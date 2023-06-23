# model_tuner.py
import numpy as np
import pandas as pd
import optuna
import optuna.integration.lightgbm as lgbm
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from catboost import Pool
from catboost import CatBoostRegressor
from sklearn.feature_selection import RFECV



def objective(trial, x_train, y_train):
    categorical_cols = x_train.select_dtypes(include=['object']).columns.tolist()
    cat_features = [x_train.columns.get_loc(col) for col in categorical_cols]
    num_boost_round = 50500
    early_stopping_round = 500

    params = {
        'loss_function': 'RMSE',
        'iterations': num_boost_round,
        'random_seed': 42,
        'thread_count': -1,
        'depth': trial.suggest_int("depth", 4, 10),
        'learning_rate': trial.suggest_loguniform("learning_rate", 1e-2, 1.0),
        'random_strength': trial.suggest_int("random_strength", 0, 100),
        'bagging_temperature': trial.suggest_loguniform("bagging_temperature", 0.01, 100.00),
        'od_type': trial.suggest_categorical("od_type", ['IncToDec', 'Iter']),
        'od_wait': trial.suggest_int("od_wait", 10, 50)
    }

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    rmses = []
    for train_index, valid_index in kf.split(x_train):
        model = CatBoostRegressor(cat_features=cat_features, **params)
        model.fit(x_train.iloc[train_index], y_train.iloc[train_index], eval_set=[(x_train.iloc[valid_index], y_train.iloc[valid_index])], early_stopping_rounds=early_stopping_round, verbose_eval=False)
        preds = model.predict(x_train.iloc[valid_index])
        rmse = np.sqrt(mean_squared_error(y_train.iloc[valid_index], preds))
        rmses.append(rmse)
    return np.mean(rmses)


def tune_model(x_train, y_train, n_trials=100):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, x_train, y_train), n_trials=n_trials)

    print('Best trial:')
    trial = study.best_trial
    print('  Value: ', trial.value)
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
    best_params = trial.params

    params = {
        'loss_function': 'RMSE',
        'iterations': 50500,
        'random_seed': 42,
        'thread_count': -1,
        **best_params  # 最適化したパラメータを追加
        }

    return params

def select_features_with_lightgbm_rfecv(all_df, target_col, seed):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'dart',
        'n_jobs': -1,
        'seed': seed,
        'verbosity': -1,
        'force_row_wise': True,
    }
    train_df = all_df[all_df['attendance'] != -1].dropna()
    test_df = all_df[all_df['attendance'] == -1]

    # Separate features and target
    categorical_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
    X = train_df.drop(columns=[target_col, 'id'] + categorical_cols)
    y = train_df[target_col]

    # Initialize the LGBMRegressor
    model = lgb.LGBMRegressor(**params)

    # Create the RFE object and compute a cross-validated score
    rfecv = RFECV(estimator=model, step=1, cv=KFold(n_splits=10, random_state=seed, shuffle=True))
    rfecv.fit(X, y)

    # Print out the ranking of features
    for rank, feature in sorted(zip(rfecv.ranking_, X.columns)):
        print(f"{feature}: {rank}")

    # Select and print out the important features
    selected_features = [f for r, f in zip(rfecv.ranking_, X.columns) if r == 1]

    # Retain selected features and target in the dataframe
    train_df_selected = train_df[selected_features + [target_col, 'id'] + categorical_cols]
    test_df_selected = test_df[selected_features + ['id'] + categorical_cols]

    return train_df_selected, test_df_selected


def select_features(all_df, target_col, seed, params):
    train_df = all_df[all_df['attendance'] != -1].dropna()
    test_df = all_df[all_df['attendance'] == -1]

    # Identify categorical columns
    categorical_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Separate features and target
    X = train_df.drop(columns=[target_col, 'id'])
    y = train_df[target_col]

    # Initialize the CatBoostRegressor
    model = CatBoostRegressor(cat_features=categorical_cols, early_stopping_rounds=100, verbose=10000, **params)

    # Perform CV
    kf = KFold(n_splits=5, random_state=seed, shuffle=True)
    feature_importance_list = []
    for train_index, valid_index in kf.split(X):
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        train_pool = Pool(X_train, y_train, cat_features=categorical_cols)
        valid_pool = Pool(X_valid, y_valid, cat_features=categorical_cols)

        model.fit(train_pool, eval_set=valid_pool)

        # Get feature importance for each split
        feature_importance = model.get_feature_importance(valid_pool)
        feature_importance_list.append(feature_importance)

    # Average feature importance across all splits
    avg_feature_importance = np.mean(feature_importance_list, axis=0)

    # Get the feature importance in descending order
    importance_df = pd.DataFrame({'feature': X.columns, 'importance': avg_feature_importance})
    importance_df = importance_df.sort_values(by='importance', ascending=False)

    # Print the feature importance
    for idx, row in importance_df.iterrows():
        print(f"{row['feature']}: {row['importance']}")

    # Select and print out the important features
    selected_features = importance_df[importance_df['importance'] > 0]['feature'].tolist()  # You may adjust the threshold

    # Retain selected features and target in the dataframe
    train_df_selected = train_df[selected_features + [target_col, 'id']]
    test_df_selected = test_df[selected_features + ['id']]

    return train_df_selected, test_df_selected

def tune_parameters_lgbm(train_df, target_col, seed):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'dart',
        'n_jobs': -1,
        'seed': seed,
        'verbosity': -1,
        'force_row_wise': True,
    }

    # Separate features and target
    X = train_df.drop(columns=[target_col, 'id'])
    y = train_df[target_col]

    # Create LightGBM datasets
    train_data = lgb.Dataset(X, label=y)

    # Tune parameters
    tuner = lgbm.LightGBMTunerCV(params, train_data, verbose_eval=False, early_stopping_rounds=500, num_boost_round=20000, folds=KFold(n_splits=10, shuffle=True, random_state=seed))
    tuner.run()
    best_params = tuner.best_params

    # Merge tuned parameters and final parameters
    params.update(best_params)

    return params



