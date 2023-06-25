# model_tuner.py
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from sklearn.feature_selection import VarianceThreshold
from statsmodels.stats.outliers_influence import variance_inflation_factor

def objective(trial, x_train, y_train, model_type, fixed_params):
    early_stopping_round = 500
    if model_type == 'cat':
        categorical_cols = x_train.select_dtypes(include=['object']).columns.tolist()
        cat_features = [x_train.columns.get_loc(col) for col in categorical_cols]
        params = {
            'depth': trial.suggest_int("depth", 4, 10),
            'learning_rate': trial.suggest_loguniform("learning_rate", 1e-2, 1.0),
            'random_strength': trial.suggest_int("random_strength", 0, 100),
            'bagging_temperature': trial.suggest_loguniform("bagging_temperature", 0.01, 100.00),
            'od_type': trial.suggest_categorical("od_type", ['IncToDec', 'Iter']),
            'od_wait': trial.suggest_int("od_wait", 10, 50)
        }

        params = {**fixed_params, **params}  # 事前に設定したパラメータと最適化したパラメータを組み合わせる

        kf = KFold(n_splits=10, shuffle=True, random_state=fixed_params['random_seed'])
        rmses = []
        for train_index, valid_index in kf.split(x_train):
            model = CatBoostRegressor(cat_features=cat_features, **params)
            model.fit(x_train.iloc[train_index], y_train.iloc[train_index], eval_set=[(x_train.iloc[valid_index], y_train.iloc[valid_index])], early_stopping_rounds=early_stopping_round, verbose_eval=False)
            preds = model.predict(x_train.iloc[valid_index])
            rmse = np.sqrt(mean_squared_error(y_train.iloc[valid_index], preds))
            rmses.append(rmse)
        return np.mean(rmses)

    elif model_type == 'lgb':
        categorical_cols = x_train.select_dtypes(include=['object','category']).columns.tolist()
        params = {
            'lambda_l1'         : trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
            'lambda_l2'         : trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
            'num_leaves'        : trial.suggest_int('num_leaves', 2, 512),
            'feature_fraction'  : trial.suggest_uniform('feature_fraction', 0.4, 1.0),
            'bagging_fraction'  : trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
            'bagging_freq'      : trial.suggest_int('bagging_freq', 0, 10),
            'min_child_samples' : trial.suggest_int('min_child_samples', 5, 100),
            'learning_rate': trial.suggest_loguniform("learning_rate", 1e-2, 1.0),
        }

        params = {**fixed_params, **params}  # 事前に設定したパラメータと最適化したパラメータを組み合わせる

        kf = KFold(n_splits=10, shuffle=True, random_state=fixed_params['seed'])
        rmses = []
        for train_index, valid_index in kf.split(x_train):
            x_train_fold, y_train_fold = x_train.iloc[train_index], y_train.iloc[train_index]
            x_valid_fold, y_valid_fold = x_train.iloc[valid_index], y_train.iloc[valid_index]
            # Convert categorical columns to 'category' dtype
            for col in categorical_cols:
                x_train_fold[col] = x_train_fold[col].astype('category')
                x_valid_fold[col] = x_valid_fold[col].astype('category')

            # Create LightGBM datasets
            train_data = lgb.Dataset(x_train_fold, label=y_train_fold, categorical_feature=categorical_cols)
            valid_data = lgb.Dataset(x_valid_fold, label=y_valid_fold, categorical_feature=categorical_cols)
            model = lgb.train(params, train_data, valid_sets=[valid_data], callbacks=[lgb.early_stopping(early_stopping_round, verbose=True), lgb.log_evaluation(False)])
            preds = model.predict(x_valid_fold)
            rmse = np.sqrt(mean_squared_error(y_valid_fold, preds))
            rmses.append(rmse)
        return np.mean(rmses)
    else:
        raise ValueError("model_type must be 'cat' or 'lgb'")

def tune_model(x_train, y_train, model_type, fixed_params, n_trials=100):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, x_train, y_train, model_type, fixed_params), n_trials=n_trials)

    print('Best trial:')
    trial = study.best_trial
    print('  Value: ', trial.value)
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    params = {**fixed_params, **trial.params}  # 最適化したパラメータを追加
    return params

def select_features_with_lightgbm(all_df, target_col, seed):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'dart',
        'n_jobs': -1,
        'seed': seed,
        'verbosity': -1,
        'force_row_wise': True,
    }
    train_df = all_df[all_df['attendance'] != -1]
    test_df = all_df[all_df['attendance'] == -1]

    # Separate features and target
    X = train_df.drop(columns=[target_col, 'id'], axis=1)
    y = train_df[target_col]

    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Convert categorical columns to 'category' dtype
    for col in categorical_cols:
        X[col] = X[col].astype('category')

    # Create LightGBM datasets
    train_data = lgb.Dataset(X, label=y, categorical_feature=categorical_cols, free_raw_data=False)

    # Fit model with Cross Validation
    cv_results = lgb.cv(
        params,
        train_data,
        num_boost_round=10000,
        nfold=5,
        early_stopping_rounds=100,
        stratified=False,
        seed=seed,
    )

    # Best iteration
    best_iteration = len(cv_results["rmse-mean"])

    # Train the model
    gbm = lgb.train(
        params,
        train_data,
        num_boost_round=best_iteration,
    )

    # Get feature importances
    importances = gbm.feature_importance(importance_type='gain')

    # Get indices of features with non-zero importance
    indices = np.where(importances > 0)[0]

    # Get selected features
    selected_features = X.columns[indices]

    # Retain selected features and target in the dataframe
    train_df_selected = train_df[selected_features.to_list() + [target_col, 'id']]
    test_df_selected = test_df[selected_features.to_list() + ['id']]

    return train_df_selected, test_df_selected


def calculate_vif_(X, thresh=10.0):
    # Calculating VIF
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix) for ix in range(X.iloc[:, variables].shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X.iloc[:, variables].columns[maxloc] + '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True

    print('Remaining variables:')
    print(X.columns[variables])
    return X.iloc[:, variables]

def select_features(df):
    not_features = ['id', 'attendance']  # Columns that should not be considered as features
    numeric_cols = df.drop(not_features, axis=1).select_dtypes(include=['int64', 'float64']).columns

    # Removing features with no variance
    selector = VarianceThreshold()
    df_numeric = df[numeric_cols].copy()
    selector.fit_transform(df_numeric)

    constant_columns = [column for column in df_numeric.columns
                        if column not in df_numeric.columns[selector.get_support()]]

    df.drop(constant_columns, axis=1, inplace=True)

    # Removing multicollinear features
    df_numeric = calculate_vif_(df_numeric)

    # Replace the original numeric columns with the selected numeric columns
    df.drop(numeric_cols, axis=1, inplace=True)
    df = pd.concat([df, df_numeric], axis=1)

    return df
