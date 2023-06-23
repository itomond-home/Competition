# model_tuner.py
import numpy as np
import pickle
import optuna
import optuna.integration.lightgbm as lgbm
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFECV
from catboost import CatBoostRegressor

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

def select_features_and_tune_parameters(train_df, target_col, seed):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'dart',
        'n_jobs': -1,
        'seed': seed,
        'verbosity': -1,
        'force_row_wise': True,
    }

    # Preprocessor
    # Load the encoders from a pickle file
    with open('encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)

    # Apply the encoding map to the relevant columns
    categorical_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
    for fold in range(5):
        for col in categorical_cols:
            train_df.loc[train_df['fold'] == fold, f'{col}'] = train_df.loc[train_df['fold'] == fold, col].map(encoders[f'fold_{fold}_{col}'])

    # Separate features and target
    X = train_df.drop(columns=[target_col, 'id'])
    y = train_df[target_col]

    # Print categorical columns in X
    categorical_cols_X = X.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"Categorical columns in X: {categorical_cols_X}")

    # Instantiate the model with the parameters
    gbm = lgb.LGBMRegressor(**params)

    # Perform Recursive Feature Elimination using RFECV
    rfecv = RFECV(estimator=gbm, step=1, cv=KFold(n_splits=10, shuffle=True, random_state=seed), scoring='neg_root_mean_squared_error', min_features_to_select=10, verbose=2)
    rfecv.fit(X, y)

    selected_features = features = train_df.columns.drop(['id', 'attendance'])[rfecv.support_]
    print(selected_features)

    # Retain selected features and target in the dataframe
    train_df_selected = train_df[selected_features + [target_col]]

    # Create LightGBM datasets
    train_data = lgb.Dataset(train_df_selected.drop(columns=[target_col]), label=train_df_selected[target_col])

    # Tune parameters
    tuner = lgbm.LightGBMTunerCV(params, train_data, verbose_eval=False, early_stopping_rounds=500,
                                num_boost_round=20000, folds=KFold(n_splits=10, shuffle=True, random_state=seed))
    tuner.run()
    best_params = tuner.best_params

    # Merge tuned parameters and final parameters
    params.update(best_params)

    return train_df_selected, params


