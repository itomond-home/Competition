# model_tuner.py
import numpy as np
import optuna
import optuna.integration.lightgbm as lgb
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
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
    # Separate features and target
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]

    # Identify categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    # Convert categorical features to category dtype
    for col in categorical_cols:
        X[col] = X[col].astype('category')

    # Define initial parameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
    }

    # Create LightGBM datasets
    train_data = lgb.Dataset(X, label=y, categorical_feature=categorical_cols)

    # Tune parameters
    tuner = lgb.LightGBMTunerCV(params, train_data, verbose_eval=False, early_stopping_rounds=500, num_boost_round=20000, folds=KFold(n_splits=5, shuffle=True, random_state=seed))
    tuner.run()
    best_params = tuner.best_params

    # Final parameters
    final_params = {
        'objective':'regression',
        'metric': 'rmse',
        'boosting': 'dart',
        'n_jobs': -1,
        'seed': seed,
    }

    # Merge tuned parameters and final parameters
    final_params.update(best_params)

    # Perform Recursive Feature Elimination
    dtrain = lgb.Dataset(X, label=y)
    gbm = lgb.train(final_params, dtrain)
    selector = RFECV(gbm, step=1, cv=5)
    selector = selector.fit(X, y)

    # Get selected features
    selected_features = X.columns[selector.support_]

    # Retain selected features and target in the dataframe
    train_df_selected = train_df[selected_features.to_list() + [target_col]]

    return train_df_selected, final_params
