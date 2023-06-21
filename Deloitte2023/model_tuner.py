# model_tuner.py
import optuna
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from category_encoders import OneHotEncoder

def objective(trial, x_train, y_train, model_type):

    if model_type == "lightgbm":
        model = LGBMRegressor(
            num_leaves=trial.suggest_int("num_leaves", 2, 256),
            learning_rate=trial.suggest_loguniform("learning_rate", 1e-8, 1.0),
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            reg_alpha=trial.suggest_loguniform("reg_alpha", 1e-3, 10.0),
            reg_lambda=trial.suggest_loguniform("reg_lambda", 1e-3, 10.0),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            subsample=trial.suggest_loguniform("subsample", 0.2, 1.0),
            colsample_bytree=trial.suggest_loguniform("colsample_bytree", 0.2, 1.0),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
            random_state=42,
        )
    elif model_type == "xgboost":
        model = XGBRegressor(
            learning_rate=trial.suggest_loguniform("learning_rate", 1e-8, 1.0),
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            reg_alpha=trial.suggest_loguniform("reg_alpha", 1e-3, 10.0),
            reg_lambda=trial.suggest_loguniform("reg_lambda", 1e-3, 10.0),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            subsample=trial.suggest_loguniform("subsample", 0.2, 1.0),
            colsample_bytree=trial.suggest_loguniform("colsample_bytree", 0.2, 1.0),
            random_state=42,
        )
    elif model_type == "catboost":
        cat_indices = [i for i, col in enumerate(x_train.columns) if x_train[col].dtype == 'object']
        model = CatBoostRegressor(
            learning_rate=trial.suggest_loguniform("learning_rate", 1e-3, 1.0),
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            reg_lambda=trial.suggest_loguniform("reg_lambda", 1e-3, 10.0),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            subsample=trial.suggest_loguniform("subsample", 0.2, 1.0),
            colsample_bylevel=trial.suggest_loguniform("colsample_bylevel", 0.01, 1.0),
            cat_features=cat_indices,  # cat_indicesはカテゴリ変数のインデックスのリスト
            random_state=42,
        )
        score = cross_val_score(model, x_train, y_train, cv=10, scoring='neg_mean_squared_error')
        return score.mean()

    else:
        raise ValueError('Invalid model_type')

    # エンコーダとImputerの定義
    one_hot_enc = OneHotEncoder(cols=x_train.columns, handle_unknown='indicator')

    # パイプラインの作成
    pipeline = Pipeline([
        ('one_hot_enc', one_hot_enc),
        ('model', model)
    ])

    # 交差検証によるモデルの評価
    score = cross_val_score(pipeline, x_train, y_train, cv=10, scoring='neg_mean_squared_error')
    return score.mean()

def tune_model(x_train, y_train, model_type='lgb', n_trials=100):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, x_train, y_train, model_type), n_trials=n_trials)

    print('Best trial:')
    trial = study.best_trial
    print('  Value: ', trial.value)
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
