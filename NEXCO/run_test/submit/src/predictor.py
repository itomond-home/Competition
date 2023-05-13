import pandas as pd
import numpy as np
import pickle
import jpholiday
import datetime as dt
from tqdm import tqdm

class ScoringService(object):
    @classmethod
    def get_model(cls, model_path, inference_df):
        """Get model method

        Args:
            model_path (str): Path to the trained model directory.
            inference_df: Past data not subject to prediction.

        Returns:
            bool: The return value. True for success.
        """
        cls.allCars_model = pickle.load(open("/opt/ml/model/allCars_lgbm_42.pkl","rb"))
        cls.OCC_model = pickle.load(open("/opt/ml/model/OCC_lgbm_42.pkl","rb"))
        cls.search_1h_model = pickle.load(open("/opt/ml/model/search_1h_lgbm_42.pkl","rb"))
        cls.speed_model = pickle.load(open("/opt/ml/model/speed_lgbm_42.pkl","rb"))
        cls.data = inference_df

        return True

    @classmethod
    def predict(cls, input):
        """Predict method

        Args:
            input: meta data of the sample you want to make inference from (DataFrame)

        Returns:
            prediction: Inference for the given input. Return columns must be ['datetime', 'start_code', 'end_code'](DataFrame).

        Tips:
            You can use past data by writing "cls.data".
        """
        # 入力データの選択 予測に使ったデータのみにする
        input = input[[col for col in input.columns if col not in ["start_name","end_name"]]]
        # directionをダミー変数化 "direction_下り"になる
        input = pd.get_dummies(input, drop_first=True)

        def DateFeatGen(data):
            # 日付などの処理を行う関数
            data["year"] = data["year"].dt.year
            data["month"] = data["datetime"].dt.month
            data["day"] = data["datetime"].dt.day
            data["dayofweek"] = data["datetime"].dt.dayofweek
            data["hour"] = data["datetime"].dt.hour
            data["is_holiday"] = data["datetime"].map(jpholiday.is_holiday).astype(int)
            return data

        def engineer_time_series_features(data):
            outputs = [data]
            grp_df = data.groupby("start_code")[["OCC","allCars","search_1h"]]

            for lag in range(1,6):
                # shift
                outputs.append(grp_df.shift(lag).add_prefix(f'shift{lag}_'))
                # diff
                outputs.append(grp_df.diff(lag).add_prefix(f'diff{lag}_'))

            # rolling
            for window in [3]:
                tmp_df = grp_df.rolling(window, min_periods=1)
                # 移動平均を取る
                tmp_df = tmp_df.mean().add_prefix(f'rolling{window}_mean_')
                outputs.append(tmp_df.reset_index(drop=True))

            df = pd.concat(outputs, axis=1)
            return df

        # 予測をまとめるリストを作成
        predictions = []
        # "start_code" と "end_code" を１組として、順に処理していく
        for start_end in input[["start_code", "end_code"]].drop_duplicates().values:
            print(f"start_code is{start_end[0]}, end_code is {start_end[1]}")

            # "start_code" と "end_code" が同じデータのみに絞る つまりデータが24行になる
            data = input[(input["start_code"]==start_end[0]) & (input["end_code"]==start_end[1])].copy()

            # -23~0まで24時間分、以下を繰り返す
            for h in range(-23, 1): 

                # 直近24時間分にする
                data_tmp = data[-24:]
                
                # ラグ特徴量を生成する
                data_tmp = engineer_time_series_features(data_tmp)

                # 最後の時間のデータのみにする
                data_tmp = data[-1:].copy()
                # 日付を１日ずらして、データを再生成
                data_tmp["datetime"] += dt.timedelta(hours=1)
                # 提出用のデータ登録
                Onecode_pred = [(data_tmp["datetime"]).values[0],start_end[0],start_end[1]]
                
                # 日付の特徴量生成    
                data_tmp = DateFeatGen(data_tmp)

                # 現時刻のデータを１時間古いものとする（speed予測時のため）
                data_tmp["OCC_1h_old"] = data_tmp["OCC"].copy()
                data_tmp["allCars_1h_old"] = data_tmp["allCars"].copy()
                data_tmp["search_1h_old"] = data_tmp["search_1h"].copy()
                data_tmp["speed_1h_old"] = data_tmp["speed"].copy()

                # １時間後の"OCC","allCars","search_1h"を予測、格納
                unused_variable_names = ["datetime","inference_date","start_code","end_code","OCC","allCars","search_1h","speed"]
                data_tmp["OCC"] = 0 # OCC_model.predict(data_tmp[[col for col in data_tmp.columns if col not in unused_variable_names+["OCC"]]])
                data_tmp["allCars"] = 0 # allCars_model.predict(data_tmp[[col for col in data_tmp.columns if col not in unused_variable_names+["allCars"]]])
                data_tmp["search_1h"] = 0 # search_1h_model.predict(data_tmp[[col for col in data_tmp.columns if col not in unused_variable_names+["search_1h"]]])  

                data_tmp = DateFeatGen(data_tmp)
                # 現時刻の"speed"を予測、格納
                unused_variable_names = ["datetime","inference_date","start_code","end_code","speed"]
                pred_speed = 0 # speed_model.predict(data_tmp[[col for col in data_tmp.columns if col not in unused_variable_names]])  
                data_tmp["speed"] = pred_speed

                # 提出用データに格納
                Onecode_pred.append(pred_speed)
                # 大元の提出用データに格納
                predictions.append(Onecode_pred)
                # 新しい時刻をマージ
                data = pd.concat([data,data_tmp])
                # 予測に使う変数はいったん捨てる
                data = data.drop(["month","day","dayofweek","hour","is_holiday","OCC_1h_old","allCars_1h_old","search_1h_old","speed_1h_old"],axis=1).reset_index(drop=True)
                
        prediction = pd.DataFrame(predictions,columns=["datetime","start_code","end_code","prediction"])
        prediction["datetime"] -= dt.timedelta(days=1)
        return prediction
