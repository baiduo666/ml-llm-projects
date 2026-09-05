# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import datetime
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.log import Logger
from utils.common import load_power_data
from sklearn.metrics import mean_absolute_error
import matplotlib.ticker as mick
import joblib
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['font.size'] = 15


def pred_feature_extract(row_history, pred_time, weather_info, all_weather_categories):
    hour = pred_time.hour
    month = pred_time.month
    weekday = pred_time.weekday()
    is_workday = 1 if weekday <= 4 else 0

    # 缺失历史负荷时默认用 0.0（比 600.0 更中性，避免引入偏差）
    DEFAULT_LAG = 0.0
    lag1 = row_history.get(pred_time - pd.Timedelta(minutes=15), DEFAULT_LAG)
    lag2 = row_history.get(pred_time - pd.Timedelta(minutes=30), DEFAULT_LAG)
    lag3 = row_history.get(pred_time - pd.Timedelta(minutes=45), DEFAULT_LAG)
    lag4 = row_history.get(pred_time - pd.Timedelta(minutes=60), DEFAULT_LAG)
    lag5 = row_history.get(pred_time - pd.Timedelta(minutes=75), DEFAULT_LAG)
    lag6 = row_history.get(pred_time - pd.Timedelta(minutes=90), DEFAULT_LAG)
    lag7 = row_history.get(pred_time - pd.Timedelta(minutes=105), DEFAULT_LAG)
    lag8 = row_history.get(pred_time - pd.Timedelta(minutes=120), DEFAULT_LAG)
    lag9 = row_history.get(pred_time - pd.Timedelta(minutes=135), DEFAULT_LAG)
    lag10 = row_history.get(pred_time - pd.Timedelta(minutes=150), DEFAULT_LAG)
    lag11 = row_history.get(pred_time - pd.Timedelta(minutes=165), DEFAULT_LAG)
    lag12 = row_history.get(pred_time - pd.Timedelta(minutes=180), DEFAULT_LAG)

    yesterday_load = row_history.get(pred_time - pd.Timedelta(days=1), DEFAULT_LAG)
    last_week_load = row_history.get(pred_time - pd.Timedelta(days=7), DEFAULT_LAG)

    t_max = float(weather_info["temp_max"])
    t_min = float(weather_info["temp_min"])
    temp_diff = t_max - t_min

    # 基础数值特征
    base_dict = {
        "temp_max": [t_max],
        "temp_min": [t_min],
        "temp_diff": [temp_diff],
        "hour_sin": [np.sin(2 * np.pi * hour / 24)],
        "hour_cos": [np.cos(2 * np.pi * hour / 24)],
        "month_sin": [np.sin(2 * np.pi * month / 12)],
        "month_cos": [np.cos(2 * np.pi * month / 12)]
    }

    # 小时独热
    for h in range(24):
        base_dict[f"hour_{h}"] = [1 if h == hour else 0]
    # 月份独热
    for m in range(1, 13):
        base_dict[f"month_{m}"] = [1 if m == month else 0]
    # 工作日
    base_dict["workday_0"] = [1 if is_workday == 0 else 0]
    base_dict["workday_1"] = [1 if is_workday == 1 else 0]

    # 天气独热
    weather_now = weather_info["weather"]
    for cate in all_weather_categories:
        col_name = f"weather_{cate}"
        base_dict[col_name] = [1 if cate == weather_now else 0]

    # 12个时序滞后负荷
    base_dict["前1个时段负荷"] = [lag1]
    base_dict["前2个时段负荷"] = [lag2]
    base_dict["前3个时段负荷"] = [lag3]
    base_dict["前4个时段负荷"] = [lag4]
    base_dict["前5个时段负荷"] = [lag5]
    base_dict["前6个时段负荷"] = [lag6]
    base_dict["前7个时段负荷"] = [lag7]
    base_dict["前8个时段负荷"] = [lag8]
    base_dict["前9个时段负荷"] = [lag9]
    base_dict["前10个时段负荷"] = [lag10]
    base_dict["前11个时段负荷"] = [lag11]
    base_dict["前12个时段负荷"] = [lag12]

    # 日滞后负荷
    base_dict["yesterday_load"] = [yesterday_load]
    base_dict["last_week_load"] = [last_week_load]

    df = pd.DataFrame(base_dict)
    return df


def prediction_plot(data):
    if len(data) == 0:
        return
    fig = plt.figure(figsize=(30, 12))
    ax = fig.add_subplot()
    ax.plot(data['时间'], data['真实值'], label='真实值')
    ax.plot(data['时间'], data['预测值'], label='预测值')
    ax.set_ylabel('总有功功率负荷')
    ax.set_title('电力负荷预测值与真实值对比')
    ax.xaxis.set_major_locator(mick.MultipleLocator(100))
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    (DATA_DIR / "fig").mkdir(parents=True, exist_ok=True)
    plt.savefig(DATA_DIR / "fig" / "预测效果图.png")


class PowerLoadPredict(object):
    def __init__(self):
        logfile_name = "predict_" + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.logfile = Logger(str(PROJECT_ROOT), logfile_name).get_logger()
        path_load = DATA_DIR / "附件1-区域15分钟负荷数据.csv"
        path_industry = DATA_DIR / "附件2-行业日负荷数据.csv"
        path_weather = DATA_DIR / "附件3-气象数据.csv"
        self.df_full = load_power_data(path_load, path_industry, path_weather)
        self.history_dict = self.df_full.set_index("time")["power_load"].to_dict()
        self.all_weather = self.df_full["weather"].dropna().unique()
        # 读取训练时固定的特征顺序
        with (MODEL_DIR / "feature_order.json").open("r", encoding="utf-8") as f:
            self.train_feature_order = json.load(f)

    def run_predict(self):
        model = joblib.load(MODEL_DIR / "xgb_model.pkl")
        evaluate_list = []
        test_samples = self.df_full.tail(300).reset_index(drop=True)
        for idx, row in test_samples.iterrows():
            t = row["time"]
            history = {k: v for k, v in self.history_dict.items() if k < t}
            w_info = {
                "temp_max": row["temp_max"],
                "temp_min": row["temp_min"],
                "weather": row["weather"]
            }
            x_df = pred_feature_extract(history, t, w_info, self.all_weather)
            # 强制对齐训练特征顺序
            x_df = x_df[self.train_feature_order]
            pred_val = model.predict(x_df)[0]
            true_val = row["power_load"]
            evaluate_list.append([t, true_val, pred_val])

        evaluate_df = pd.DataFrame(evaluate_list, columns=["时间", "真实值", "预测值"])
        mae_score = mean_absolute_error(evaluate_df["真实值"], evaluate_df["预测值"])
        print(f"滚动预测MAE：{mae_score}")
        self.logfile.info(f"滚动预测MAE：{mae_score}")
        prediction_plot(evaluate_df)


if __name__ == '__main__':
    pred_obj = PowerLoadPredict()
    pred_obj.run_predict()
