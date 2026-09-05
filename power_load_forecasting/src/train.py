# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import json
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.log import Logger
from utils.common import load_power_data, data_split_train_test
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['font.size'] = 15


def ana_data(data):
    """
    数据集综合探索性数据分析与可视化绘图
    :param data: 完整融合数据集
    """
    if len(data) == 0:
        print("数据集为空，跳过可视化")
        return
    data = data.copy(deep=True)
    print(data.info())
    print(data.head())
    fig = plt.figure(figsize=(20, 40))

    ax1 = fig.add_subplot(6, 1, 1)
    ax1.hist(data['power_load'], bins=100)
    ax1.set_title('区域总有功功率负荷分布直方图')

    ax2 = fig.add_subplot(6, 1, 2)
    data['hour'] = data['time'].dt.hour
    data_hour_avg = data.groupby(by='hour', as_index=False)['power_load'].mean()
    ax2.plot(data_hour_avg['hour'], data_hour_avg['power_load'], color='b', linewidth=2)
    ax2.set_title('各小时平均负荷趋势图')
    ax2.set_xlabel('小时')
    ax2.set_ylabel('负荷')

    ax3 = fig.add_subplot(6, 1, 3)
    data['week_day'] = data['time'].dt.weekday
    data['is_workday'] = data['week_day'].apply(lambda x: 1 if x <= 4 else 0)
    power_load_workday_avg = data[data['is_workday'] == 1]['power_load'].mean()
    power_load_holiday_avg = data[data['is_workday'] == 0]['power_load'].mean()
    ax3.bar(x=['工作日平均负荷', '周末平均负荷'], height=[power_load_workday_avg, power_load_holiday_avg])
    ax3.set_title('工作日与周末平均负荷对比')

    ax4 = fig.add_subplot(6, 1, 4)
    weather_type = data['weather'].dropna().unique()
    weather_mean = []
    for w in weather_type:
        sub = data[data['weather'] == w]
        weather_mean.append(sub['power_load'].mean())
    ax4.bar(weather_type, weather_mean)
    ax4.set_title('不同天气下平均负荷对比')
    plt.setp(ax4.get_xticklabels(), rotation=30)

    ax5 = fig.add_subplot(6, 1, 5)
    temp_valid = data.dropna(subset=["temp_max"])
    if len(temp_valid) > 0:
        sample = temp_valid.sample(min(2000, len(temp_valid)), random_state=1)
        ax5.scatter(sample["temp_max"], sample["power_load"], alpha=0.4)
    ax5.set_xlabel("最高温度")
    ax5.set_ylabel("区域负荷")
    ax5.set_title("最高温度与电力负荷散点关系")

    plt.tight_layout()
    (DATA_DIR / "fig").mkdir(parents=True, exist_ok=True)
    plt.savefig(DATA_DIR / "fig" / "综合数据分析图.png")


def feature_engineering(data, logger):
    """
    完整特征工程处理
    1.时间周期特征：小时、月份、工作日标识独热编码
    2.时序滞后特征：前12个时段负荷、昨日同一时刻、上周同一时刻负荷
    3.气象环境特征：温度、温差、天气状况独热编码
    :param data: 原始融合数据集
    :param logger: 日志记录对象
    :return: result, feature_names 特征数据集、特征名称列表
    """
    logger.info("===============开始进行特征工程处理===============")
    result = data.copy(deep=True)
    logger.info("===============提取基础时间特征===================")

    result['hour'] = result['time'].dt.hour
    result['month'] = result['time'].dt.month
    result['week_day'] = result['time'].dt.weekday
    result['is_workday'] = result['week_day'].apply(lambda x: 1 if x <= 4 else 0)

    # 新增：三角函数周期特征
    result['hour_sin'] = np.sin(2 * np.pi * result['hour'] / 24)
    result['hour_cos'] = np.cos(2 * np.pi * result['hour'] / 24)
    result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
    result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)

    hour_encoding = pd.get_dummies(result['hour'], prefix="hour")
    month_encoding = pd.get_dummies(result['month'], prefix="month")
    workday_encoding = pd.get_dummies(result['is_workday'], prefix="workday")
    weather_encoding = pd.get_dummies(result['weather'], prefix="weather")

    logger.info("==============提取时序滞后负荷特征====================")
    # 窗口扩大到前12个15分钟时段（3小时历史负荷）
    window_size = 12
    shift_list = [result['power_load'].shift(i) for i in range(1, window_size + 1)]
    shift_data = pd.concat(shift_list, axis=1)
    shift_data.columns = [f'前{i}个时段负荷' for i in range(1, window_size + 1)]

    # 昨日同一时刻
    result["time_lag_1d"] = result["time"] - pd.Timedelta(days=1)
    lag_dict = result.set_index("time")["power_load"].to_dict()
    result["yesterday_load"] = result["time_lag_1d"].map(lag_dict)

    # 新增：上周同一时刻（复用已构建的 lag_dict，避免重复 to_dict）
    result["time_lag_7d"] = result["time"] - pd.Timedelta(days=7)
    result["last_week_load"] = result["time_lag_7d"].map(lag_dict)

    # 新增气象衍生特征：昼夜温差
    result["temp_diff"] = result["temp_max"] - result["temp_min"]

    feature_parts = [
        # 基础数值特征 + 新增三角函数周期特征 + 温差
        result[["temp_max", "temp_min", "temp_diff", "hour_sin", "hour_cos", "month_sin", "month_cos"]],
        hour_encoding,
        month_encoding,
        workday_encoding,
        weather_encoding,
        shift_data,
        # 两个滞后日负荷
        result[["yesterday_load","last_week_load"]]
    ]
    result = pd.concat(feature_parts, axis=1)
    result["power_load"] = data["power_load"]

    result.dropna(axis=0, inplace=True)

    feature_names = list(result.columns)
    feature_names.remove("power_load")
    logger.info(f"最终特征列表：{feature_names}")
    return result, feature_names


def model_train(data, features, logger):
    """
    XGBoost模型训练、评估与保存
    :param data: 特征工程后数据集
    :param features: 特征名称列表
    :param logger: 日志记录对象
    """
    if len(data) == 0:
        logger.error("特征数据集为空，终止训练")
        return
    logger.info("=========开始模型训练===================")
    # 划分特征、标签
    x_data = data[features]
    y_data = data['power_load']

    # 时序分割
    split_idx = int(len(x_data) * 0.7)
    x_train = x_data.iloc[:split_idx, :]
    x_test = x_data.iloc[split_idx:, :]
    y_train = y_data.iloc[:split_idx]
    y_test = y_data.iloc[split_idx:]

    # 原最优固定参数，网格搜索反而效果变差
    xgb = XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=22)
    xgb.fit(x_train, y_train)

    y_pred_train = xgb.predict(x_train)
    y_pred_test = xgb.predict(x_test)

    mse_train = mean_squared_error(y_true=y_train, y_pred=y_pred_train)
    mae_train = mean_absolute_error(y_true=y_train, y_pred=y_pred_train)
    mse_test = mean_squared_error(y_true=y_test, y_pred=y_pred_test)
    mae_test = mean_absolute_error(y_true=y_test, y_pred=y_pred_test)

    print(f"训练集 MSE:{mse_train}, MAE:{mae_train}")
    print(f"测试集 MSE:{mse_test}, MAE:{mae_test}")
    logger.info(f"训练集 MSE:{mse_train}, MAE:{mae_train}")
    logger.info(f"测试集 MSE:{mse_test}, MAE:{mae_test}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(xgb, MODEL_DIR / "xgb_model.pkl")
    with (MODEL_DIR / "feature_order.json").open("w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False)
    logger.info("模型与特征顺序配置文件保存完成：../model/")

class PowerLoadModel(object):
    def __init__(self):
        logfile_name = "train_" + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.logfile = Logger(str(PROJECT_ROOT), logfile_name).get_logger()
        path_load = DATA_DIR / "附件1-区域15分钟负荷数据.csv"
        path_industry = DATA_DIR / "附件2-行业日负荷数据.csv"
        path_weather = DATA_DIR / "附件3-气象数据.csv"
        self.full_data = load_power_data(path_load, path_industry, path_weather)
        print(f"融合后总样本数量：{len(self.full_data)}")

    def run_train(self):
        processed_data, feature_cols = feature_engineering(self.full_data, self.logfile)
        model_train(processed_data, feature_cols, self.logfile)


if __name__ == '__main__':
    model = PowerLoadModel()
    full_df = model.full_data
    ana_data(full_df)
    if len(full_df) == 0:
        print("融合数据集为空，程序退出，请检查三份csv的日期范围是否重叠")
    else:
        model.run_train()
