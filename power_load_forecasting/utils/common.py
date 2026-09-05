import pandas as pd
import numpy as np


def load_power_data(path_load, path_industry, path_weather):
    """
    加载电力负荷、行业用电、气象三份数据集并基于日期字段融合
    :param path_load: 区域短时负荷时序数据文件路径
    :param path_industry: 行业日用电统计数据文件路径
    :param path_weather: 气象环境数据文件路径
    :return: merge_df 融合后的完整样本数据集
    """
    df_load = pd.read_csv(path_load, encoding="utf-8-sig", header=0)
    df_load.columns = ["time", "power_load"]
    df_load["time"] = pd.to_datetime(df_load["time"], format="mixed", errors="coerce")
    df_load["date"] = df_load["time"].dt.strftime("%Y-%m-%d")
    df_load.sort_values("time", inplace=True)

    df_industry = pd.read_csv(path_industry, encoding="utf-8-sig", header=0)
    df_industry.columns = ["industry_type", "date", "big_industry", "min_power"]
    df_industry["date"] = pd.to_datetime(df_industry["date"], format="%Y/%m/%d", errors="coerce").dt.strftime("%Y-%m-%d")

    df_weather = pd.read_csv(path_weather, encoding="utf-8-sig", header=0)
    df_weather.columns = ["date", "weather", "temp_max", "temp_min", "wind_day", "wind_dir"]
    df_weather["date"] = pd.to_datetime(df_weather["date"], format="%Y年%m月%d日", errors="coerce").dt.strftime("%Y-%m-%d")

    # 清洗温度列，去除℃符号并转为数值
    # 先转为字符串再处理，避免纯数字列导致 AttributeError
    df_weather["temp_max"] = df_weather["temp_max"].astype(str).str.replace("℃", "", regex=False).astype(float)
    df_weather["temp_min"] = df_weather["temp_min"].astype(str).str.replace("℃", "", regex=False).astype(float)

    merge_df = pd.merge(df_load, df_weather, on="date", how="left")
    merge_df = pd.merge(merge_df, df_industry, on="date", how="left")

    merge_df["time_str"] = merge_df["time"].dt.strftime('%Y-%m-%d %H:%M:%S')
    merge_df.drop(columns=["date"], inplace=True)
    merge_df = merge_df.dropna(subset=["power_load"])
    return merge_df


def data_split_train_test(df, split_time_str):
    """
    基于时间节点切分训练集与测试集（时序切分，不随机打乱样本顺序）
    :param df: 完整融合数据集
    :param split_time_str: 时间分割节点字符串
    :return: train_df, test_df 训练集、测试集
    """
    split_time = pd.to_datetime(split_time_str)
    train_df = df[df["time"] < split_time].copy()
    test_df = df[df["time"] >= split_time].copy()
    return train_df, test_df