# -*- coding: utf-8 -*-
import sys
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import datetime
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QTextEdit,
                             QVBoxLayout, QHBoxLayout, QWidget, QLabel, QFileDialog, QGroupBox)
from PyQt5.QtGui import QFont
from PyQt5 import QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.ticker as mick

# 添加项目根目录用于导入自定义工具模块
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils.log import Logger
from utils.common import load_power_data

# 设置绘图全局参数
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['font.size'] = 12


class MplCanvas(FigureCanvasQTAgg):
    """
    Matplotlib绘图画布，用于嵌入Qt窗口内展示图表
    """
    def __init__(self, width=6, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


def feature_engineering(data, logger):
    """
    完整特征工程处理函数
    构造时间周期特征、时序滞后特征、气象衍生特征与独热编码特征
    :param data: 融合后的完整原始数据集
    :param logger: 日志记录对象
    :return: result:添加全部特征后的数据集，feature_names:特征名称列表
    """
    logger.info("===============开始进行特征工程处理===============")
    result = data.copy(deep=True)
    logger.info("===============提取基础时间特征===================")

    # 提取基础时间字段
    result['hour'] = result['time'].dt.hour
    result['month'] = result['time'].dt.month
    result['week_day'] = result['time'].dt.weekday
    result['is_workday'] = result['week_day'].apply(lambda x: 1 if x <= 4 else 0)

    # 三角函数周期特征
    result['hour_sin'] = np.sin(2 * np.pi * result['hour'] / 24)
    result['hour_cos'] = np.cos(2 * np.pi * result['hour'] / 24)
    result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
    result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)

    # 类别独热编码
    hour_encoding = pd.get_dummies(result['hour'], prefix="hour")
    month_encoding = pd.get_dummies(result['month'], prefix="month")
    workday_encoding = pd.get_dummies(result['is_workday'], prefix="workday")
    weather_encoding = pd.get_dummies(result['weather'], prefix="weather")

    logger.info("==============提取时序滞后负荷特征====================")
    # 前12个15分钟时段历史负荷
    window_size = 12
    shift_list = [result['power_load'].shift(i) for i in range(1, window_size + 1)]
    shift_data = pd.concat(shift_list, axis=1)
    shift_data.columns = [f'前{i}个时段负荷' for i in range(1, window_size + 1)]

    # 昨日同一时刻负荷
    result["time_lag_1d"] = result["time"] - pd.Timedelta(days=1)
    lag_dict = result.set_index("time")["power_load"].to_dict()
    result["yesterday_load"] = result["time_lag_1d"].map(lag_dict)

    # 上周同一时刻负荷
    result["time_lag_7d"] = result["time"] - pd.Timedelta(days=7)
    lag_7dict = result.set_index("time")["power_load"].to_dict()
    result["last_week_load"] = result["time_lag_7d"].map(lag_7dict)

    # 昼夜温差衍生特征
    result["temp_diff"] = result["temp_max"] - result["temp_min"]

    # 拼接所有特征
    feature_parts = [
        result[["temp_max", "temp_min", "temp_diff", "hour_sin", "hour_cos", "month_sin", "month_cos"]],
        hour_encoding,
        month_encoding,
        workday_encoding,
        weather_encoding,
        shift_data,
        result[["yesterday_load", "last_week_load"]]
    ]
    result = pd.concat(feature_parts, axis=1)
    result["power_load"] = data["power_load"]
    # 删除存在缺失值的样本
    result.dropna(axis=0, inplace=True)

    feature_names = list(result.columns)
    feature_names.remove("power_load")
    logger.info(f"最终特征总数量：{len(feature_names)}")
    return result, feature_names


def pred_feature_extract(row_history, pred_time, weather_info, all_weather_categories):
    """
    滚动预测单样本特征构造函数
    根据指定时间点、历史负荷、气象信息生成与训练集结构一致的特征向量
    :param row_history: 历史负荷字典 {时间:负荷值}
    :param pred_time: 当前预测时间
    :param weather_info: 当前时间气象信息字典
    :param all_weather_categories: 训练集中全部天气类别
    :return: 单样本特征DataFrame
    """
    hour = pred_time.hour
    month = pred_time.month
    weekday = pred_time.weekday()
    is_workday = 1 if weekday <= 4 else 0

    # 近3小时15分钟间隔历史负荷（缺失时默认 0.0）
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

    # 小时独热编码
    for h in range(24):
        base_dict[f"hour_{h}"] = [1 if h == hour else 0]
    # 月份独热编码
    for m in range(1, 13):
        base_dict[f"month_{m}"] = [1 if m == month else 0]
    # 工作日独热编码
    base_dict["workday_0"] = [1 if is_workday == 0 else 0]
    base_dict["workday_1"] = [1 if is_workday == 1 else 0]

    # 天气独热编码
    weather_now = weather_info["weather"]
    for cate in all_weather_categories:
        col_name = f"weather_{cate}"
        base_dict[col_name] = [1 if cate == weather_now else 0]

    # 时序滞后负荷特征
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


class MainWindow(QMainWindow):
    """
    GUI主窗口类，提供数据加载、模型训练、滚动预测可视化交互功能
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("电力负荷预测可视化操作界面")
        self.resize(1200, 800)
        self.df_full = None
        self.feature_names = None
        self.logfile = None

        self.init_ui()

    def init_ui(self):
        """初始化窗口布局、按钮、绘图面板与日志框"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        btn_layout = QHBoxLayout()
        self.btn_load = QPushButton("1.加载原始数据集")
        self.btn_train = QPushButton("2.模型训练")
        self.btn_predict = QPushButton("3.滚动预测绘图")
        self.btn_clear = QPushButton("清空日志与画布")

        self.btn_load.clicked.connect(self.load_data)
        self.btn_train.clicked.connect(self.train_model)
        self.btn_predict.clicked.connect(self.run_rolling_predict)
        self.btn_clear.clicked.connect(self.clear_all)

        btn_layout.addWidget(self.btn_load)
        btn_layout.addWidget(self.btn_train)
        btn_layout.addWidget(self.btn_predict)
        btn_layout.addWidget(self.btn_clear)

        canvas_layout = QHBoxLayout()
        self.canvas_eda = MplCanvas(width=6, height=4)
        self.canvas_pred = MplCanvas(width=6, height=4)
        canvas_layout.addWidget(self.canvas_eda)
        canvas_layout.addWidget(self.canvas_pred)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        font = QFont("Consolas", 10)
        self.log_text.setFont(font)

        main_layout.addLayout(btn_layout)
        main_layout.addLayout(canvas_layout)
        main_layout.addWidget(QLabel("运行日志："))
        main_layout.addWidget(self.log_text)

    def print_log(self, msg):
        """向界面日志框追加信息"""
        time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_text.append(f"[{time_str}] {msg}")
        QtCore.QCoreApplication.processEvents()

    def load_data(self):
        """打开文件夹选择框，加载三份原始CSV数据集并融合"""
        folder = QFileDialog.getExistingDirectory(self, "选择data数据集文件夹")
        if not folder:
            self.print_log("未选择文件夹，取消加载")
            return
        folder = os.path.normpath(folder)
        self.print_log(f"选中数据集文件夹：{folder}")

        path_load = os.path.join(folder, "附件1-区域15分钟负荷数据.csv")
        path_ind = os.path.join(folder, "附件2-行业日负荷数据.csv")
        path_wea = os.path.join(folder, "附件3-气象数据.csv")

        exist1 = os.path.exists(path_load)
        exist2 = os.path.exists(path_ind)
        exist3 = os.path.exists(path_wea)
        if not all([exist1, exist2, exist3]):
            self.print_log("文件夹内数据文件不完整，请检查文件名称")
            return

        logfile_name = "gui_" + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.logfile = Logger(str(PROJECT_ROOT), logfile_name).get_logger()
        try:
            self.df_full = load_power_data(path_load, path_ind, path_wea)
            self.print_log(f"数据加载完成，总样本量：{len(self.df_full)}")
            self.draw_eda()
        except Exception as e:
            self.print_log(f"数据加载失败：{str(e)}")

    def draw_eda(self):
        """绘制小时平均负荷EDA图"""
        if self.df_full is None:
            return
        ax = self.canvas_eda.axes
        ax.clear()
        data = self.df_full.copy()
        data['hour'] = data['time'].dt.hour
        data_hour_avg = data.groupby(by='hour', as_index=False)['power_load'].mean()
        ax.plot(data_hour_avg['hour'], data_hour_avg['power_load'], color='b', linewidth=2)
        ax.set_title('各小时平均负荷趋势图')
        ax.set_xlabel('小时')
        ax.set_ylabel('负荷')
        self.canvas_eda.draw()

    def train_model(self):
        """执行完整特征工程、数据集时序分割与XGBoost模型训练"""
        if self.df_full is None or self.logfile is None:
            self.print_log("请先加载数据集！")
            return
        try:
            processed_data, self.feature_names = feature_engineering(self.df_full, self.logfile)
            self.print_log(f"特征工程完成，可用样本数量：{len(processed_data)}")

            x_data = processed_data[self.feature_names]
            y_data = processed_data['power_load']
            split_idx = int(len(x_data) * 0.7)
            x_train = x_data.iloc[:split_idx, :]
            x_test = x_data.iloc[split_idx:, :]
            y_train = y_data.iloc[:split_idx]
            y_test = y_data.iloc[split_idx:]

            xgb = XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=22)
            xgb.fit(x_train, y_train)

            y_pred_train = xgb.predict(x_train)
            y_pred_test = xgb.predict(x_test)
            mse_train = mean_squared_error(y_true=y_train, y_pred=y_pred_train)
            mae_train = mean_absolute_error(y_true=y_train, y_pred=y_pred_train)
            mse_test = mean_squared_error(y_true=y_test, y_pred=y_pred_test)
            mae_test = mean_absolute_error(y_true=y_test, y_pred=y_pred_test)

            self.print_log(f"训练集 MSE:{mse_train:.2f}, MAE:{mae_train:.2f}")
            self.print_log(f"测试集 MSE:{mse_test:.2f}, MAE:{mae_test:.2f}")
            self.logfile.info(f"训练集 MSE:{mse_train}, MAE:{mae_train}")
            self.logfile.info(f"测试集 MSE:{mse_test}, MAE:{mae_test}")

            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(xgb, MODEL_DIR / "xgb_model.pkl")
            with (MODEL_DIR / "feature_order.json").open("w", encoding="utf-8") as f:
                json.dump(self.feature_names, f, ensure_ascii=False)
            self.print_log("模型文件与特征配置文件已保存至model目录")
        except Exception as e:
            self.print_log(f"模型训练异常：{str(e)}")

    def run_rolling_predict(self):
        """读取保存的模型，执行末尾300条样本滚动预测并绘图对比"""
        if self.df_full is None:
            self.print_log("请先加载数据集！")
            return
        if not (MODEL_DIR / "xgb_model.pkl").exists():
            self.print_log("未检测到训练模型，请先执行模型训练！")
            return
        try:
            model = joblib.load(MODEL_DIR / "xgb_model.pkl")
            with (MODEL_DIR / "feature_order.json").open("r", encoding="utf-8") as f:
                train_feature_order = json.load(f)
            history_dict = self.df_full.set_index("time")["power_load"].to_dict()
            all_weather = self.df_full["weather"].dropna().unique()
            test_samples = self.df_full.tail(300).reset_index(drop=True)
            evaluate_list = []

            # 预排序时间键，用于 bisect 快速查找历史数据
            sorted_times = sorted(history_dict.keys())
            for idx, row in test_samples.iterrows():
                t = row["time"]
                # 用 bisect 快速找到 t 之前的时间点，避免 O(n²) 全量过滤
                import bisect
                pos = bisect.bisect_left(sorted_times, t)
                valid_times = sorted_times[:pos]
                history = {k: history_dict[k] for k in valid_times}
                w_info = {
                    "temp_max": row["temp_max"],
                    "temp_min": row["temp_min"],
                    "weather": row["weather"]
                }
                x_df = pred_feature_extract(history, t, w_info, all_weather)
                x_df = x_df[train_feature_order]
                pred_val = model.predict(x_df)[0]
                true_val = row["power_load"]
                evaluate_list.append([t, true_val, pred_val])

            evaluate_df = pd.DataFrame(evaluate_list, columns=["时间", "真实值", "预测值"])
            mae_score = mean_absolute_error(evaluate_df["真实值"], evaluate_df["预测值"])
            self.print_log(f"滚动预测MAE：{mae_score:.2f}")
            if self.logfile is not None:
                self.logfile.info(f"滚动预测MAE：{mae_score}")

            ax = self.canvas_pred.axes
            ax.clear()
            ax.plot(evaluate_df['时间'], evaluate_df['真实值'], label='真实值')
            ax.plot(evaluate_df['时间'], evaluate_df['预测值'], label='预测值')
            ax.set_ylabel('总有功功率负荷')
            ax.set_title('电力负荷预测值与真实值对比')
            ax.xaxis.set_major_locator(mick.MultipleLocator(50))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            ax.legend()
            self.canvas_pred.draw()
        except Exception as e:
            self.print_log(f"滚动预测异常：{str(e)}")

    def clear_all(self):
        """清空界面日志与两张绘图画布"""
        self.log_text.clear()
        self.canvas_eda.axes.clear()
        self.canvas_pred.axes.clear()
        self.canvas_eda.draw()
        self.canvas_pred.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
