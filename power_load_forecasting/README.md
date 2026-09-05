# 电力负荷预测

基于 **XGBoost** 的电力负荷预测应用,融合区域 15 分钟负荷、行业日用电、气象三份数据,提供数据探索、训练、预测与 PyQt5 图形界面。

## 功能
- **数据融合**:基于日期字段融合负荷 / 行业用电 / 气象数据(load_power_data)。
- **exploratory EDA**:特征分布与趋势可视化(ana_data)。
- **特征工程**:小时、月、星期、是否工作日,以及滞后负荷特征(pred_feature_extract)。
- **模型**:XGBoost 回归,输出 RMSE / MAE 评估。
- **三种用法**:训练(src/train.py)、批量预测(src/predict.py)、GUI 加载并预测(src/gui_load_predict.py)。

## 技术栈
Python · XGBoost · scikit-learn · Pandas · NumPy · Matplotlib · PyQt5 · joblib

## 运行
将三份数据放入 `data/`(参考 common.py 中的文件说明),然后执行:

```bash
pip install -r requirements.txt
python src/train.py      # 训练并保存模型
python src/predict.py    # 预测
python src/gui_load_predict.py  # 图形界面
```

## 说明
数据集与训练得到的模型二进制体积较大,未纳入仓库,请按 `data/` / `model/` 自行准备。
