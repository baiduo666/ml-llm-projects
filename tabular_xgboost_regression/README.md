# 表格回归

基于 **XGBRegressor** 的表格回归任务。

## 要点
- 特征工程:由 `hour` 派生 `is_day`(7–22 点标记为白天)。
- 模型:XGBoost 回归器(400 棵树, max_depth=5)。
- 评估:RMSE 与 MAE。

## 技术栈
Python · XGBoost · scikit-learn · Pandas · NumPy

## 运行
将 `train.csv` 与 `test.csv` 放入 `data/`,然后:

```bash
pip install -r requirements.txt
python train.py
```

## 说明
数据集体积较大,未纳入仓库,请自行准备 `data/train.csv` 与 `data/test.csv`。
