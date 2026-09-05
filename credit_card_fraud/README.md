# 信用卡欺诈检测

基于 **XGBoost** 的信用卡欺诈检测(二分类),针对**严重类别不平衡**场景。

## 要点
- 调用 `compute_sample_weight` 计算类平衡样本权重,缓解少数类(欺诈)样本不足。
- 使用 `StratifiedKFold` + `GridSearchCV` 交叉验证与超参搜索,避免过拟合与偏差。
- 评估指标:ROC-AUC、混淆矩阵、classification_report。

## 技术栈
Python · XGBoost · scikit-learn · Pandas

## 运行
将 `creditcard.csv` 放入 `data/`,然后:

```bash
pip install -r requirements.txt
python train.py
```

## 说明
数据集体积较大,未纳入仓库,请自行下载 `creditcard.csv` 放入 `data/`。
