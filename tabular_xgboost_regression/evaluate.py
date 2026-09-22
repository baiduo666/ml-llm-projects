# -*- coding: utf-8 -*-
"""评估与可视化: 在验证集上计算 RMSE / MAE / R2, 并输出预测值对比与特征重要性图。"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parent
DATA, ASSETS = BASE / "data", BASE / "assets"
ASSETS.mkdir(exist_ok=True)

df = pd.read_csv(DATA / "train.csv")
X, y = df.drop(["id", "y"], axis=1), df["y"]
X["is_day"] = X["hour"].apply(lambda v: 1 if 7 <= v <= 22 else 0)

X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=417)
model = xgb.XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05,
                         objective="reg:squarederror", random_state=417)
model.fit(X_tr, y_tr)

pred = model.predict(X_va)
rmse = float(np.sqrt(mean_squared_error(y_va, pred)))
mae = float(mean_absolute_error(y_va, pred))
r2 = float(r2_score(y_va, pred))
print("验证集 RMSE=%.2f  MAE=%.2f  R2=%.4f" % (rmse, mae, r2))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(y_va, pred, s=6, alpha=0.35, color="#3b7dd8")
lo, hi = float(min(y_va.min(), pred.min())), float(max(y_va.max(), pred.max()))
axes[0].plot([lo, hi], [lo, hi], "r--", lw=1.2, label="理想预测")
axes[0].set_xlabel("真实值"); axes[0].set_ylabel("预测值")
axes[0].set_title("预测值 vs 真实值  (RMSE=%.2f, MAE=%.2f, R2=%.4f)" % (rmse, mae, r2))
axes[0].legend()

imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(12)[::-1]
axes[1].barh(imp.index, imp.values, color="#5aa469")
axes[1].set_title("特征重要性 Top 12")
plt.tight_layout()
out = ASSETS / "pred_vs_true.png"
plt.savefig(out, dpi=120)
print("已保存:", out)
