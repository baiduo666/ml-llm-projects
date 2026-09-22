# -*- coding: utf-8 -*-
"""评估与可视化: 输出 ROC-AUC、混淆矩阵与分类报告, 并保存结果图。"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             classification_report, precision_recall_curve, average_precision_score)
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parent
ASSETS = BASE / "assets"
ASSETS.mkdir(exist_ok=True)

df = pd.read_csv(BASE / "data" / "creditcard.csv")
X = df.drop(["Label", "ID", "V_Time"], axis=1)
y = df["Label"]
print("样本数 %d, 特征数 %d, 欺诈占比 %.4f%%" % (len(df), X.shape[1], 100 * y.mean()))

X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
w = compute_sample_weight(class_weight="balanced", y=y_tr)
model = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08,
                          objective="binary:logistic", eval_metric="auc", random_state=42)
model.fit(X_tr, y_tr, sample_weight=w)

proba = model.predict_proba(X_va)[:, 1]
pred = model.predict(X_va)
auc = float(roc_auc_score(y_va, proba))
ap = float(average_precision_score(y_va, proba))
print("验证集 ROC-AUC=%.4f  PR-AUC=%.4f" % (auc, ap))
print(classification_report(y_va, pred, digits=4))

cm = confusion_matrix(y_va, pred)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

fpr, tpr, _ = roc_curve(y_va, proba)
axes[0].plot(fpr, tpr, lw=1.6, color="#c0392b", label="AUC = %.4f" % auc)
axes[0].plot([0, 1], [0, 1], "k--", lw=1)
axes[0].set_xlabel("假正率 FPR"); axes[0].set_ylabel("真正率 TPR")
axes[0].set_title("ROC 曲线"); axes[0].legend(loc="lower right")

pre, rec, _ = precision_recall_curve(y_va, proba)
axes[1].plot(rec, pre, lw=1.6, color="#2e86de", label="PR-AUC = %.4f" % ap)
axes[1].set_xlabel("召回率 Recall"); axes[1].set_ylabel("精确率 Precision")
axes[1].set_title("Precision-Recall 曲线"); axes[1].legend(loc="lower left")

axes[2].imshow(cm, cmap="Blues")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        axes[2].text(j, i, format(cm[i, j], ","), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
axes[2].set_xticks([0, 1]); axes[2].set_yticks([0, 1])
axes[2].set_xticklabels(["正常", "欺诈"]); axes[2].set_yticklabels(["正常", "欺诈"])
axes[2].set_xlabel("预测"); axes[2].set_ylabel("真实")
axes[2].set_title("混淆矩阵")

plt.tight_layout()
out = ASSETS / "roc_pr_confusion.png"
plt.savefig(out, dpi=110)
print("已保存:", out)
