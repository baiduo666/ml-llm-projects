import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split


DATA_FILE = Path(__file__).resolve().parent / "data" / "creditcard.csv"


def train_creditcard():
    df = pd.read_csv(DATA_FILE)

    X = df.drop(["Label", "ID", "V_Time"], axis=1)
    y = df["Label"]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)

    classes_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    # XGB模型
    estimator = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.08,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42
    )
    # 训练时传入样本权重
    estimator.fit(X_train, y_train, sample_weight=classes_weights)

    # 4 模型评估
    y_valid_pred = estimator.predict(X_valid)
    y_valid_proba = estimator.predict_proba(X_valid)[:, 1]
    print("混淆矩阵：")
    print(confusion_matrix(y_valid, y_valid_pred))
    print(classification_report(y_true=y_valid, y_pred=y_valid_pred))
    print("验证集AUC：", roc_auc_score(y_valid, y_valid_proba))


def grid_search_cv():
    df = pd.read_csv(DATA_FILE)
    X = df.drop(["Label", "ID", "V_Time"], axis=1)
    y = df["Label"]
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=417, stratify=y)

    # 分层
    spliter = StratifiedKFold(n_splits=5, shuffle=True, random_state=417)
    # 超参网格
    param_grid = {
        'max_depth': [3, 4, 5],
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.05, 0.08, 0.1]
    }
    # 基础模型
    base_xgb = xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc",
                                 random_state=417)
    # 网格搜索+CV
    grid = GridSearchCV(estimator=base_xgb, param_grid=param_grid, cv=spliter, scoring="roc_auc")
    # 训练带权重
    w = compute_sample_weight("balanced", y_train)
    grid.fit(X_train, y_train, sample_weight=w)

    # 最优参数.评估
    print("最优参数：", grid.best_params_)
    best_model = grid.best_estimator_
    pred = best_model.predict(X_valid)
    print(classification_report(y_valid, pred))


if __name__ == '__main__':
    train_creditcard()
