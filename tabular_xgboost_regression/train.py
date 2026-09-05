import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb

DATA_DIR = Path(__file__).resolve().parent / "data"
df_train = pd.read_csv(DATA_DIR / "train.csv")
df_test = pd.read_csv(DATA_DIR / "test.csv")

# 保存id用于最后输出
sub_id = df_test["id"]

# 拆分特征与标签
X = df_train.drop(["id", "y"], axis=1)
y = df_train["y"]
X_test = df_test.drop(["id"], axis=1)

X["is_day"] = X["hour"].apply(lambda x: 1 if 7 <= x <= 22 else 0)
X_test["is_day"] = X_test["hour"].apply(lambda x: 1 if 7 <= x <= 22 else 0)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=417
)

model = xgb.XGBRegressor(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    objective="reg:squarederror",
    random_state=417,
)
model.fit(X_train, y_train)


valid_pred = model.predict(X_valid)
rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))
mae = mean_absolute_error(y_valid, valid_pred)
print(f"【验证集】RMSE={rmse:.2f}, MAE={mae:.2f}")

y_pred = model.predict(X_test)
y_pred = np.clip(y_pred, a_min=0, a_max=None).round().astype(int)

result = pd.DataFrame({
    "id": sub_id,
    "y": y_pred
})
result.to_csv(DATA_DIR / "result.csv", index=False, encoding="utf-8-sig")
print("预测结果已保存至 result.csv")
