# ml-llm-projects

个人机器学习 / 深度学习 / 大模型 项目合集。

## 项目总览

| 项目 | 领域 | 技术栈 | 说明 |
| --- | --- | --- | --- |
| [power_load_forecasting](./power_load_forecasting) | 时序预测 | Python, XGBoost, scikit-learn, PyQt5, Matplotlib | 电力负荷预测:多源数据融合 + 特征工程 + 训练/预测 + GUI 可视化 |
| [credit_card_fraud](./credit_card_fraud) | 分类 / 异常检测 | Python, XGBoost, scikit-learn | 信用卡欺诈检测:类别不平衡处理(样本权重)、分层交叉验证、GridSearch、ROC-AUC |
| [tabular_xgboost_regression](./tabular_xgboost_regression) | 表格回归 | Python, XGBoost, scikit-learn | 基于 XGBoost 的表格回归:特征工程 + RMSE / MAE 评估 |
| [rnn_lyrics_generator](./rnn_lyrics_generator) | 深度学习 / 序列生成 | PyTorch, jieba | 基于 RNN 的歌词生成:文本预处理 + 序列训练 + 生成 |
| [rag](./rag) | 大模型 / RAG | LangChain, Ollama, OpenAI 兼容接口, Redis | 检索增强生成(RAG)模块:本地向量化 + LLM 调用 + Milvus/Redis 集成 |

## 目录结构

```
ml-llm-projects/
├── README.md
├── .gitignore
├── power_load_forecasting/       # 电力负荷预测(XGBoost 回归 + PyQt5 GUI)
├── credit_card_fraud/            # 信用卡欺诈检测(XGBoost 二分类)
├── tabular_xgboost_regression/   # 表格回归(XGBRegressor)
├── rnn_lyrics_generator/         # 歌词生成(PyTorch RNN + jieba)
└── rag/                          # RAG 模块(LangChain + Ollama + Milvus/Redis)
```

## 使用

每个项目目录内均有独立的 `README.md` 与 `requirements.txt`,请按需进入对应目录查看:

```bash
cd <项目目录>
pip install -r requirements.txt
python <入口脚本>.py
```

