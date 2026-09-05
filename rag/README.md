# RAG 模块

基于 **LangChain** 的检索增强生成(RAG)接入模块。

## 组成
- **models_embeddling_model.py** — 文本向量化:调用本地 Ollama 的 `mxbai-embed-large` 嵌入模型(OllamaEmbeddings),支持单条与批量向量化。
- **models_llms_chatopenai.py** — LLM 调用:通过 OpenAI 兼容接口(Agnes)调用大模型,API Key 从环境变量 `AGNES_API_KEY` 读取。
- **验证Milvus和redis.py** — 连通性测试:验证 Redis(及可选的 Milvus)连接。

## 技术栈
Python · LangChain · Ollama · OpenAI 兼容接口 · Redis · Milvus

## 运行
```bash
export AGNES_API_KEY=你的key   # Linux/Mac
set AGNES_API_KEY=你的key      # Windows
pip install -r requirements.txt
python models_embeddling_model.py
python models_llms_chatopenai.py
```

## 安全说明
API Key 通过环境变量注入,仓库内不含任何真实凭据;Redis 密码为占位符。
