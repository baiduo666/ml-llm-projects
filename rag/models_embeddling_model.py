import warnings

warnings.filterwarnings("ignore")
# 导包
# 专门负责文本转向量
from langchain_ollama import OllamaEmbeddings


# 1.初始化本地嵌入向量模型对象.
# 参1: Ollama本地已下载的专用向量化模型.
# 注意: 嵌入模型不支持 temperature 参数，已移除.
model = OllamaEmbeddings(model='mxbai-embed-large')

# 2.单个文本向量化.
res1 = model.embed_query('这是第1个测试文档')
print(f'res1内容:{res1}')
print(f'res1长度:{len(res1)}')
# 3.批量多文本向量化,embed_documents方法.
res2 = model.embed_documents(['这是第1个测试文档', '这是第2个测试文档'])
print(f'res2内容:{res2}')
