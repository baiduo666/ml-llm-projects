import warnings
import os
warnings.filterwarnings("ignore")
from langchain_openai import ChatOpenAI

# Agnes 对接
api_key = os.environ.get('AGNES_API_KEY')
if not api_key:
    raise EnvironmentError(
        "未找到 AGNES_API_KEY 环境变量，请设置: "
        "export AGNES_API_KEY=your_key (Linux/Mac) 或 set AGNES_API_KEY=your_key (Windows)"
    )

model = ChatOpenAI(
    base_url=os.environ.get('base_url', 'https://apihub.agnes-ai.com/v1'),
    model='agnes-2.5-flash',
    api_key=api_key,
    temperature=0,
    max_tokens=10240
)

result = model.invoke("给我讲个笑话吧")
print(result)
print(result.content)
