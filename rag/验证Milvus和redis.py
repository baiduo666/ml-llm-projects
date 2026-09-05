# from pymilvus import connections, utility
#
# # 连接到 Milvus
# connections.connect(host="localhost", port="19530")
#
# # 检查版本
# print(f"Milvus version: {utility.get_server_version()}")
import redis

# 连接到 Redis
client = redis.Redis(host="localhost", port=6379, password=<your_redis_password>, decode_responses=True)

# 测试读写
client.set("test_key", "Hello, Redis!")
value = client.get("test_key")
print(f"Redis value: {value}")