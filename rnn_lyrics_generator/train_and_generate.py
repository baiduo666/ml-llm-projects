"""
案例:
    演示RNN案例 -> AI歌词生成器.

需求:
    用周杰伦的歌词训练模型, 让模型基于用户录入的"提示词", 生成对应的歌词.

研发流程:
    1. 构建词表等.
    2. 构建数据集.
    3. 搭建神经网络.
    4. 模型训练.
    5. 模型测试.
"""

# 导包
from pathlib import Path

import jieba
import torch            # 用于: 深度学习相关操作.
from torch.utils.data import DataLoader     # 用于: 构建数据集 -> 数据加载器, 可以分批次获取数据.
import torch.nn as nn                       # 用于: 搭建神经网络.
import torch.optim as optim                 # 用于: 优化器.
import time                                 # 用于: 获取时间.


PROJECT_ROOT = Path(__file__).resolve().parent
LYRICS_FILE = PROJECT_ROOT / "data" / "jaychou_lyrics.txt"
MODEL_FILE = PROJECT_ROOT / "model" / "text_generator.pth"


# todo 1. 获取数据, 并进行分词, 获取词表.
def build_vocab():
    # 1. 定义遍历, 记录: 歌词文件的路径.
    # 2. 定义吧了, 记录: 分词结果存储位置.
    # unique_words: 存储: 去重后所有的词(即: 词汇表), 即: [词1, 词2, 词3...]
    # all_words: 每行文本分词结果, 即: [[第1行切词后结果], [第2行切词后结果], ...]
    unique_words, all_words = [], []
    # 3. 遍历数据集, 获取每行文本.
    for line in LYRICS_FILE.open('r', encoding='utf-8'):
        # 3.1 获取每行文本, 并进行分词.
        words = jieba.lcut(line)
        # 3.2 所有分词结果存储到all_words列表中, 其中包含重复的词组.
        all_words.append(words)
        # 3.3 遍历分词结果, 去重后存储到unique_words列表中.
        for word in words:
            if word not in unique_words:
                unique_words.append(word)

    # 4.统计语料中词的数量 -> 即: 词汇表的大小(去重后的)
    word_count = len(unique_words)
    # 5. 构建词表字典, 键: 词, 值: 该词对应的索引.
    # 确保空格在词表中（jieba 分词可能不产生空格 token）
    if ' ' not in unique_words:
        unique_words.append(' ')
    word_to_index = {word: i for i, word in enumerate(unique_words)}
    # 6. 歌词文本用词表索引表示.
    corpus_idx = []
    # 7. 遍历每一行的分词结果.
    for words in all_words:
        # 7.1 定义遍历, 记录: 词索引列表
        tmp = []
        # 7.2 获取每一行的词, 并获取相应的索引.
        for word in words:
            tmp.append(word_to_index[word])
        # 7.3 在每行词之间, 添加空格隔开.
        tmp.append(word_to_index[' '])
        # 7.4 把上述的 每个句子(对应的词索引列表) 添加到corpus_idx列表中.
        corpus_idx.extend(tmp)
    # 8. 返回: 唯一词列表, 词表, (去重后)词的数量, 歌词文本用词表索引表示.
    return unique_words, word_to_index, word_count, corpus_idx


# todo 2. 构建数据集 -> 目的是: 构建数据加载器对象, 思路为: 自定义的数据集对象 -> 数据加载器对象(DataLoader)
class LyricDataset(torch.utils.data.Dataset):
    # todo 2.1 初始化词索引, 词个数等...
    # 参1: 歌词文本用词表索引表示的列表, 参2: (假设1个句子的)词数量
    def __init__(self, corpus_idx, num_chars):
        # 1. 文档数据中的词索引.
        self.corpus_idx = corpus_idx
        # 2. 每个句子中的词数量.
        self.num_chars = num_chars
        # 3. 文档中, 词的数量(不去重)
        self.word_count = len(self.corpus_idx)
        # 4. 计算句子数量.
        self.number = self.word_count // self.num_chars     # 每个句子的词都是独立的, 即: 步长为5
        # self.number = self.word_count - num_chars + 1     # 步长为1

    # todo 2.2 定义魔法方法len, 当使用 len(obj)的时候, 会自动调用该方法.
    def __len__(self):
        return self.number      # 返回句子的数量.

    # todo 2.3 定义魔法方法getitem, 当使用 obj[index]的时候, 会自动调用该方法.
    def __getitem__(self, idx):
        # 1. 确保start索引在合法的范围内, 避免越界, start: 当前样本的起始索引.
        # idx: 词的索引.
        # self.word_count: 文档中, 词的数量(不去重)
        # self.num_chars: 每个句子中的词数量.
        start = min(max(idx, 0), self.word_count - self.num_chars - 1)
        # 2. 获取当前句子的结束索引.
        end = start + self.num_chars
        # 3. 获取具体的句子, 作为: 输入x
        x = self.corpus_idx[start:end]
        # 4. 获取句子对应的标签, 作为: 输出y
        y = self.corpus_idx[start+1:end+1]
        # 5. 封装成张量, 并返回结果.
        return torch.tensor(x), torch.tensor(y)


# todo 3. 搭建神经网络.
class TextGenerator(nn.Module):
    # todo 3.1 初始化方法, 搭建神经网络.
    def __init__(self, unique_word_count):
        # 1. 初始化父类的成员.
        super().__init__()
        # 2. 搭建词嵌入层: 维度128 -> 每个词用128维的向量表示.
        self.ebd = nn.Embedding(unique_word_count, 128)
        # 3. 搭建RNN(循环网络层), 输入维度: 128, 输出维度: 256, 网络层数: 1
        self.rnn = nn.RNN(128, 256, 1)
        # 4. 构建全连接层(输出层), 输入维度: 256, 输出维度: unique_word_count(即: 5703)
        self.out = nn.Linear(256, unique_word_count)

    # todo 3.2 前向传播方法.
    # 参1: 输入值(即: 当前时刻的输入, 就是x), 参2: 隐藏层初始值(即: 上一时刻的隐藏层状态, ht-1)
    def forward(self, inputs, hidden):
        # 1. 初始化 词嵌入层处理.
        # embed形状: (句子数量, 句子长度, 词向量维度)
        embed = self.ebd(inputs)
        # 2. 初始化 rnn层处理.
        # rnn层, x要的输入形状: (句子长度, 句子数量, 词向量维度)
        output, hidden = self.rnn(embed.transpose(0, 1), hidden)
        # 3. 初始化全连接层(2维).
        # 输入维度: (句子长度, 句子数量, 词向量维度)
        # output: 每个词的分值分布, 后续要结合softmax()进行概率分布预测.
        output = self.out(output.reshape(shape=(-1, output.shape[-1])))
        # 4. 返回结果.
        return output, hidden


    # todo 3.3 隐藏层的初始化方法.
    def init_hidden(self, bs):
        return torch.zeros(1, bs, 256)  # bs: batch_size 批次大小


# todo 4. 模型训练.
def train():
    # 1. 构建词表.
    unique_words, word_to_index, unique_word_count, corpus_idx = build_vocab()
    # 2. 获取数据集.
    lyrics = LyricDataset(corpus_idx, 32)
    # 3. 初始化模型.
    model = TextGenerator(unique_word_count)
    # 4. 创建数据加载器对象.
    lyrics_loader = DataLoader(lyrics, batch_size=5, shuffle=True)
    # 5. 创建损失函数.
    criterion = nn.CrossEntropyLoss()
    # 6. 创建优化器.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # 7. 训练模型.
    # 7.1 定义变量, 记录: 训练的轮数.
    epochs = 10
    # 7.2 具体的每轮训练动作.
    for epoch_idx in range(epochs):
        # 7.3 定义变量, 记录: 本轮开始训练时间, 迭代次数, 训练损失.
        start, iter_num, total_loss = time.time(), 0, 0.0
        # 7.4 从加载器中 逐批次获取数据, 并训练.
        for x, y in lyrics_loader:
            # 7.4.1 获取隐藏层初始值.
            hidden = model.init_hidden(bs=5)
            # 7.4.2 模型计算.
            output, hidden = model(x, hidden)
            # 7.4.3 计算损失.
            # 7.4.3.1 对预测结果做转换, 转成: 标量.
            y = torch.transpose(y, 0, 1).reshape(shape=(-1, ))
            # 7.4.3.2 计算损失.
            loss = criterion(output, y)
            # 7.4.4 反向传播.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 7.4.5 记录训练信息.
            # 累计训练损失
            total_loss += loss.item()
            # 累计迭代次数(批次数).
            iter_num += 1

        # 7.5 走到这里, 说明本轮(1轮)训练完毕, 打印训练信息.
        print(f'轮数: {epoch_idx+1}, 训练损失: {total_loss/iter_num:.4f}, 训练时长: {time.time()-start:.2f}s')

    # 8. 保存模型.
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_FILE)
    print('模型保存成功!')


# todo 5. 模型测试.
# 参1: 起始词(必须是 5703个词其中1个), 参2: 句子长度.
def evaluate(start_word, sentence_length):
    # 1. 构建词表.
    unique_words, word_to_index, unique_word_count, corpus_idx = build_vocab()
    # 2. 检查起始词是否在词表中
    if start_word not in word_to_index:
        print(f"错误：起始词 '{start_word}' 不在词表中，请从现有词汇中选择")
        return
    # 3. 初始化模型.
    model = TextGenerator(unique_word_count)
    # 4. 加载模型参数（weights_only=True 防止代码执行，map_location 支持 CPU/GPU 跨设备）
    model.load_state_dict(torch.load(MODEL_FILE, map_location='cpu', weights_only=True))
    # 5. 切换到评估模式（禁用 Dropout 等训练专用层）
    model.eval()
    # 6. 加载隐藏层信息.
    hidden = model.init_hidden(bs=1)
    # 7. 将起始词转换为: 索引.
    word_idx = word_to_index[start_word]
    # 7. 定义列表, 记录: 产生的词的索引.
    generate_sentence = [word_idx]
    # 8. 遍历句子长度, 获取到(预测的)每个词
    with torch.no_grad():  # 推理时不需要计算梯度，节省内存并加速
        for _ in range(sentence_length):
            # 8.1 模型预测.
            output, hidden = model(torch.tensor([[word_idx]]), hidden)
            # 8.2 获取预测结果.
            word_idx = torch.argmax(output)
            # 8.3 添加到列表中.
            generate_sentence.append(word_idx)

    # 8. 把索引转成: 词, 并打印.
    for idx in generate_sentence:
        print(unique_words[idx], end='')




# todo 6. 测试代码.
if __name__ == '__main__':
    # # 1. 获取数据, 并进行分词, 获取词表.
    # unique_words, word_to_index, word_count, corpus_idx = build_vocab()
    # # print(f'词的数量: \n {word_count}')
    # # print(f'去重后的词: \n {unique_words}')
    # # print(f'每个词的索引: \n {word_to_index}')
    # # print(f'歌词文本用词表索引表示: \n {corpus_idx}')
    #
    # # 2. 构建数据集.
    # # 2.1 创建数据集对象.
    # dataset = LyricDataset(corpus_idx, 5)
    # # 2.2 查看句子数量
    # print(f'句子数量: \n {len(dataset)}')
    # # 2.3 查看输入值 和 目标值.
    # x, y = dataset[3]
    # print(f'输入值: \n {x}')
    # print(f'目标值: \n {y}')
    #
    # # 3. 创建模型.
    # model = TextGenerator(word_count)
    # # for name, param in model.named_parameters():
    # #     print(f'参数名称: \n {name}, 参数形状: \n {param.shape}')

    # 4. 模型训练.
    train()

    # 5. 模型预测.
    # evaluate('分手', 102)
