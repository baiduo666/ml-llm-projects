# RNN 歌词生成

基于 **PyTorch RNN** 的歌词生成器:学习周杰伦歌词语料,根据用户输入的「提示词」生成后续歌词。

## 生成样例

> 提示词「分手」

~~~
分手的话像语言暴力
我已无能为力再提起 决定中断熟悉
然后在这里 不限日期
然后将过去 慢慢温习
让我爱上你 那场悲剧
是你完美演出的一场
~~~

> 提示词「晴天」

~~~
晴天 蝴蝶自在飞
花也布满天 一朵一朵因你而香
试图让夕阳飞翔 带领你我环绕大自然
迎著风 开始共渡每一天
手牵手 一步两步三步
~~~

更多样例见 [`samples/generated_examples.txt`](samples/generated_examples.txt)。

## 流程

1. **构建词表**:jieba 分词 + 去重构建词表(build_vocab),本语料词表大小 **5703**;
2. **构建数据集**:自定义 `LyricDataset`,按步长 32 切分为输入 / 标签序列(词索引);
3. **搭建网络**:Embedding(128 维) → RNN(128→256) → Linear(→词表大小);
4. **训练**:CrossEntropyLoss + Adam(lr=1e-4),10 轮;
5. **生成**:给定起始词,逐词贪心解码生成(`evaluate('分手', 102)`)。

## 技术栈

Python · PyTorch · jieba

## 运行

语料与训练好的模型**已随仓库提供**,clone 后可直接生成:

~~~bash
pip install -r requirements.txt
python train_and_generate.py   # 默认会重新训练; 只想生成可注释掉 train() 并放开 evaluate()
~~~

## 目录

~~~
rnn_lyrics_generator/
├── train_and_generate.py       # 词表构建 / 数据集 / 网络 / 训练 / 生成
├── data/jaychou_lyrics.txt     # 周杰伦歌词语料(每行一句)
├── model/text_generator.pth    # 训练好的模型权重(约 9 MB)
└── samples/                    # 生成样例
~~~

## 说明

模型规模很小(词嵌入 128 维、单层 RNN 256 维),生成结果以**模仿语料的句式与韵律**为主,不具备长程语义一致性 —— 这也是 RNN 相比 Transformer 的典型局限。
