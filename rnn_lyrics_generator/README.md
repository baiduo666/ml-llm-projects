# RNN 歌词生成

基于 **PyTorch RNN** 的歌词生成器:学习周杰伦歌词语料,根据用户输入的“提示词”生成后续歌词。

## 流程
1. **构建词表**:jieba 分词 + 去重构建词表(build_vocab)。
2. **构建数据集**:自定义 `LyricDataset`,步长切分为输入 / 标签序列(词索引)。
3. **搭建网络**:Embedding(128 维) → RNN(128→256) → Linear(→词表大小)。
4. **训练**:CrossEntropyLoss + Adam,多轮训练并保存模型。
5. **生成**:给定起始词,逐词采样生成(`evaluate('分手', 102)`)。

## 技术栈
Python · PyTorch · jieba

## 运行
将周杰伦歌词语料(每行一句)保存为 `data/jaychou_lyrics.txt`,然后:

```bash
pip install -r requirements.txt
python train_and_generate.py
```

## 说明
语料与训练得到的 `model/text_generator.pth` 体积较大,未纳入仓库。
