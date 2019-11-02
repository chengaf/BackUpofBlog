---
title: Translation With a Sequence to Sequence Network and Attention
date: 2019-10-10 13:07:23
tags: [深度学习,自然语言处理,seq2seq,attention]
toc: true
---

Pytorch Tutorial: NLP FROM SCRATCH: TRANSLATION WITH A SEQUENCE TO SEQUENCE NETWORK AND ATTENTION
Tutorial 地址: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
任务：Translate French to English  SeqtoSeq + Attention
&emsp; 类似name分类每个字符用一个one-hot表示，这里每个单词用一个one-hot。但是因为语料中单词数量很大，因此 only use a few thousand words per language。

<!--more-->

做好word和index映射：word2index，index2word，并记录词频：word2count。

准备数据的完整过程是：

读取文本文件并拆分为行，将行拆分为成对
规范文本，按长度（保留短句子便于快速计算，tutorial嘛）和内容过滤
成对建立句子中的单词列表

------

以下具体attention如何实现

Sequence to Sequence

Traditional： Encoder: The encoder reads an input sequence and outputs a single vector. Decoder: The decoder reads that vector to produce an output sequence.
Unlike sequence prediction with a single RNN, where every input corresponds to an output, the seq2seq model frees us from sequence length and order, which makes it ideal for translation between two languages.

### The Encoder

The encoder of a seq2seq network is a RNN that outputs some value for every word from the input sentence. For every input word the encoder outputs a vector and a hidden state, and uses the hidden state for the next input word.

### The Decoder

The decoder is another RNN that takes the encoder output vector(s) and outputs a sequence of words to create the translation.

#### Simple Decoder

In the simplest seq2seq decoder we use only last output of the encoder. This last output is sometimes called the *context vector* as it encodes context from the entire sequence. This context vector is used as the initial hidden state of the decoder.

At every step of decoding, the decoder is given an input token and hidden state. The initial input token is the start-of-string `<SOS>` token, and the first hidden state is the context vector (the encoder’s last hidden state).

### Attention

```python
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length) 
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
				# 获取权重，self.attn是全连接输出长度是max_length，权重根据输入embedded和hidden得到。
        # 这里的hidden就是上一个单元输出的hidden，
        # 这里输入embedded根据有无 Teacher 分两种：
        #     从target_tensor获取或者从历史得到（上一个decoder_output取top1然后据此获得）
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        # 权重乘以encoder的输出（应用权重）
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        # 拼接input和应用权重之后的tensor
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        # 上一步拼接完之后size变为2倍的，在经过全连接self.attn_combine给降到self.hidden_size
        output = self.attn_combine(output).unsqueeze(0)
				# 这里F.relu作用的output是结合了embedded（也就是input）和focus了重要程度的encoder_outputs（乘了权重）
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
```

attention 的图

<img src="https://pytorch.org/tutorials/_images/attention-decoder-network.png" alt="img" />

------

以下知识补充

**pytorch torch.nn.Embedding** 

```python
torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None)
# 一个简单的查找表，用于存储固定字典和大小的嵌入。
# 该模块通常用于存储单词嵌入并使用索引检索它们。模块的输入是索引列表，而输出是相应的词嵌入。
# 例子：
# an Embedding module containing 10 tensors of size 3
embedding = nn.Embedding(10, 3)
# a batch of 2 samples of 4 indices each
input = torch.LongTensor([[1,2,4,5],[4,3,2,9]]) # 按照id取embedding
embedding(input)
tensor([[[-0.0251, -1.6902,  0.7172],
         [-0.6431,  0.0748,  0.6969],
         [ 1.4970,  1.3448, -0.9685],
         [-0.3677, -2.7265, -0.1685]],

        [[ 1.4970,  1.3448, -0.9685],
         [ 0.4362, -0.4004,  0.9400],
         [-0.6431,  0.0748,  0.6969],
         [ 0.9124, -2.3616,  1.1151]]])


# example with padding_idx
embedding = nn.Embedding(10, 3, padding_idx=0) # 将input中id为padding_idx的位置补为0
input = torch.LongTensor([[0,2,0,5]])
embedding(input)
tensor([[[ 0.0000,  0.0000,  0.0000],
         [ 0.1535, -2.0309,  0.9315],
         [ 0.0000,  0.0000,  0.0000],
         [-0.1655,  0.9897,  0.0635]]])
```

**pytorch torch.bmm**  批矩阵相乘（应用权重时用的）

总结：通过这个Tutorial看 Seq2Seq + Attention怎么实现