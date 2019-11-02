---
title: Classifying Names With a Character-level RNN
date: 2019-10-09 09:04:26
tags: [自然语言处理,RNN]
toc: true
---

Pytorch Tutorial:  CLASSIFYING NAMES WITH A CHARACTER-LEVEL RNN 利用RNN对姓名所属的语言进行分类。

Tutorial 地址: https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

分为几个步骤：

> 1. 读取数据name。
> 2. 根据数据构造输入，将原始数据name转为模型输入tensor。
> 3. 构建神经网络。
> 4. 训练（自己写的random sample和梯度下降没用封装好的batch选取和优化器）。
> 5. 预测。

<!--more-->

#### 第一部分 读取数据

```python
# 使用glob.glob(path)获取path下所有符合条件的文件名，返回的是个list，存储所有满足条件的
def findFiles(path): return glob.glob(path)


print(findFiles('data/names/*.txt'))

import unicodedata
import string

# all_letters='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,;'
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


# 将 Unicode 编码转为 ASCII （统一编码）
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


print(unicodeToAscii('Ślusàrski'))

# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []


# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')  # 一下子全部读取？
    return [unicodeToAscii(line) for line in lines]


# 这个函数和原教程里的readLines输出一样
def readLiness(filename):
    lines = []
    with open(filename, 'r', encoding='utf-8') as data:  # 获取文件对象
        for d in data:  # 逐行读取
            line = d.strip()
            lines.append(unicodeToAscii(line))
    return lines


# 将不同语言的数据转为dictionary，key值为语言类别，value为list，存储所有name。
for filename in findFiles('data/names/*.txt'):
    caf0 = os.path.basename(filename)
    caf1 = os.path.splitext(caf0)
    category = caf1[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)
```

#### 第二部分 根据数据构建模型的输入（将name转为tensor）

```python
import torch


# Find letter index from all_letters, e.g. "a" = 0 找某一个字符在字符串 all_letters 中的index
def letterToIndex(letter):
    return all_letters.find(letter)


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters) # 每个字符是一个 1*len(all_letters) 的 tensor（one-hot）
    tensor[0][letterToIndex(letter)] = 1 # 字符对应位置为1
    return tensor


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line): # 每个name是一个len(name)*1*len(all_letters)的tensor
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

print(letterToTensor('J'))
print(lineToTensor('Jones').size())
```

#### 第三部分 构建神经网络

```python
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size) # input to hidden
        self.i2o = nn.Linear(input_size + hidden_size, output_size) # input to output
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1) # 拼接输入和隐藏层输出
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden
    # 初始化隐藏层单元
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

# 初始化网络
# RNN 的输入大小input_size为len(all_letters)，因为每个cell每次处理一个字符，每个name是一个字符序列。
# RNN 的输出大小output_size为类别个数，最后预测某一个name属于哪一种语言。
n_hidden = 128 # 隐藏层size
rnn = RNN(n_letters, n_hidden, n_categories)

# 输入例子
input = letterToTensor('A')
hidden =torch.zeros(1, n_hidden) # 初始化隐藏单元值为全0
output, next_hidden = rnn(input, hidden)
print(output, next_hidden)

input = lineToTensor('Albert')
hidden = torch.zeros(1, n_hidden)
output, next_hidden = rnn(input[0], hidden) # input[0]：name-'Albert'的第一个字符
print(output, next_hidden)
```

#### 第四部分 训练网络

##### 训练准备

```python
# 返回 最大值 top_n 及其位置 top_i ，在类别列表 all_categories 中查询所属类别，在返回所属类别及其id
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i
print(categoryFromOutput(output))

import random


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)] # 随机取l中的一个元素


def randomTrainingExample():
    category = randomChoice(all_categories) # 随机取一个类
    line = randomChoice(category_lines[category])# 随机取类category中的一个name
    category_id = all_categories.index(category)
    category_tensor = torch.tensor([category_id], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor


for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category =', category, '/ line =', line)
```

##### 训练

```python
# 定义损失函数
criterion = nn.NLLLoss()


# train函数
learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn
def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)

    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    # 没用优化器直接梯度下降
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()

  
import time
import math

n_iters = 100000
print_every = 5000


# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample() # 一个name一个name的训练？！
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

```

#### 第五部分 预测

```python
def evaluate(line_tensor):
    hidden = rnn.initHidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    return output

def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])

predict('Dovesky')
predict('Jackson')
predict('Satoshi')
```

#### 总结：

1. 模型输入的构建：name的每个字符是一个one-hot，name表示为一系列的字符one-hot。
2. 利用RNN做的分类，也就是多对一的结构。
3. Tutorial的代码还是很实在的：读取数据一下读全部、一个样本一个的训练（自己采样的没有用内部封装的batch data选取）、优化直接用的 ’p.data.add_(-learning_rate, p.grad.data)‘（没有用封装的优化器）。
4. 预测时用的 “with torch.no_grad()”，没有用“model.eval()”。区别：
   - model.eval() will notify all your layers that you are in eval mode, that way, batchnorm or dropout layers will work in eval mode instead of training mode.
   - torch.no_grad() impacts the autograd engine and deactivate it. It will reduce memory usage and speed up computations but you won’t be able to backprop (which you don’t want in an eval script).