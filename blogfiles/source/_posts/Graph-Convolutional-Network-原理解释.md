---
title: Graph Convolutional Network 原理解释
date: 2019-10-09 09:33:38
tags: [图神经网络]
mathjax: true
toc: true
---

写在前面

> 1. 思路：为什么提出 GCN &ensp;--> 如何从传统意义的 CNN 到 GCN（谱角度解释）-->GCN 的代表性工作(方法的发展) 
> 2. 根据一些blog（参照最后资源君，侵删）和原论文整理的笔记，加了一些帮助我理解的东西，加了个图。
> 3. 清华整理的关于GNN 的 must-read paper list：https://github.com/thunlp/GNNPapers 
> 4. GNN系列方法的实现1：https://github.com/rusty1s/pytorch_geometric  [pytorch]
> 5. GNN系列方法的实现2：https://github.com/dmlc/dgl [pytorch]

<!--more-->

## 1. 为什么要研究GCN

### 1.1 传统CNN的认识
#### 1.1.1 卷积定义

**卷积本质上就是一种加权求和**。参见资源君-3
&emsp;卷积的定义, 称 $$(f*g)(n)$$ 为 $$f,g$$ 的卷积，

（1）连续定义为： $$(f*g)(n)=\int_{\tau=-\infty}^{\tau=+\infty}f(\tau)g(n-\tau)d\tau$$ &emsp;&emsp;公式（1）
（2）离散定义为： $$(f*g)(n)=\sum_{\tau=-\infty}^{\tau=+\infty}f(\tau)g(n-\tau)$$ &emsp;&emsp;公式（2）

&emsp;卷积的理解：“卷”：翻转，施加一种约束，限制“积”；“积”：加权求和，考虑对所有输入的效果累积。卷积本质上就是一种加权求和：最终的效果受所有输入的影响不同，以卷（滑动）控制加权求和。
&emsp;例子1：掷两个骰子，和为 $$n=4$$ 的概率。$$f,g$$  都为掷一个骰，按照卷积的定义即公式（2）可以定义  $$(f*g)(n)$$ 为： $$(f*g)(4)=\sum_{\tau=1}^{3}f(\tau)g(4-\tau)$$，两个骰子点数之和为4为约束$$g(4-\tau)$$，函数 $$f,g$$ 都为[1,2,3,4,5,6]，翻转函数 $$g$$ [6,5,4,3,2,1]，滑动函数 $$g$$ ，图解：
&emsp;&emsp; <img src="https://chengaf.github.io/afcheng.github.io/assets/blog_img/卷积掷骰子例子.jpg"  class="js-avatar show">

&emsp;例子2：工厂排放化学物质的残留。工厂不断排放化学物质， $$t$$ 时刻排放量是 $$f(t)$$ ，被排放的化学物质在排放 $$\Delta t$$ 时间后，即 $$t+\Delta t=u$$ 时刻的残留比率是 $$g(\Delta t)$$ ，于是 $$t$$ 时刻排放的化学物质在 $$t+\Delta t=u$$ 时刻还剩余 $$g(\Delta t)$$ ，而 $$u$$ 时刻的化学物质总量等于 $$u$$ 时刻之前所有时刻的总和，由公式（1）可以得出 $$u$$ 时刻的化学物质总量为： $$\int_{-\infty}^{+\infty}f(t)g(u-t)dt$$ ，这就是卷积。化学物质残留比率作为限制，对所有输入（排放的化学物质）对最终总残留值贡献量做出限制。

#### 1.1.2 Image 上的 CNN
&emsp;利用参数共享的过滤器（kernel），通过计算中心像素点以及相邻像素点与 kernel 的加权和来构成 Feature Map 实现空间特征提取，加权系数就是卷积核的权重系数。
&emsp;卷积核权重系数通过随机初始化初值，再通过误差函数反向传播梯度下降进行迭代优化。**卷积核的参数通过优化求出才能实现特征提取的作用，GCN 也想像这样引入可优化的卷积核，针对不同的任务，优化对应的卷积核。**

### 1.2 为什么非得用 CNN 处理 Graph 数据
&emsp;1. CNN 处理 Euclidean Structure 数据十分有效，能够有效地提取特征。
&emsp;2. Graph 都快能建模万物了，那 CNN 在 Graph 数据上的使用？Graph 是Non-Euclidean Structure 数据。
&emsp; 总结为什么要研究 GCN 的理由（应用范围广泛，效果又好，现有方法又没法儿弄。）：

> 1. 传统的 CNN 无法直接处理 Graph 这种 Non-Euclidean Structure 数据，传统的离散卷积无法在这种数据上保持平移不变性。也就是拓扑图中每个节点的邻居数目不相等，因此也无法用一个同样尺寸的卷积核来进行卷积。
> 2. CNN 已经被证明能够有效地提取特征，又不能直接拿来用，因此 GCN 称为研究重点，以期通过卷积操作在 Graph 上有效地提取特征。
> 3. Graph 数据形式的广泛性，广义上来讲任何数据范畴空间都可以简历拓扑关联，谱聚类（Spectral Clustering）即用了这种思想。因此，拓扑连接是一种广义的数据结构，GCN 具有非常广泛的应用空间。

## 2. CNN怎么处理Graph
&emsp;从空域（spatial space）和谱域（spectral space）两种研究思路。

### 2.1 空域（spatial space）
&emsp;根据拓扑图上的空间特征进行研究。研究点集中两点：
&emsp;&emsp; a. 给定一个节点怎么找它的 neighbors，即如何确定接收域（receptive field）。
&emsp;&emsp; b. 如何处理包含不同数目的 neighbor的节点特征。
&emsp; 缺点：计算处理必须要针对每个节点，因为要找每个节点的 neighbor。AND SO ON ...
#### 2.1.1 相关论文
> 1. 2016 ICML Learning Convolutional Neural Networks for Graphs.

### 2.2 谱域（spectral space）
&emsp; 借助图谱理论实现拓扑图上的卷积操作。
&emsp; 研究进程：首先研究 Graph Single Processing 定义 Graph 上的 Fourier Transformation，进而结合深度学习提出 Graph Convolutional Network
&emsp; 会出现的几个关键词：拉普拉斯矩阵的特征值特征向量、傅立叶变换、逆傅里叶变换。**<font face='宋体' color='0099ff'>利用傅立叶变换将时域复杂的卷积操作变为频域简单的乘积运算，这其中傅立叶变换的频谱和基对应于 Graph 拉普拉斯矩阵的特征值和特征向量。</font>**

#### 2.2.1 Spectral Graph Theory
&emsp;借助图的拉普拉斯矩阵的特征值和特征向量来研究图的性质。如何在 Graph 上做傅立叶变换的理论基础，找到 Graph 上傅立叶变换的基和频谱。
- 图相关的基本定义
图 Graph: $$G=(V,E)$$， 
邻接矩阵 adjacency matrix: $$A$$， 
定点度的对角矩阵：$$D$$， 
拉普拉斯矩阵 laplacian matrix：$$L=D-A$$。
两种正则化的拉普拉斯矩阵：
- symmetric normalized laplacian $$L_{sym}=I-D^{-\frac{1}{2}}AD^{-\frac{1}{2}}=D^{-\frac{1}{2}}DD^{-\frac{1}{2}}-D^{-\frac{1}{2}}AD^{-\frac{1}{2}}=D^{-\frac{1}{2}}LD^{-\frac{1}{2}}$$
- random walk normalized laplacian $$L_{rw}=I-D^{-1}A=D^{-1}L$$
- 拉普拉斯矩阵的特征分解（谱分解）
将矩阵分解为由其特征值和特征向量表示的矩阵之积的方法，**只有可对角矩阵才能进行特征分解**。相关概念：
  
> &emsp;1. 可对角化矩阵：如果对于一个方阵 $$A$$， 存在一个可逆矩阵 $$P$$ 使得 $$P^{-1}AP$$ 是对角矩阵阵，则称 $$A$$ 是可对角化。
> &emsp;2. 可对角化、特征分解、谱分解是同一个概念。
> &emsp;3. 对称矩阵一定有n个线性无关的特征向量。
> &emsp;4. 半正定矩阵的特征值一定非负。
> &emsp;5. 对称矩阵的特征向量相互正交，即所有特征向量构成的矩阵为正交矩阵。* 

矩阵特征值分解：
> *假设 $$L$$ 是一个 $$N\times N$$ 的方阵，且有 $$N$$ 个线性无关的特征向量 $$u_{i}(i=1,2,...N)$$ ，这样 $$L$$ 可以被分解为 $$L=U\Lambda U^{-1}$$。其中 $$U$$ 是$$N\times N$$ 的方阵， $$U$$ 的第 $$i$$  列为的第 $$i$$ 个特征向量。$$\Lambda$$ 是对角矩阵，对角线上的元素为对应的特征值，即 $$\Lambda_{ii}=\lambda_{i}$$。*

**拉普拉斯矩阵是半正定矩阵（本身就是对称矩阵）**，因此可以进行谱分解一定有n个线性无关的特征向量，特征向量两两正交，特征值一定非负。

拉普拉斯矩阵特征值分解：$$L=U\Lambda U^{-1}$$， 其中，$$U\in R^{N\times N}$$ 是特征向量矩阵，$$U=\{\vec{u_{1}}, \vec{u_{2}}, ..., \vec{u_{n}}\}$$，向量 $$\vec{u_{i}}$$ 是列向量，$$\Lambda$$ 是与特征向量对应特征值构成的对角矩阵。由于特征向量矩 $$U$$ 正交，因此 $$UU^{-1}=E$$，即 $$U^{T}=U^{-1}$$ ，
因此特征分解可表示为 ** $$L=U\Lambda U^{T}$$ 拉普拉斯矩阵分解的特殊形式，就不用求矩阵的逆了，直接用转置矩阵就好**。 
$$\Lambda=\left( \begin{matrix} \lambda_{1} &  & \\ & \ddots &  \\  &  & \lambda_{n}\end{matrix} \right)$$

#### 2.2.2 从传统的傅立叶变换到 Graph 上的傅立叶变换
将传统拉普拉斯算子的特征函数 $$e^{-iwt}$$ 变为 Graph 拉普拉斯矩阵分解的特征向量。
> 1. 傅立叶变换：找到一组基表示任何连续的函数；拉普拉斯矩阵分解：找到特征值和特征向量去唯一表示每个节点。
> 2. 矩阵的谱就是矩阵特征值是矩阵所固有的特性，所有的特征值形成了一个频谱，每个特征值是矩阵的一个“谐振频点”。
> 3. 矩阵的谱是其所有特征值的集合，所有特征值构成了矩阵的频谱。矩阵所有特征向量组成了矩阵的基，可以理解为坐标系的轴。
> 4. 信号与基函数（拉普拉斯算子的特征方程）的积分，到 Graph 上就是输入节点的 Feature（输入信号） 与特征向量（ Graph 的基）的内积。


- 传统的傅立叶变换
傅立叶变换 $$F(w)$$, 定义：信号 $$f(t)$$ 与基函数 $$e^{-iwt}$$ 的积分。即： $$F(w)=\int_{-\infty}^{+\infty}f(t) e^{-iwt}dt$$。为什么找 $$e^{-iwt}$$ 做为基函数？数学上看，$$e^{-iwt}$$ 是拉普拉斯算子的特征函数（满足特征方程），$$w$$就和特征值相关。
- 传统的逆傅立叶变换
是对频率 $$w$$ 求积分：$$F^{-1}[F(w)]=\frac{1}{2\pi}\int F(w)e^{iwt}dw$$
- 广义特征方程
$$AV=\lambda V$$ ，其中 $$A$$ 是一种变换， $$V$$ 是一种特征向量或者特征函数（无穷维特征向量），$$\lambda$$ 是特征值。特征向量在变换 $$A$$ 的作用下，进行了比率为特征值的伸缩。
- Graph 上的傅立叶变换
Graph节点对应传统傅立叶变换的时刻 $$t$$ ，节点离散，离散积分定义为内积形式：$$F(\lambda_{l})=\hat{f}(\lambda_{l})=\sum_{i=1}^{N}f(i)u_{l}(i)$$ 。Graph节点对时刻 $$t$$，因此 $$f(i)$$ 对应节点，特征函数 $$e^{-iwt}$$ 与拉普拉斯矩阵的特征向量对应。$$u_{l}(i)$$ 表示第 $$l$$ 个特征向量的第 $$i$$ 个分量。所有节点的输入表示为 $$f=[f(1),f(2),...f(N)]^{T}$$ ，$$f(i)$$ 表示一个节点，求和中的一项 $$f(i)u_{l}(i)$$ 针对每个节点，共 $$N$$ 个节点（时刻），每个特征向量（基）有 $$N$$维，与 $$N$$ 个节点对应相乘，写成矩阵形式，如下：
$$\left( \begin{matrix} 
u_{1}(1)& u_{1}(2)& \cdots&u_{1}(N)\\
u_{2}(1)& u_{2}(2)& \cdots&u_{2}(N)\\
\vdots & \vdots & \ddots &\vdots\\  
u_{N}(1) & u_{N}(2) & \cdots &u_{N}(N)
\end{matrix}\right)
\left( \begin{matrix}
f(1)\\
f(2)\\
\vdots\\
f(N)
\end{matrix}\right)=\hat{f}(\lambda_{l})=\left( \begin{matrix}
\hat f(\lambda_{1})\\
\hat f(\lambda_{2})\\
\vdots\\
\hat f(\lambda_{N})
\end{matrix}\right)$$
即： $$f$$ 在 Graph 上的傅立叶变换为 $$\hat{f}=U^{T}f$$。
- Graph 上的逆傅立叶变换
对特征值 $$\lambda_{l}$$ 求和：$$f(i)=\sum_{i=1}^{N}\hat{f}(\lambda_{i}) u_{l}(i)$$，同样利用矩阵乘法将其推广到矩阵形式：
$$\left( \begin{matrix}
f(1)\\
f(2)\\
\vdots\\
f(N)
\end{matrix}\right)=
\left( \begin{matrix} 
u_{1}(1)& u_{2}(1)& \cdots&u_{N}(1)\\
u_{1}(2)& u_{2}(2)& \cdots&u_{N}(2)\\
\vdots & \vdots & \ddots &\vdots\\
u_{1}(N) & u_{2}(N) & \cdots &u_{N}(N)
\end{matrix}\right)\left( \begin{matrix}
\hat f(\lambda_{1})\\
\hat f(\lambda_{2})\\
\vdots\\
\hat f(\lambda_{N})
\end{matrix}\right)$$
即：$$f$$ 在 Graph 上的逆傅立叶变换为 $$f=U \hat{f}$$。


#### 2.2.3 Graph卷积从谱域操作的总体的理解图
- &emsp;结合2.2.1与2.2.2总结如何从频谱角度进行 Graph 上的卷积操作。

<img src="https://chengaf.github.io/afcheng.github.io/assets/blog_img/图卷积傅立叶变换.jpg"  class="js-avatar show">

- &emsp;Tips:
> 卷积：一种运算方式。
> 傅立叶变换：让函数在时域和频域相互转换。
> 卷积定理：时域卷积等于频域乘积，时域乘积等于频域卷积。

- Graph Convolution Network 目的是需要设计共享的（Parameter Sharing）、可学习的（Trainable）卷积核（Convolutional Kernel）。对 Graph Convolution 而言就是 $$U\Lambda U^{T}X$$ 这个 $$U\Lambda U^{T}$$ 与传统CNN不同的是这玩意儿对一个graph而言可以提前计算不用学习，传播过程中学习权重矩阵就行了。

- 此外许多地方将图卷积写成 $$(f*h)_{G}=U((U^{T}h)\bigodot(U^{T}f))$$，其中$$\bigodot$$ 为哈达玛积，对应元素相乘。

把 $$\bigodot$$ 转为矩阵乘法，因为：
$$\left( \begin{matrix} x_{1}\\ \vdots \\x_{n} \end{matrix}\right)\bigodot \left( \begin{matrix} y_{1}\\ \vdots \\y_{n}\end{matrix}\right)=\left( \begin{matrix} x_{1}&\cdots&0 \\ \vdots&\ddots&\vdots \\0&\cdots&x_{n}\end{matrix}\right)\left( \begin{matrix} y_{1}\\ \vdots \\y_{n} \end{matrix}\right)$$ 

所以可将 $$(U^{T}h)\bigodot$$ 这部分写成 $$diag(\hat{h}(\lambda_{l}))$$，

即为另一种形式：$$(f*h)_{G}=U\left( \begin{matrix} \hat{h}(\lambda_{1}) &  & \\  & \ddots &  \\  &  & \hat{h}(\lambda_{n})\end{matrix}\right)U^{T}f$$，中间 $$diag(\hat{h}(\lambda_{l}))$$ 这部分就是图卷积的参数。

#### 2.2.4 Graph Convolutional（谱）的发展
- 第一代，SCNN。
- 直接把 $$diag(\hat{h}(\lambda_{l}))$$ 变为  $$diag(\theta_{l})$$ ，通过初始化赋值、误差反向传播进行学习参数 $$\Theta=(\theta_{1},\theta_{2},...,\theta_{n})$$。加上激活函数 $$\sigma(\cdot)$$，给定输入 $$x$$ 经过图卷积输出为：$$y_{output}=\sigma\left(U\left( \begin{matrix} \theta_{1} &  & \\  & \ddots &  \\  &  & \theta_{n}\end{matrix}\right)U^{T}x\right)$$
- 从 $$y_{output}$$ 的计算可以看出，每次前向传播都需要计算 $$U$$、$$diag(\theta_{l})$$ 以及 $$U^{T}$$ 的乘积，计算复杂度是 $$O(n^{2})$$，对大规模图而言，复杂度比较高。另外，卷积核参数需要 $$n$$ 个。

> Bruna, Joan, et al. "Spectral Networks and Locally Connected Networks on Graphs." Computer Science (2013).

- 第二代，ChebNet。
- 把 $$diag(\hat{h}(\lambda_{l}))$$ 变为 $$diag(\sum_{j=0}^{K}\alpha_{j}\lambda_{l}^{j})$$。其中，$$(\alpha_{1}, \alpha_{2}, ..., \alpha_{K})$$ 为需要学习的参数，同样需要初始化赋值通过误差反向传播进行学习。具体的，$$diag(\sum_{j=0}^{K}\alpha_{j}\lambda_{l}^{j})=\left( \begin{matrix} \sum_{j=0}^{K}\alpha_{j}\lambda_{1}^{j} &  & \\  & \ddots &  \\  &  & \sum_{j=0}^{K}\alpha_{j}\lambda_{n}^{j}\end{matrix}
  \right)$$ ，假设 $$\Lambda = diag(\lambda_{1},...,\lambda_{n})$$，那么$$diag(\sum_{j=0}^{K}\alpha_{j}\lambda_{l}^{j})=\left( \begin{matrix} \sum_{j=0}^{K}\alpha_{j}\lambda_{1}^{j} &  & \\  & \ddots &  \\  &  & \sum_{j=0}^{K}\alpha_{j}\lambda_{n}^{j}\end{matrix}
  \right)=\sum_{j=0}^{K}\alpha_{j}\Lambda^{j}$$ ，又因为 $$U\sum_{j=0}^{K}\alpha_{j}\Lambda^{j}U^{T}=\sum_{j=0}^{K}\alpha_{j}U\Lambda^{j}U^{T}=\sum_{j=0}^{K}\alpha_{j}L^{j}$$，
所以对于输入 $$x$$ ，经过图卷积操作得到输出 $$y_{output}$$ 可公式化描述为：
$$y_{output}=\sigma(\sum_{j=0}^{K}\alpha_{j}L^{j} x)$$
- 卷积核参数为 $$K+1$$ 个，远小于 $$n$$ 。矩阵变换之后不需要做特征分解，直接用拉普拉斯矩阵进行变换，计算复杂度变为 $$O(n)$$。

> Defferrard, Michaël, X. Bresson, and P. Vandergheynst. "Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering." in NIPS (2016).

- 第三代，简化版的 ChebNet 。
- 参见单独文章，[在这里GCN](https://chengaf.github.io/afcheng.github.io/2019/10/09/Semi-Supervised-Classfication-with-Graph-Convolutional-Networks/) 。

> Kipf,Thomas N,and M.Welling. "Semi-Supervised Classification with Graph Convolutional Networks."inICLR(2017).

<p style="background-color:SeaGreen;">.</p>
<font face='宋体' color='0099ff'>资源君</font>



> 1. 知乎GCN：https://www.zhihu.com/search?type=content&q=GCN
> 2. 知乎如何通俗易懂地解释卷积？https://www.zhihu.com/question/22298352 [帮助理解的例子：掷两个骰子点数之和为某个数的概率、工厂排放化学物质的残留]
> 3. 谱图论：https://en.wikipedia.org/wiki/Spectral_graph_theory
> 4. 数学院博士生导师周川老师关于 Graph Neural Network 的 Slide：http://ddl.escience.cn/ff/endL





--于2019年10月2-10月4，这么多公式全手敲，脑壳疼。