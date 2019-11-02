---
title: GCN原理简略思路
date: 2019-10-30 15:24:01
tags: 图神经网络
mathjax: true
toc: true
---

写在前面，GCN提出目的、输入输出、方法分类。

> 1. GCN提出目的：在Graph这一广泛使用的数据结构上做深度学习。传统CNN无法直接迁移到Graph上，因此提出GCN这一研究体系/方法框架。
> 2. GCN方法分类：时域、频域（有理论保证是目前的主流方法，这里的总结）。
> 3. 输入：Graph的邻接矩阵 $$A\in R^{N\times N}$$ ,节点的特征矩阵 $$X\in R^{N\times F}$$ 。
> 4. 输出：节点的低维向量表示 $$F_{output}\in R^{N\times d}$$，输出 $$F_{output}$$ 是融合了结构信息的向量表示。在 $$F_{output}$$ 基础上可以再接对应task的损失函数。
> 5. GCN最终干了个啥，就是个根据网络全局节点信息的一个节点表示过程。

<!--more-->



基于谱的GCN的总结思路

> 1. 在频域进行操作，首先要将节点特征转化到频域。时域到频域的相互转换是利用傅立叶变换 $$F(\cdot)$$ 与逆变换 $$F^{-1}(\cdot)$$ 进行。
> 2. 卷积定理：频域乘积等于时/空域卷积。因此，要求时域卷积只需要求转换到频域的乘积，实际上也就是要在频域做特征提取（频域的频谱相关）。
> 3. 因此时域卷积 $$g_{\theta}\star x$$ 操作的流程是：将输入特征利用傅立叶变换转化 $$F(\cdot)$$ 到频域 $$F(x)$$ ； 然后在频域对信号进行特征提取，如果假设特征提取函数是 $$g_{\theta}$$ ，那么提取之后的特征可以表示为 $$g_{\theta}F(x)$$ ，乘积操作是因为频域乘积等于时域卷积；最后利用傅立叶逆变换将提取到的特征转回到时域得到最终的卷积结果 $$F^{-1}(g_{\theta}F(x))$$ 。那么需要解决的问题就是：**1. Graph上的傅立叶变换 $$F(\cdot)$$ 与逆变换 $$F^{-1}(\cdot)$$ 的定义；2. 特征提取函数 $$g_{\theta}$$ （频域的频谱相关）的定义。**
> 4. 从传统傅立叶变换到Graph上的傅立叶变换。
> 5. 从频域研究Graph的体系（谱图论）中，如何定义Graph的频谱和基。
> 6. 因此一下思路：定义Graph的频谱和基（Graph傅立叶变换中会用到，所以先定义）--> Graph上傅立叶变换 --> GCN的发展（从原始版本到最终得以广泛应用的版本，共三版）



#### 1. Graph上的频谱和基

谱图论：借助拉普拉斯矩阵的特征值和特征向量对Graph的性质进行研究的一套方法体系。特征值就相当于图的频谱，特征向量相当于频域的基。（恩...这就提出了图拉普拉斯矩阵的特征分解）

拉普拉斯矩阵定义：$$L=I_{N}-D^{-\frac{1}{2}}AD^{-\frac{1}{2}}$$  （拉普拉斯矩阵的一种形式）。

特征分解：$$L=U\Lambda U^{-1}$$ ，其中 $$U=[u_{1},u_{2},...,u_{N}]$$ 为特征向量矩阵，$$u_{1}$$ 为列向量，$$\Lambda$$ 特征值对角矩。

对Graph而言，特征值 $$\Lambda$$ 是频谱，特征向量 $$U$$ 是基。

#### 2. Graph上的傅立叶变换与逆变换

##### 2.1 Graph上的傅立叶变换

传统傅立叶变换定义：$$F(w)=\int_{-\infty}^{+\infty}f(t)e^{-iwt}dt$$ ，信号 $$f(t)$$ 与基函数 $$e^{-iwt}$$ 的积分，$$w$$ 为实数。这里的基函数 $$e^{-iwt}$$ 是拉普拉斯算子的特征方程，实数 $$w$$ 和特征值相关，基函数 $$e^{-iwt}$$ 与和 $$w$$ 相关的特征向量相关（就是和特征值对应的特征向量相关）。

对于Graph数据而言，节点离散，节点 $$i$$ 输入特征用 $$f(i)$$ 表示（节点的特征矩阵 $$X\in R^{N\times F}$$ 的一行），Graph上的傅立叶变换可定义为：$$F(\lambda_{l})=\hat{f}(\lambda_{l})=\sum_{i=1}^{N}f(i)u_{l}(i)$$ ，展开用矩阵形式表示：

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

即：$$f$$ 在Graph上的傅立叶变换为 $$\hat{f}=U^{T}f$$ ，变换之后就是 $$\hat{f}$$ 了。

##### 2.2 Graph上的傅立叶逆变换

传统傅立叶逆变换定义：$$F^{-1}(F(w))=\frac{1}{2\pi}\int F(w)e^{-iwt}dw$$ ，也就是反过来了。Graph上定义为：$$f(i)=\sum_{i=1}^{N}\hat{f}(\lambda_{i})u_{l}(i)$$ ，推广到矩阵形式：

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

即：Graph上的傅立叶变换为 $$f=U\hat{f}$$。

#### 3. Graph上卷积定义

假设频域特征提取函数为 $$g_{\theta}$$ ，节点Feature特征矩为 $$X$$ ，Graph卷积可以数学描述为：$$Ug_{\theta}U^{T}X$$ 这也是一层卷积。其中 $$g_{\theta}$$ 与特征值相关，可以写成 $$g_{\theta}(\Lambda)$$ , $$\theta$$ 为需要学习的参数。然后就是如何定义函数 $$g_{\theta}(\Lambda)$$ 了，涉及三代GCN的演变，复杂度在逐渐降低。

<p style="background-color:SeaGreen;">.</p>
> 本文是大体思路，细节参考[Graph-Convolutional-Network-原理解释](https://chengaf.github.io/afcheng.github.io/2019/10/09/Graph-Convolutional-Network-%E5%8E%9F%E7%90%86%E8%A7%A3%E9%87%8A/)，以及[第三代GCN](https://chengaf.github.io/afcheng.github.io/2019/10/09/Semi-Supervised-Classfication-with-Graph-Convolutional-Networks/) 。

