---
title: Semi-Supervised Classfication with Graph Convolutional Networks
date: 2019-10-09 10:28:50
tags: [图神经网络,论文笔记]
mathjax: true
toc: true
---

关于论文 Kipf,Thomas N,and M.Welling. “Semi-Supervised Classification with Graph Convolutional Networks.” in ICLR(2017) 的笔记。

第三代GCN，基于Chebyshev多项式，使用最多的GCN。

<!--more-->

#### 关键点
- 图卷积：输入信号 $$x\in R^{N}$$， 卷积核（Filter）：$$g_{\theta}=diag(\theta)$$，$$\theta$$ 为参数。图卷积公式：$$g_{\theta}\star x=Ug_{\theta}U^{T}x$$。其中，$$U$$是正则化拉普拉斯矩阵的特征向量矩阵 $$L=I_{N}-D^{-\frac{1}{2}}AD^{-\frac{1}{2}}=U\Lambda U^{T}$$，$$\Lambda$$ 是特征值对角矩阵。理解 $$g_{\theta}$$ 为 $$L$$ 特征值的函数：$$g_{\theta}(\Lambda)$$。
- 用 $$K$$ 阶截断切比雪夫多项式 $$T_{k}(x)$$ 近似 $$g_{\theta}(\Lambda)$$：$$g_{\theta '}(\Lambda)=\sum_{k=0}^{K}\theta_{k}'T_{k}(\tilde{\Lambda})$$，
其中，$$\tilde{\Lambda}=\frac{2}{\lambda_{max}}\Lambda - I_{N}$$ 是重新缩放，$$\lambda_{max}$$ 是 $$L$$ 的最大特征值，$$\theta'$$是切比雪夫系数。切比雪夫多项式满足：$$T_{k}(x)=2xT_{k-1}(x)-T_{k-2}(x)$$，$$T_{0}(x)=1$$，$$T_{1}(x)=x$$。因此图卷积可以表示为：
$$g_{\theta'}\star x=\sum_{k=0}^{K}\theta_{k}'T_{k}(\tilde{L})x$$，其中$$\tilde{L}=\frac{2}{\lambda_{max}}L - I_{N}$$。

#### 推导
&emsp;1. 取 $$K=1$$，且 $$\lambda_{max}\thickapprox2$$。因此图卷积为：
&emsp;&emsp;&emsp;$$g_{\theta'}\star x\thickapprox \theta_{0}'T_{0}(\tilde{L})x+\theta_{1}'T_{1}(\tilde{L})x$$
&emsp;&emsp;&emsp;$$=\theta_{0}'x+\theta_{1}'\tilde{L}x$$
&emsp;&emsp;&emsp;$$=\theta_{0}'x+\theta_{1}'(L-I_{N})x$$

&emsp;2. 取 $$\theta=\theta_{0}'=-\theta_{1}'$$：
&emsp;&emsp;&emsp;$$g_{\theta'}\star x\thickapprox \theta x-\theta(L-I_{N})x$$，因为:
&emsp;&emsp;&emsp; $$L=I_{N}-D^{-\frac{1}{2}}AD^{-\frac{1}{2}}$$，所以：
&emsp;&emsp;&emsp;$$g_{\theta'}\star x\thickapprox \theta(I_{N}+D^{-\frac{1}{2}}AD^{-\frac{1}{2}})x$$

&emsp;3. 令 $$\tilde A=A+I_{N}$$，$$\tilde D_{ii}=\sum_{j}\tilde A_{ij}$$
&emsp;&emsp;&emsp;$$g_{\theta'}\star x\thickapprox\theta (\tilde D^{-\frac{1}{2}}\tilde A\tilde D^{-\frac{1}{2}}) x$$

#### 具体使用
$$F_{out} = \hat{D}^{-\frac{1}{2}} \hat{A} \hat{D}^{-\frac{1}{2}} F_{in} W$$
$$F_{out} = \hat{D}^{-\frac{1}{2}} \hat{A} \hat{D}^{-\frac{1}{2}} F_{in} W$$
$$\hat{A} = A + I$$
$$\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}$$