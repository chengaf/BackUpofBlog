---
title: PageRank and RWR
date: 2019-09-09 23:14:42
tags: 基础算法
mathjax: true
---


*a. 一篇paper里用RWR (random walk with restart)，得到网络Network中每个节点的初始化特征。
b. RWR以及PageRank的简单理解*

<!--more-->

- #### 符号化表示

  1, $$N$$ 个网页（Network中$$N$$个节点），
  2, 归一化的网页间链接关系（节点间的邻接关系）$$A\in R^{N*N}$$(可以类似马氏链的状态转移矩阵)。

  - **PageRank** 

    最原始PageRank要得到网页的重要性(被用户访问到的可能性) 用向量 $$B=(b_{1},b_{2},...,{b_{N}})^{T}$$ 表示。$$B$$ 可根据如下公式计算:
    $$B_{t}=B_{t-1}\ A \ \ \ 公式(1)$$ 
    $$t$$ 时刻的网页重要性$$B_{t}$$ 取决于$$t-1$$ 时刻的 $$B_{t-1}$$ 以及链接关系$$A$$ 。如果$$B_{t}=B_{t-1}A=B_{t-1}$$ ,则此时的 $$B_{t+1}=B_{t}=B_{t-1}$$ 称为马尔科夫链的平稳分布(stationary distribution)，PageRank需要的网页被用户看到的概率就是这个平稳分布。

  - **Personalized PageRank**

    考虑到将用户个人偏好(用户肯定是想访问啥访问啥)，所以在公式(1)基础上除了考虑链接关系得再加一项考虑以一定概率随机访问其它所有页面，公式化描述如下：
    $$B_{t} = c\ B_{t-1}\ A+(1-c)\ p\ \ \ 公式(2)$$ 
    其中，$$ 1-c$$ 表示用户不按照链接关系随机访问其它页面的概率，$$p=(\frac{1}{N},\frac{1}{N},...)^{T}$$ 表示均匀的访问其它页面。

  - **Random Walk with Restart** 

    与Personalized PageRank是一个东西吧，用起来一样。