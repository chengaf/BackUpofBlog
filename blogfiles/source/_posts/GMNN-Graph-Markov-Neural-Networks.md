---
title: 'GMNN: Graph Markov Neural Networks'
date: 2019-09-12 06:53:56
tags: [图神经网络,论文笔记]
---

传统的针对节点分类问题（semi-suprivised object classification in relational data）的研究可分为两类方法：统计关系学习和图神经网络。前者通过条件随机场有效建模object之间的依赖关系实现分类，后者通过端对端的训练学习object的表示进而实现分类。本文融合这两类方法即结合条件随机场与图神经网络，条件随机场建模object/node标签联合分布(此篇文章中与node同义，都表示graph中的一个实体，因为在统计关系学习中常用object但是在图神经网络中习惯用node)，利用变分EM算法进行训练。

<!--more-->

**Introduction**
		现实世界中实体通过各种各样的关系相连，比如网页通过超链接相连、社交媒体用户通过朋友关系相连。Relational data 广泛存在，此类数据的研究有许多应用：实体分类、链接预测、链接分类等。

许多应用可以归纳为半监督节点分类这一基础问题，而针对半监督节点分类问题，统计关系学习（statistical relational leaning）是一类典型的方法。有代表性的工作包括关系马尔科夫网络RMN（relational markov network）以及马尔科夫逻辑网络MLN（markov logic network）。通常这些方法利用条件随机场建模object labels的依赖关系，也正是因为能够有效建模标签的依赖关系此类方法往往能够获得令人信服的半监督obeject分类效果。**但是，也存在不足：（1）此类方法通常将条件随机场中的potential functions定义为一些手工特征的线性组合，这是十分启发式的方法而且也不够高效。（2）因为object之间的复杂依赖关系，所以推断未标记object标签的后验分布仍是一个极具挑战性的问题。**

另一类方法是基于目前兴起的图神经网络。此类方法通过非线性的神经网络框架学习object的表示，可由端到端的训练方式完成框架的学习。比如，GCN [1] 利用结合自身和周围节点迭代更新节点的表示。依赖relational data的有效表示，这些方法已经被证明能够获得state-of-the-art的效果。**但是，根据节点表示进行object标签预测的过程中object之间的标签是独立的，这忽略了object之间的依赖关系。**

在本文中，提出了Graph Markov Neural Network(GMNN)，结合统计关系学习与图神经网络。GMNN不仅能够有效地学习object的表示而且还能建模object之间的依赖关系。与SRL相似，GMNN利用一个条件随机场根据object的attributes建模object标签的联合分布。模型通过变分EM框架优化，包括推断过程（E步）和学习过程（M步）。

待续......













Reference

[1] Kipf, T. N. and Welling, M. Semi-supervised classiﬁcation with graph convolutional networks. In ICLR, 2017.