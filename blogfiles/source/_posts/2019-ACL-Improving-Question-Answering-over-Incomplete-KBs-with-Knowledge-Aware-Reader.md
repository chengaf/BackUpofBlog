---
title: 2019 ACL Improving Question Answering over Incomplete KBs with Knowledge-Aware Reader
date: 2019-10-16 13:57:06
tags: [自然语言处理,论文笔记,问答]
mathjax: true
toc: true
---



##### 论文信息

Title: Improving Question Answering over Incomplete KBs with Knowledge-Aware Reader

Wenhan Xiong † , Mo Yu ∗ , Shiyu Chang ∗ , Xiaoxiao Guo ∗ , William Yang Wang †
†University of California, Santa Barbara ∗IBM Research

<!--more-->



##### 摘要

提出端到端的问答模型，模型从不完备的知识库和查询到的文本片段聚集答案证据。结构化的知识库更易查询并且获取到的知识可以帮助理解非结构化的文本。模型首先从问题相关的知识库子图中获取知识实体（knowledge entities），然后在隐空间中重新定义问题、读取以获取的实体知识。知识库以及文本的证据最终会结合起来预测答案。在KBQA测评WebQSP上实验，并获得一定的提升。

##### 立意

知识库被认为是回答事实性问题的重要资源。但是使用精心设计的、复杂的架构去构架知识库需要大量的人力，这不可避免的限制了知识库的覆盖范围。事实上，知识库通常是不完整的，不足以覆盖开放域问题所需要的全部证据。
另一方面，互联网上大量的非结构化文本能够覆盖的知识域非常广泛，这些知识也经常被用于开放域问答。因此为了提高知识库的覆盖范围，最直接的就是通过文本数据扩大知识库。近来，基于文本的QA模型通过处理单篇文档（文档中保证包含答案）已经取得了卓越的性能。但是当纳入多文档时，它们仍然存在不足。我们假设这是由于在区分相关和不相关信息时缺乏对应的背景知识。图1举例说明文本知识和不完整知识库的组合必要性。
为了更好的利用文本信息来改进不完备知识库的问答，本文提出一个新的端到端的模型，包含（1）一个简单而且有效的子图读取器，该子图阅读器从一个问题相关的知识库子图积累每个知识库实体的知识; （2）知识感知的文本阅读器，该文本阅读器通过一个新颖的条件门控机制有选择地合并所学的有关实体的KB知识。通过特别设计的门控函数，模型能够在编码问题和段落时动态地决定合并多少KB知识，从而使结构化知识和文本信息更兼容。

##### 任务定义

这里的问答任务需要阅读基于三元组的知识$$K=\{(e_{s},r,e_{o})\}$$和查询的Wikipedia文档$$D$$。每个问题只考虑一个subgraph，这个subgraph是通过personalized pagerank来选取，主题实体来源于问题 $$E_{0}=\{e|e\in Q\}$$。文档 $$D$$是通过已有的文档检索器获得并通过Lucene 索引排序。文档中的实体已有注释并且链接到KB实体。对于每一个问题，模型从一个包含所有KB和文档实体的候选集检索答案实体。

##### 模型

###### SubGraphReader-SGREADER
利用图注意力机制从与实体连接的邻居$$N_{e}$$中积累每个子图的实体$$e$$。图注意力机制是考虑两个重要方面所设计的：（1）邻居实体是否和问题相关;（2）邻居实体是不是问题的主题实体。通过传播，SGREADER最终输出每个实体的向量表示，编码知识由其与其相连的邻居指示（知识通过与之相连的邻居进行了编码）。

- Question-Relation Matching
  为了能够在同构隐藏空间中匹配问题和KB关系，使用一个共享的LSTM去编码问题$$\{w_{1}^{q},w_{2}^{q},...,w_{l_{q}}^{q}\}$$和标记的关系$$\{w_{1}^{r},w_{2}^{r},...,w_{l_{r}}^{r}\}$$ 利用生成的问题隐层状态$$\mathbf{h^{q}}\in R^{l_{q}*d_{h}}$$以及关系隐层状态$$\mathbf{h^{r}}\in R^{l_{r}*d_{h}}$$ 首先利用self-attention encoder获取关系的表示：
  $$\vec{r}=\sum_{i}\alpha_{i}\vec{h_{i}^{r}}, \alpha_{i}\propto exp(\vec{w_{r}}\cdot \vec{h_{i}^{r}})$$
  因为每个问题匹配多个关系，但是每个关系只描述问题的某一部分，因此需要更细粒度地计算关系-问题的匹配得分。
  $$s_{r}=\vec{r}\cdot \sum{j}\beta_{j}\vec{h_{j}^{q}}, \beta_{j}\propto exp(\vec{r}\cdot \vec{h_{j}^{q}})$$
- Extra Attention over Topic Entity Neighbors
  利用主题实体从新计算关系-实体得分$$\hat{s}_{(r_{i},e_{i})}$$，如果一个主题实例$$e$$连接的邻居$$(r_{i}, e_{i})$$出现在了问题中$$I[e_{i}\in E_{0}]$$，那么对应 KB 中的三元组$$(e,r_{i}, e_{i})$$将会比那些非主题实例的邻居对问题的语义表示更有用，因此在邻居节点上的注意力值最终表示为：
  $$\hat{s}_{(r_{i},e_{i})}\propto exp(I[e_{i}\in E_{0}]+s_{r_{i}})$$
- Information Propagation from Neighbors
  为了从链接的三元组中积累知识，对每个实体$$e$$定一个传播规则：
  $$\vec{e^{'}}=r^{e}\vec{e}+(1-r^{e})\sum_{e_{i},r_{i}\in N_{e}}\hat{s}_{r_{i},e_{i}}\sigma(W_{e}[\vec{r_{i}};\vec{e_{i}}])$$
  $$r_{e}$$是一个trade-off参数，通过一个线性门函数定义：
  $$r_{e}=g(\vec{e},\sum_{e_{i},r_{i}}\in N_{e}) \hat{s}_{(r_{r},e_{i})}\sigma(W_{e}[\vec{r_{i}};\vec{e_{i}])}$$，用$$r_{e}$$来控制从源表示$$\vec{e}$$保留多少信息。

###### Knowledge-Aware Text Reader - KAREADER
上一个模块SGREADER中获取了KB的表示，这里利用它加深对问题和文档的理解，用了个已有的阅读理解模型。(With the learned KB embeddings, our model enhances text reading with KAREADER . Briefly, we use an existing reading comprehension model (Chen et al., 2017) and improve it by learning more knowledge-aware representations for both question and documents.)

- Query Reformulation in Latent Space
  对问题进行更新表示，收集关于问题的主题实体，将这些信息用门控机制融合。原始的问题表示$$\mathbf{h^{q}}$$ 利用self-attention机制获取问题的stand-alone表示$$\vec{q}=\sum_{i}b_{i}\vec{h_{i}^{q}}$$。问题的主题实体知识$$\vec{e^{q}}=\sum_{e\in E_{0}}\vec{e'}/|E_{0}|$$，然后利用门控机制获取问题表示$$\vec{q'}=\gamma^{q}\vec{q}+(1-\gamma^{q})tanh(\mathbf{W^{q}}[\vec{q},\vec{e^{q}},\vec{q}-\vec{e^{q}}])$$，其中，$$\mathbf{W^{q}}\in R^{h_{d}\times 3h_{d}}$$，linear gate：$$\gamma^{q}=sigmoid(\mathbf{W^{gq}}[\vec{q},\vec{e^{q}},\vec{q}-\vec{e^{q}}])$$。
  也就是，利用上一模块更新的主题实体表示更新问题的表示。
- Knowledge-aware Passage Enhancement
  上一步得到问题的更新表示，对文档进行更新表示。没有用标准的门控机制，为了允许模型动态选择跟问题相关的输入，提出一种根据问题表示的门控机制，进而增强对文档的理解，获得更好的文档语义表示（我理解也就是问题相关的文档表示）。段落$$w_{i}^{d}$$特征$$\vec{f_{w_{i}}^{d}}$$以及链接的实体$$e_{w_{i}}$$，$$\vec{e'_{w_{i}}}$$是SGRREADER学习的表示。条件门控机制定义为：
  $$\vec{i_{w_{i}}^{d}}=\gamma^{d}\vec{e'_{w_{i}}}+(1-\gamma^{d})\vec{f_{w_{i}}^{d}}$$
  $$\gamma^{d}=sigmoid(\mathbf{W^{gd}}[\vec{q}\cdot\vec{e'}_{w_{i}};\vec{q}\cdot\vec{f_{w_{i}}^{d}}])$$
- Entity Info Aggregation from Text Reading
  实体信息聚合，将从Text Reader中得到的信息进行融合，首先使用一个co-attention计算问题$$\vec{q'}$$和bi-lstm的隐层状态$$\vec{h_{w_{i}}^{d}}$$的相关程度$$\lambda_{i}\vec{q'}^{T}\vec{h}_{w_{i}}^{d}$$，然后对这些隐层状态进行加权和$$\vec{d}=\sum_{i}\lambda_{i}\vec{h}^{d}_{w_{i}}$$，对于文档对应的实例，使用均值池化得到最后的表示，对一个实体$$e$$以及所有包含$$e$$的文档$$D^{e}={d|e\in d}$$，那这个出现在多篇文档的实体最终的表示为$$\vec{e}_{d}=\frac{1}{|D^{e}|}\sum_{d\in D^{e}}\vec{d}$$。
###### Answer Prediction
  模型得到实体的表示$$\vec{e'},\vec{e^{d}}$$使用一个非线性变化和sigmoid函数来求得每个实例是否是答案的概率。$$s^{e}=\sigma_{s}(\vec{q'}^{T}\mathbf{W_{s}}[\vec{e'};\vec{e^{d}}])$$

##### 总结

问答系统，尤其是开放性的问答系统需要非常多的先验知识来辅助答案的预测，虽然我们可以通过知识库来整合一些先验知识，但毕竟无法覆盖到所有的情况。作者通过知识库和大规模网络文本的相互辅助，从而提升了模型的整体性能。
同时我们也可以看到，知识库的使用正在变得越来越普及，无论是问答系统，对话，推理还是常识理解，都将知识库作为一个非常好的先验信息源，因此知识库的使用也变得越来越重要，非常值得关注一下。

