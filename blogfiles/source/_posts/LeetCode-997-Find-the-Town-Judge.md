---
title: LeetCode 997. Find the Town Judge
date: 2019-10-16 14:48:33
tags: [LeetCode,图]
---



#### 题目描述

找到小镇法官

在一个小镇里，按从 1 到 N 标记了 N 个人。传言称，这些人中有一个是小镇上的秘密法官。

如果小镇的法官真的存在，那么：

小镇的法官不相信任何人。
每个人（除了小镇法官外）都信任小镇的法官。
只有一个人同时满足属性 1 和属性 2 。
给定数组 trust，该数组由信任对 trust[i] = [a, b] 组成，表示标记为 a 的人信任标记为 b 的人。

如果小镇存在秘密法官并且可以确定他的身份，请返回该法官的标记。否则，返回 -1。



<!--more-->



#### 输入输出示例

示例 1：
输入：N = 2, trust = [[1,2]]
输出：2

示例 2：
输入：N = 3, trust = [[1,3],[2,3]]
输出：3

示例 3：
输入：N = 3, trust = [[1,3],[2,3],[3,1]]
输出：-1

示例 4：
输入：N = 3, trust = [[1,2],[2,3]]
输出：-1

示例 5：
输入：N = 4, trust = [[1,3],[1,4],[2,3],[2,4],[4,3]]
输出：3

#### 提示：

> 1. 1 <= N <= 1000
> 2. trust.length <= 10000
> 3. trust[i] 是完全不同的
> 4. trust[i][0] != trust[i][1]
> 5. 1 <= trust[i][0], trust[i][1] <= N

#### 解

一、暴力解法，步骤：

> 1. 统计被信任人（信任对的第二个标记）的被信任次数（出现在信任对的第二个位置就被信任一次）。
> 2. 如果某人被信任N-1次且不信任任何人（从未出现在信任对的第一个位置），则TA为法官。
> 3. 特殊情况：只有一个人且信任对列表为空。

```python
class Solution:
    def findJudge(self, N: int, trust: List[List[int]]) -> int:
        # 只有一个人
        if N==1 and len(trust)==0:
            return 1
        # 建一个dictionary，key是标记-value是被信任次数
        # 法官被信任次数是N-1
        mark_trusted_num = {}
        trust_other_set = set()
        for i in trust:
            if i[1] in mark_trusted_num:
                mark_trusted_num[i[1]] += 1
            else:
                mark_trusted_num[i[1]] = 1
            trust_other_set.add(i[0])
        for k, v in mark_trusted_num.items():
            # 法官不信任任何人
            if v == N-1 and k not in trust_other_set:
                return k
        return -1
```

注：需要额外的一个dictionary、一个set。

二、图角度，步骤：

> 1. 记录N个人的出入度。
> 2. 如果出度为0，入度为N-1则返回当前值。
> 3. 没有满足条件的，则返回-1。

```python
class Solution:
    def findJudge(self, N: int, trust: List[List[int]]) -> int:
        # 只有一个人
        if N==1 and len(trust)==0:
            return 1
        count_in = [0]*(N+1)
        count_out = [0]*(N+1)
        for i in trust:
            count_in[i[1]] += 1
            count_out[i[0]] += 1
        for i in range(1, N+1):
            if count_in[i] == N-1 and count_out[i]==0:
                return i
        return -1
```

