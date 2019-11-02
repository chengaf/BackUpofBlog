---
title: LeetCode 104. Maximum Depth of Binary Tree
date: 2019-09-09 10:59:38
tags: [LeetCode,树]
---

*二叉树深度-递归*

<!--more-->

题目：

​		Given a binary tree, find its maximum depth.
​		The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

解：基础递归

代码 (python)：

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root == None:
            return 0
        return max(self.maxDepth(root.left) + 1, self.maxDepth(root.right) + 1)
```

