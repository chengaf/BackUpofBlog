---
title: LeetCode 101. Symmetric Tree
date: 2019-09-10 00:59:08
tags: [LeetCode,树]
---

题目：Given a binary tree, check whether it is a mirror of itself (ie, symmetric around its center).

<!--more-->

解：深度优先搜索

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        return self.bfs(root, root)
    
    def bfs(self, L, R):
        if not L and not R:
            return True
        elif not L or not R:
            return False
        return L.val == R.val and self.bfs(L.left, R.right) and self.bfs(L.right, R.left)
```

