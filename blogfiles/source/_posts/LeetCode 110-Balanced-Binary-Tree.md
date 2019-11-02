---
title: LeetCode 110. Balanced Binary Tree
date: 2019-09-09 11:33:51
tags: [LeetCode,树]
---

*平衡二叉树*

<!--more-->

题目：

​		Given a binary tree, determine if it is height-balanced.

​		For this problem, a height-balanced binary tree is defined as:

> a binary tree in which the depth of the two subtrees of *every* node never differ by more than 1.

解：1) 简单思路（暴力）：求左右子树深度，判断深度差。LeetCode104（求树的深度）基础上，加一层判断。

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if root == None:
            return True
        else:
            lh = self.get_h(root.left)
            rh = self.get_h(root.right)
            if lh - rh > 1 or rh -lh > 1:
                return False
            else:
                return self.isBalanced(root.left) and self.isBalanced(root.right)
        
    def get_h(self, root):
        if root == None:
            return 0
        else:
            return max(self.get_h(root.left), self.get_h(root.right)) + 1
```

解：2) 在暴力解基础上，增加一个返回值，get_h函数做递归得到深度的同时就直接判断是否为平衡二叉树，如果有一个高度差超过1那就直接返回False。与1)不同的点:1)中相当于从整体到局部，从根节点一直遍历到最后叶子结点，没走一层都要从当前层遍历到叶子加点。时间复杂度为O(N*logN)；2) 从根节点遍历到叶子节点，在每一层就立马做出判断，时间复杂度O(N)。

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        f, _ = self.get_h(root)
        return f
        
    def get_h(self, root):
        if root == None:
            return True, 0
        fl, l = self.get_h(root.left)
        fr, r = self.get_h(root.right)
        if fl == False or fr == False:
            return False, 0
        return abs(l-r) <= 1, max(l,r) + 1

```

