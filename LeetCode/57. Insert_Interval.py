class Solution(object):
    def insert(self, intervals, newInterval):
        """
        :type intervals: List[List[int]]
        :type newInterval: List[int]
        :rtype: List[List[int]]
        """
        # 判断 插入 or 合并
        if len(intervals) == 0:
            return [newInterval]
        s = newInterval[0]
        t = newInterval[1]
        res = []
        i = 0
        while i < len(intervals):
            v = intervals[i]
            if v[1] < s: # 当前区间v直接添加入res
                res.append(v)
            else:
                if v[0] > t: #当前区间在[s,t]之后，break之后通过res.append([s,t])添加，记录位置i添加剩余的元素
                    break
                s = min(v[0], s)
                t = max(v[1], t)
            i += 1
        res.append([s,t])
        res += intervals[i:]
        return res
#=====================
‘’‘
Given a set of non-overlapping intervals, insert a new interval into the intervals (merge if necessary).
You may assume that the intervals were initially sorted according to their start times.
Example 1:
Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
Output: [[1,5],[6,9]]
Example 2:
Input: intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
Output: [[1,2],[3,10],[12,16]]
Explanation: Because the new interval [4,8] overlaps with [3,5],[6,7],[8,10].
’‘’
