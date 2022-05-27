#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   snippets.py
@Time    :   2022/05/25 12:45:05
@Author  :   csong-idea
@Email   :   songchao@idea.edu.cn
@Copyright : International Digital Economy Academy (IDEA)
'''


# 字典排序
scores = {
    'joe': 85,
    'jane': 90,
    'alex': 80,
    'beth': 91
}

students = list(scores.keys())
sorted(students, key=scores.get, reverse=True)


# 嵌套遍历
words = ['hello', 'world']
nums = []
for word in words:
    for ch in word:
        nums.append(ord(ch))
print(nums)
# [104, 101, 108, 108, 111, 119, 111, 114, 108, 100]

[ord(ch) for word in words for ch in word]        
# [104, 101, 108, 108, 111, 119, 111, 114, 108, 100]


# 列表拆分
nums = [1, 2, 3, 4, 5]

first, *reset = nums
print(f'{first=}, {reset=}')
# first=1, reset=[2, 3, 4, 5]

first, *middle, last = nums
print(f'{first=}, {middle=}, {last=}')
# first=1, middle=[2, 3, 4], last=5

a, b, *c, d = nums
print(f'{a=}, {b=}, {c=}, {d=}') 
# a=1, b=2, c=[3, 4], d=5