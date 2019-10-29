# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 23:06:27 2019
Part-1 Programming Review 编程回顾
@author: us
"""

#1. Re-code the Linear-Regression Model using scikit-learning(10 points)=======

import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LinearRegression

random_data = np.random.random((20, 2))
X = random_data[:, 0]
y = random_data[:, 1]

def assmuing_function(x):
    # 在我们的日常生活中是常见的
    # 体重 -> 高血压的概率
    # 收入 -> 买阿玛尼的概率
    # 其实都是一种潜在的函数关系 + 一个随机变化
    return 13.4 * x + 5 + random.randint(-5, 5)

y = [assmuing_function(x) for x in X]
plt.scatter(X, y)
y = np.array(y)
reg = LinearRegression().fit(X.reshape(-1, 1), y)
reg.score(X.reshape(-1, 1), y)
reg.coef_
reg.intercept_
def f(x): 
    return reg.coef_ * x + reg.intercept_
plt.scatter(X, y)
plt.plot(X, f(X), color='red')

#2. Complete the unfinished KNN Model using pure python to solve the previous Line-Regression problem. (8 points)
from scipy.spatial.distance import cosine

def model(X, y):
    return [(Xi, yi) for Xi, yi in zip(X, y)]
def distance(x1, x2):
    return cosine(x1, x2)
def predict(x, k=5):
    most_similars = sorted(model(X, y), key=lambda xi: distance(xi[0], x))[:k]
    print(most_similars)
    # 已经获得了最相似的数据集
    # 然后呢，Counter() -> most_common() -> 就可以获得出现最多的这个y了 
    sum_sim = 0
    for sim in most_similars:
        sum_sim = sum_sim + sim[1]
    pre_y = sum_sim/k 
    return pre_y
predict(2)


