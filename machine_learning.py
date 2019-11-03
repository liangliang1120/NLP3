# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 23:06:27 2019
Part-1 Programming Review 编程回顾
@author: guliang
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
import pandas as pd

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
#3. Re-code the Decision Tree, which could sort the features by salience. (12 points)
from collections import Counter
def entropy(elements):
    #计算信息熵 群体的混乱程度
    counter = Counter(elements)
    probs = [counter[c] / len(elements) for c in set(elements)]
    return - sum(p * np.log(p) for p in probs)

# entropy([1, 1, 1, 1])
# entropy([1, 1, 1, 0])
mock_data = {
    'gender':['F', 'F', 'F', 'F', 'M', 'M', 'M'],
    'income': ['+10', '-10', '+10', '+10', '+10', '+10', '-10'],
    'family_number': [1, 1, 2, 1, 1, 1, 2],
    'bought': [1, 1, 1, 0, 0, 0, 1],
}
dataset = pd.DataFrame.from_dict(mock_data)
sub_split_1 = dataset[dataset['family_number'] == 1]['bought'].tolist()
sub_split_2 = dataset[dataset['family_number'] != 1]['bought'].tolist()
_sub_split_1 = dataset[dataset['gender'] != 'F']['bought'].tolist()
_sub_split_2 = dataset[dataset['gender'] != 'M']['bought'].tolist()

r1 = entropy(_sub_split_1) + entropy(_sub_split_2)
r2 = entropy(sub_split_1) + entropy(sub_split_2)
# r2比较小，家庭收入作为区分标准更好

from icecream import ic
def find_the_min_spilter(training_data: pd.DataFrame, target: str) -> str:
    # ->用于指示函数返回的类型
    x_fields = set(training_data.columns.tolist()) - {target}
    
    spliter = None
    min_entropy = float('inf')
    
    for f in x_fields:
        ic(f)
        values = set(training_data[f])
        ic(values)
        for v in values:
            sub_spliter_1 = training_data[training_data[f] == v][target].tolist()
            ic(sub_spliter_1)
            entropy_1 = entropy(sub_spliter_1)
            ic(entropy_1)
            sub_spliter_2 = training_data[training_data[f] != v][target].tolist()
            ic(sub_spliter_2)
            entropy_2 = entropy(sub_spliter_2)
            ic(entropy_2)
            entropy_v = entropy_1 + entropy_2
            ic(entropy_v)
            
            if entropy_v <= min_entropy:
                min_entropy = entropy_v
                spliter = (f, v)
    
    print('spliter is: {}'.format(spliter))
    print('the min entropy is: {}'.format(min_entropy))
    
    return spliter
find_the_min_spilter(dataset, 'bought')
find_the_min_spilter(dataset[dataset['family_number'] == 1], 'bought')
dataset1 = dataset[dataset['family_number'] == 1]
find_the_min_spilter(dataset1[dataset1['income'] == '+10'], 'bought')

# 之后无法分割,熵不再下降


# 4. Finish the K-Means using 2-D matplotlib (8 points)========================
from sklearn.cluster import KMeans

X = [random.randint(0, 100) for _ in range(100)]
Y = [random.randint(0, 100) for _ in range(100)]

plt.scatter(X, Y)

tranning_data = [[x, y] for x, y in zip(X, Y)]
cluster = KMeans(n_clusters=6, max_iter=500)
cluster.fit(tranning_data)
cluster.cluster_centers_
cluster.labels_
from collections import defaultdict
centers = defaultdict(list)
for label, location in zip(cluster.labels_, tranning_data):
    centers[label].append(location)
    
color = ['red', 'green', 'grey', 'black', 'yellow', 'orange']

for i, c in enumerate(centers):
    for location in centers[c]:
        plt.scatter(*location, c=color[i])
        
for center in cluster.cluster_centers_:
    plt.scatter(*center, s=100)


