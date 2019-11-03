# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 15:51:26 2019
Part-03 Programming Practice 编程练习
In our course and previous practice, 
we complete some importance components of Decision Tree. 
In this problem, you need to build a completed Decision Tree Model. 
You show finish a predicate() function, 
which accepts three parameters <gender, income, family_number>, 
and outputs the predicated 'bought': 1 or 0. (20 points)

<评阅点>

1是否将之前的决策树模型的部分进行合并组装， predicate函数能够顺利运行(8')
2是够能够输入未曾见过的X变量，
 例如gender, income, family_number 分别是： <M, -10, 1>, 模型能够预测出结果 (12')
 
@author: guliang
"""
import numpy as np
import pandas as pd
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

def predicate(gender, income, family_number, dataset, target, initial_entropy):
    def find_the_min_spliter(training_data: pd.DataFrame, target: str) -> str:
        # ->用于指示函数返回的类型
        x_fields = set(training_data.columns.tolist()) - {target}
        
        spliter = None
        min_entropy = float('inf')
        
        for f in x_fields:
            values = set(training_data[f])
            for v in values:
                sub_spliter_1 = training_data[training_data[f] == v][target].tolist()
                entropy_1 = entropy(sub_spliter_1)
                sub_spliter_2 = training_data[training_data[f] != v][target].tolist()
                entropy_2 = entropy(sub_spliter_2)
                entropy_v = entropy_1 + entropy_2
                
                if entropy_v <= min_entropy:
                    min_entropy = entropy_v
                    spliter = (f, v)
        
        print('spliter is: {}'.format(spliter))
        print('the min entropy is: {}'.format(min_entropy))
        return spliter,min_entropy
    
    info = {'gender':gender, 'income':income, 'family_number':family_number}
    
    c,r_entropy = find_the_min_spliter(dataset, target)
    sub_spliter_n = dataset[dataset[c[0]] == c[1]][target].tolist()
    if (entropy(sub_spliter_n) == 0 and info[c[0]] == c[1]):
        #输入信息正好可以找到确定解，返回结果
        bought = sub_spliter_n[0]
        return bought
    elif r_entropy >= initial_entropy:
        #信息熵不再降低，返回结果，按概率大的返回
        temp = dataset['bought'].tolist()
        counter = Counter(temp)
        a = 0
        b = 0
        for i in counter:
            m = counter[i]
            if a <= m:
                a = m
                b = i
        return b
    else:
        #信息熵降低，没有得到解，继续
        dataset = dataset[dataset[c[0]] != c[1]]
        initial_entropy = r_entropy
        return predicate(gender, income, family_number, dataset, target, initial_entropy)
    
gender = 'M'
income = '-10'
family_number = 1
initial_entropy = float('inf')    
bought_r = predicate(gender, income, family_number, dataset, 'bought', initial_entropy)   
print('result:the gender:{},income:{},family_number:{}people will be bought:{}'.format(gender, income, family_number,bought_r))

