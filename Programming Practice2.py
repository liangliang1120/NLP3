# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 18:02:11 2019
将上一节课(第二节课)的线性回归问题中的Loss函数改成"绝对值"，
并且改变其偏导的求值方式，观察其结果的变化。(19 point)

是否将Loss改成了“绝对值”(3')
是否完成了偏导的重新定义(5')
新的模型Loss是否能够收敛 (11’)
@author: guliang
"""

from sklearn.datasets import load_boston
import random
import matplotlib.pyplot as plt

dataset = load_boston()
x,y=dataset['data'],dataset['target']
# dataset['DESCR']
X_rm = x[:,5]

# plot the RM with respect to y
plt.scatter(X_rm,y)

#define target function
def price(rm, k, b):
    return k * rm + b
'''
Define mean square loss
loss=1/n*∑|yi−yi^|
loss=1/n*∑|yi−(kxi+bi)|
'''
# define loss function 
def loss(y,y_hat):
    return sum(abs(y_i - y_hat_i) for y_i, y_hat_i in zip(list(y),list(y_hat)))/len(list(y))

'''
Define partial derivatives
分段函数求导：
化简后合并
∂loss∂k=−1/n∑(yi−yi^)xi
∂loss∂b=−1/n∑(yi−yi^)
'''
# define partial derivative 
def partial_derivative_k(x, y, y_hat):
    n = len(y)
    gradient = 0
    for x_i, y_i, y_hat_i in zip(list(x),list(y),list(y_hat)):
        gradient += (y_i-y_hat_i) * x_i
    return -1/n * gradient

def partial_derivative_b(y, y_hat):
    n = len(y)
    gradient = 0
    for y_i, y_hat_i in zip(list(y),list(y_hat)):
        gradient += (y_i-y_hat_i)
    return -1 / n * gradient

k = random.random() * 200 - 100  # -100 100
b = random.random() * 200 - 100  # -100 100


learning_rate = 1e-3

iteration_num = 200 
losses = []
for i in range(iteration_num):
    
    price_use_current_parameters = [price(r, k, b) for r in X_rm]  # \hat{y}
    
    current_loss = loss(y, price_use_current_parameters)
    losses.append(current_loss)
    print("Iteration {}, the loss is {}, parameters k is {} and b is {}".format(i,current_loss,k,b))
    
    k_gradient = partial_derivative_k(X_rm, y, price_use_current_parameters)
    b_gradient = partial_derivative_b(y, price_use_current_parameters)
    
    k = k + (-1 * k_gradient) * learning_rate
    b = b + (-1 * b_gradient) * learning_rate
best_k = k
best_b = b

plt.plot(list(range(iteration_num)),losses)

price_use_best_parameters = [price(r, best_k, best_b) for r in X_rm]
plt.scatter(X_rm,y)
plt.scatter(X_rm,price_use_current_parameters)











