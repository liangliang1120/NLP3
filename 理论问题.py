# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 09:41:53 2019
理论问题
@author: us
"""

'''
Part-2 Question and Answer 问答
1. What's the model? why all the models are wrong, but some are useful? (5 points)
Ans:模型是对现实情况的抽象，仅考虑有限的影响因素，
    不能把所有因素考虑进来，所以从这个角度来看模型是错的。
    但是研究问题时只要将显著的特征研究清楚，就能解决一些实际问题，所以是有用的。
    
2. What's the underfitting and overfitting? List the reasons that could make model overfitting or underfitting. (10 points)
Ans:underfitting：模型只学习了样本的一部分特点，导致在真实模型应用中产生很多错误的预测，
        在训练集中变现得不好，这种情况就是欠拟合
    overfitting：在训练集中表现的好，在测试集中表现的不好。
    underfitting原因：模型太简单。特征维度过少，导致拟合的函数无法满足训练集，误差较大。
    overfitting原因：模型太复杂，数据太少，数据分布不对，模型系数过大。
        特征维度过多，导致拟合的函数完美的经过训练集，但是对新数据的预测结果则较差。

3. What's the precision, recall, AUC, F1, F2score. What are they mainly target on? (12')
Ans:precision=tp/(tp+fp) 查准率，应用场景－当你想知道“挑出的西瓜中有多少比例是好瓜”
    recall = tp/(tp+fn) 查全率，当你想知道“所有好瓜中有多少比例被挑出来了” 
    AUC：roc（receiver operating characteristic curve），
        roc曲线上每个点反映着对同一信号刺激的感受性，
        AUC是ROC曲线下的面积。是侧重反映敏感性和特异性连续变量的综合指标
        在样本分布非常不均匀时使用
    F1=2*precision*recall/(precision+recall) 综合评价指标,把precision和recall综合考虑
    F2-Score:
    F-score=（1+β^2）*precision*recall/(β^2*(precision+recall))，
        是综合考虑Precision和Recall的调和值
        当有些情况下我们认为精确率更为重要，那就调整 β 的值小于 1 ，
        如果我们认为召回率更加重要，那就调整 β的值大于1，比如F2-Score。

4. Based on our course and yourself mind, what's the machine learning? (8')
Ans:机器学习是一个研究领域，使计算机无需明确编程即可学习。
    机器学习教一台机器来解决难以通过算法解决的各种复杂任务。
    在传统编程中，需要对程序的行为进行硬编码。在机器学习中，是将大量内容留给机器去学习数据。
    ML 不是替代品，而是传统编程方法的补充。
    思维方式，机器学习通过统计信息而非逻辑来分析结果。

5. "正确定义了机器学习模型的评价标准(evaluation)， 问题基本上就已经解决一半". 这句话是否正确？你是怎么看待的？ (8‘)
Ans:评价标准可以说明模型的性能，辨别模型的结果。
    我们建立一个模型后，计算指标，从指标获取反馈，再继续改进模型，直到达到理想的准确度。
    在预测之前检查模型的准确度至关重要，否则可能会忽视模型的缺陷，造成无法训练出有效的模型。
    

'''
