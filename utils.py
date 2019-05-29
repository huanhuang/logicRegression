
# coding: utf-8
import numpy as np

def fl_score(logits,labels):
    '''
    Function:
        计算各分类的fl_score值
    Arguments:
    logits-- 预测值
    labels -- 真实值
    '''
    class_list = set(labels)
    for y in class_list:
        TP = np.sum(((labels == logits) & (logits == y)))
        n_pred_pos = np.sum(logits==y)
        n_real_pos = np.sum(labels==y)
        precision = recall = flscore = 0
        precision = TP/n_pred_pos
        recall = TP/n_real_pos
        fl_score = (2*precision*recall)/(precision+recall)
        print(f"{y} 类的fl_score 为:{fl_score}")



def calc_accuracy_class(y_pred,y):   
    '''
    Function:
    计算分类正确率
    Arguments:
    y_pred---预测值
    y -- 真实值
    Return:
    正确率
    '''
    y = y.reshape(-1,1)
    m = y.shape[0]
    correct_num = np.sum((np.squeeze(y_pred)==np.squeeze(y)))
    return correct_num/m






