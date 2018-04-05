import mxnet as mx

"""
precision buy: of the times the model says the user will buy, what percentage are  correct?
recall buy: of the times the user would have bought the item, what percentage did the model say they would buy?
"""

import  numpy as np

def classifer_metrics(label, pred):
    """
    compute precision, recall and F1 score for 'buy' predictions
    :return:
    """
    #take modal prediction
    pred=np.round(pred, decimals=0)

    #compute metrics
    selected_elements = np.extract(pred==1, pred)
    selected_labels = np.extract(pred == 1, label)
    relevent_elements = np.extract(label==1, pred)
    relevent_labels = np.extract(label==1, label)

    buy_precision=np.mean(selected_elements==selected_labels)
    buy_recall=np.mean(relevent_elements==relevent_labels)
    buy_F1=2*buy_precision*buy_recall/(buy_precision+buy_recall)
    return buy_precision, buy_recall, buy_F1

def buy_precision(label, pred):
    return classifer_metrics(label, pred)[0]
def buy_recall(label, pred):
    return classifer_metrics(label, pred)[1]
def buy_f1(label, pred):
    return classifer_metrics(label, pred)[2]


def PreRecF1():
    """
    compute precision, recall and F1 score for 'buy' predictions
    :return:
    """
    P=mx.metric.CustomMetric(feval=buy_precision, name='buy precision', output_names=['pred_output'],
                             label_names=['softmax_label'])
    R=mx.metric.CustomMetric(feval=buy_recall, name='buy recall', output_names=['pred_output'],
                               label_names=['softmax_label'])
    F=mx.metric.CustomMetric(feval=buy_f1, name='buy F1', output_names=['pred_output'],
                               label_names=['softmax_label'])
    L=mx.metric.Loss(name='loss', output_names=['pred_output'])

    metrics=[P, R, F, L]

    return mx.metric.CompositeEvalMetric(metrics)
