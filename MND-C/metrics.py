import torch
from torch.nn import functional as F
from itertools import combinations
import numpy as np
from torchility.metrics import MetricBase
from torchmetrics.functional import accuracy, precision, recall, f1_score, auroc



class PairBase(MetricBase):
    def prepare(self, preds, targets):
        preds = preds  #取第一个元素作为实际预测值
        targets = targets #取每一行的第一个元素，组成（n,1）的二维数组
        return preds, targets

class PairAUC(PairBase):
    def __init__(self, name='AUC'):
        super().__init__(name=name)

    def forward(self, preds, targets):
        return auroc(preds, targets, task='binary')

class PairAccuracy(PairBase):
    def __init__(self, name='Acc'):
        super().__init__(name=name)

    def forward(self, preds, targets):
        return accuracy(preds, targets, task='binary')


class PairRecall(PairBase):
    def __init__(self, name='Acc'):
        super().__init__(name=name)

    def forward(self, preds, targets):
        return recall(preds, targets, task='binary')


class PairPrecision(PairBase):
    def __init__(self, name='P'):
        super().__init__(name=name)

    def forward(self, preds, targets):
        return precision(preds, targets, task='binary')


class PairF1(PairBase):
    def __init__(self, name='F1'):
        super().__init__(name=name)

    def forward(self, preds, targets):
        return f1_score(preds, targets, task='binary')


'''
class Loss:
    def __init__(self, alpha=1., beta=1., eps=1e-9, device='cpu'):
        """
        Args:
            alpha (float, optional): 度量损失的系数. Defaults to 1.
            beta (float, optional): 两对节点是否同属一组的分类损失系数. Defaults to 1.
            eps (float, optional): 避免反向传播梯度为nan的很小的数字. Defaults to 1e-9.
            device (str or pytorch device, optional): Defaults to 'cpu'.
        """
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.device = device

    def __call__(self, model_out, labels):
        cls_pred, pair_embeds, self.threshold = model_out
        cls_labels, group_labels = labels[:, 0].unsqueeze(-1), labels[:, 1].unsqueeze(-1)

        # 节点对分类损失
        loss = self.pair_loss(cls_pred, cls_labels.float())

        # 构造2元组或3元组数据的id和标签。它们会在metric_loss和group_loss中共同
        if self.alpha > 0 or self.beta > 0:
            self.batching(pair_embeds, group_labels)

        # 度量损失
        if self.alpha > 0:
            loss += self.alpha * self.metric_loss()
        # 节点对是否具有共同邻节点分类损失
        if self.beta > 0:
            loss += self.beta * self.group_loss()
        return loss

    def pair_loss(self, preds, targets):  #用于计算二分类交叉熵损失函数的值，并将其返回
        return F.binary_cross_entropy(preds, targets)

    def batching(self, pair_embeds, group_labels):  #batching方法是一个抽象方法，没有具体的实现
        raise NotImplementedError()                 #在基类中定义抽象方法是为了强制子类去实现这个方法

    def metric_loss(self):                          #这些抽象方法的目的是在基类中定义接口，
        raise NotImplementedError()                 #以便子类能够根据自己的需求来实现具体的逻辑

    def group_loss(self):
        raise NotImplementedError()
'''


