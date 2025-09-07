import torch
import random as rand
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
import networkx as nx
from main import *
from data_prepare import NPDataLoader
from model import *
from metrics import *
import pickle
import warnings
warnings.simplefilter("ignore")

"""
rand.seed(50)   #设置随机种子以确保随机过程的可重现性
np.random.seed(50)
torch.manual_seed(50)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(50)
    device = 'cuda:0'
    gpu = 1
else:
    device = 'cpu'
    gpu = 0
"""

model = SimSiam()
# 加载已训练好的模型的一部分
checkpoint = torch.load('G:/YJSXX/documents/detection of missing nodes/daimalianxi/y333/encoder_model.pth')
model.load_state_dict(checkpoint)

with open('./data1/ENZYMES.pkl', 'rb') as f:
    data = pickle.load(f)  #使用函数加载数据集
feats = data['feats']   #数据集包含这些内容：特征矩阵
adj = data['adj']   #邻接矩阵
train_ds = data['train_ds']   #训练集，包含图中节点的索引列表
test_ds = data['test_ds']
g = nx.from_dict_of_lists(adj)   #将邻接矩阵转换为图对象g
g = dgl.from_networkx(g)
g = dgl.to_homogeneous(g)
feats = np.array([feats[n] for n in g.nodes().tolist()])   #提取节点特征矩阵feats中与图中节点对应的特征
feats = torch.tensor(feats, dtype=torch.float64)
feat_dim = feats.shape[1]   #确定特征维度

batch_size = 100
hop_ws = [-1,-1,-1]  # [15, 10]   #跳数宽度列表，用于控制图中每个节点的邻居范围，[-1,1]表示不限制邻居范围
train_dl = NPDataLoader(g, train_ds, batch_size=batch_size, node_feats=feats, hop_widths=hop_ws)  #创建数据加载器
test_dl = NPDataLoader(g, test_ds, batch_size=len(test_ds), node_feats=feats, hop_widths=hop_ws)


# 优化器
embed_dim = 32
linear_layer = Fc(64, embed_dim)
classifier = Classifier(embed_dim, [16,2])
complete_model = CompleteModel(linear_layer, classifier)
learning_rate = 0.005
optimizer = optim.Adam(complete_model.parameters(), lr=learning_rate)


# 训练循环
for epoch in range(10):
    complete_model.train()
    total_loss = 0.0
    total_samples = 0

    for __, results in train_dl:
            # 定义空列表来存储转换后的键和值
        labels_list = []
        blocks_lists = []
        # 遍历字典中的键和值，并分别进行转换
        for label, blocks_list in results.items():
            label = 0 if label < 0 else 1
            # 转换并添加键到列表中
            labels_list.append(label)
            # 转换并添加值到列表中
            blocks_lists.append(blocks_list)

        for label, blocks_list in zip(labels_list, blocks_lists):
            blocks = rand.choice(blocks_list)
            blocks = [b.to(torch.device('cpu')) for b in blocks]
            gcn_feat = feats[blocks[0].srcdata[dgl.NID]]
            data_dict = model(blocks, gcn_feat)
            input1, input2 = data_dict['embed1'], data_dict['embed2']
            preds = complete_model(input1, input2)
            targets = torch.tensor([[label], [label]])
        # 计算损失
            loss = F.binary_cross_entropy(preds, targets.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_samples += 1
    average_loss = total_loss / total_samples
    if (epoch+1) %10 == 0:
        print(total_samples)
        print(f'Epoch {epoch}, Average_loss:{average_loss:.4f}') 


metrics = [
        PairAUC(name='auc'),
        PairAccuracy(name='pA'),
        PairPrecision(name='pP'),
        PairRecall(name='pR'),
        PairF1(name='pf1')
    ]


complete_model.eval()
with torch.no_grad():
    
    for __, result in test_dl:
        labels_list = []
        blocks_lists = []
        all_preds = []
        all_targets = []
        val_metrics = {
            'auc':0.0,
            'pA': 0.0,
            'pP': 0.0,
            'pR': 0.0,
            'pf1': 0.0 
            }
        for labell, blocks_listt in result.items():
            labell = 0 if labell < 0 else 1
            labels_list.append(labell)
            blocks_lists.append(blocks_listt)
            #labels_tensor = torch.tensor(labels_list)
        for label1, blocks_list1 in zip(labels_list, blocks_lists):
            blocks1 = rand.choice(blocks_list1)
            blocks1 = [b.to(torch.device('cpu')) for b in blocks1]
            gcn_feat = feats[blocks1[0].srcdata[dgl.NID]]
            data_dictt = model(blocks1, gcn_feat)
            input3, input4 = data_dictt['embed1'], data_dictt['embed2']
            pred = complete_model(input3, input4)
            labels = torch.tensor([[label1],[label1]])
            all_preds.append(pred)
            all_targets.append(labels)

        for metric in metrics:
            metric_result = metric(torch.cat(all_preds), torch.cat(all_targets))
            val_metrics[metric.name] += metric_result
    #打印评估指标
    for metric_name, metric_value in val_metrics.items():
        print(f'Val{metric_name} {metric_value:.4f}')


