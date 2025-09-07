import random as rand
import torch
import numpy as np
from tqdm import tqdm
from torchility import Trainer
import dgl
from optimizers import get_optimizer, LR_Scheduler
from data_prepare import NPDataLoader
from model import *
from metrics import *
from arguments import get_args
import networkx as nx
import pickle
import warnings
import matplotlib.pyplot as plt
warnings.simplefilter("ignore")

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



with open('./data/IMDB-BINARY.pkl', 'rb') as f:
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
    
def main(args):
    
    batch_size = 10
    hop_ws = [-1,-1]  # [15, 10]   #跳数宽度列表，用于控制图中每个节点的邻居范围，[-1,1]表示不限制邻居范围
    train_dl = NPDataLoader(g, train_ds, batch_size=batch_size, node_feats=feats, hop_widths=hop_ws)  #创建数据加载器
    #test_dl = NPDataLoader(g, test_ds, batch_size=len(test_ds), node_feats=feats, hop_widths=hop_ws)

    model=SimSiam()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)   #使用Adam优化器，并将模型参数传递给优化器


    print('start training:')
    #accuracy=0
    #start training
    global_progress = tqdm(range(0, args.train.stop_at_epoch, 30), desc=f'Training')
    avg_losses1 = []
    for epoch in global_progress:
        model.train()
        optimizer.zero_grad()
        total_loss1 = 0
        local_progress=tqdm(train_dl, desc=f'Epoch{epoch}/{args.train.num_epochs}', disable=args.hide_progress)
        for blocks_list, __ in local_progress:
            blocks = rand.choice(blocks_list)
            #print(blocks)
            model.zero_grad()
            blocks =[b.to(torch.device('cpu')) for b in blocks]
            gcn_feat=feats[blocks[0].srcdata[dgl.NID]]
            data_dict=model(blocks, gcn_feat)
            loss=data_dict['loss']
            total_loss1 += loss.item()
            loss.backward()
            optimizer.step()
            #lr_scheduler.step()
            #data_dict.update({'lr':lr_scheduler.get_lr()})
            local_progress.set_postfix(data_dict)
            #if gpu==0:
                #print('data_dict:',data_dict)
        avg_loss1 = total_loss1 / len(local_progress)
        avg_losses1.append(avg_loss1)
        if epoch %30 == 0:
            print(f'Epoch[{epoch}/{args.train.num_epochs}], Loss1:{avg_loss1:.4f}')
    
    plt.plot(avg_losses1, label='Loss1')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss Curve')
    plt.show()
    # 保存模型的一部分
    # 保存编码器模型
    torch.save(model.state_dict(), 'G:/YJSXX/documents/detection of missing nodes/daimalianxi/y333/encoder_model.pth')



if __name__ == "__main__":
    args=get_args()
    main(args)


#  python main.py -c configs/simsiam_cora.yaml  --hide_progress