#处理数据
import scipy.sparse as sp
import numpy as np
import torch
import networkx as nx
from itertools import combinations
from utils import set_columns, set_rows
import dgl
from torch.utils.data import DataLoader
from utils import index_of


def choice_zero(mat, symmetric=True): #从矩阵中随机选择一个为0的位置，返回该位置的坐标
    """
    mat: sparse matrix
    symmetric: 是否对称矩阵
    """
    nonzero_or_chosen = set(zip(*mat.nonzero())) #将稀疏矩阵中非零元素的坐标以元组的形式组成一个集合，并赋值给变量nonzero_or_chosen
    while True:
        t = tuple(np.random.randint(0, mat.shape[0], 2))
        if symmetric and t[0] <= t[1]:  # 对称矩阵仅取下三角区域
            continue
        if t not in nonzero_or_chosen:
            yield t
            nonzero_or_chosen.add(t)


def sample_zeros(mat, size, symmetric=True): #从稀疏矩阵中随机取出指定数量的零元素的位置（即坐标），返回一个由这些位置组成的列表
    """
    mat: sparse matrix
    size: sample number
    symmetric: 是否对称矩阵
    """
    itr = choice_zero(mat, symmetric)
    return [next(itr) for _ in range(size)]


def split_dataset(pos_ds, neg_ds, train_per=0.7, test_per=0.3):
    """划分数据集
    Args:
        pos_ds ([((src, dst), (1, n))]): 正对.
        neg_ds ([((src, dst), (0, -1))]): 负对.
        train_per (float, optional): 训练比例. Defaults to 0.4.
        test_per (float, optional): 测试比例. Defaults to 0.2.
    Return:
        train_ds
        val_ds
        test_ds
    """
    """
    if neg_ds is None:
        ds = np.array(pos_ds)
    else:
        ds = np.concatenate([pos_ds, neg_ds], 0)
    
    
    if pos_ds is None:
        ds = np.array(neg_ds)
    else:
        ds = np.concatenate([pos_ds, neg_ds], 0)
    """
    ds = np.concatenate([pos_ds, neg_ds], 0)
    size = ds.shape[0]
    rand_index = np.array(range(size))
    np.random.shuffle(rand_index)
    train_size = int(size*train_per)
    test_size = int(size*test_per)

    train_ds = ds[rand_index.astype(int)[:train_size]]
    test_ds = ds[rand_index.astype(int)[train_size:train_size+test_size]]
    
    return train_ds, test_ds


def gen_datasets(edges, feat_dict=None, remove_per=0.4, obj_min_degree=4, neg_min_degree=2, limit_size=None, feat_key='__feat__'):
    """
    Args:
        srcs (ndarray): source nodes.
        dsts (ndarray): destination nodes.
        remove_per (float, optional): 每个目标节点去除多少条边. Defaults to 0.6.
        obj_min_degree (int, optional): 目标节点最小度. Defaults to 3.
        neg_min_degree (int, optional): 负对节点最小度. Defaults to 2.
        limit_size (int, optional): 正负样本对最大数量
        feat_key: key of node feature in netowrkx
    Returns:
        [((src, dst), (1, n))]: positive pairs with object nodes.
        [((src, dst), (0, -1))]: negtive pairs with object nodes.
        nx.Graph: graph with positive edges be removed.
    """
    g = nx.Graph()
    g.add_edges_from(edges)
    if feat_dict:
        nx.set_node_attributes(g, feat_dict, feat_key)

    g = nx.convert_node_labels_to_integers(g)  # 重置节点id

    adj = nx.to_scipy_sparse_matrix(g, format='csr')  # 稀疏邻接矩阵

    # -- 正对生成
    """
    remove_edges = []
    pos_ds = []
    for n, d in g.degree:
        if d < obj_min_degree:
            continue
        remove_num = max(2, round(g.degree[n] * remove_per))
        remove = np.random.choice(list(g.neighbors(n)), remove_num, replace=False)
        for nb in remove:
            remove_edges.append((n, nb))
        for node_pair in combinations(remove, 2):
            pos_ds.append((node_pair, (1, n)))
    """
    selected_nodes = [n for n, d in g.degree() if d >= obj_min_degree]

    # Calculate the percentage of nodes to select (e.g., 10%)
    percentage_to_select = 0.5
    num_nodes_to_select = int(len(g.nodes) * percentage_to_select)
    selected_nodes = np.random.choice(selected_nodes, size=min(num_nodes_to_select, len(selected_nodes)), replace=False)
    #selected_nodes = np.random.choice(selected_nodes, size=1, replace=False)
    
    remove_edges = []
    pos_ds = []
    for n in selected_nodes:
        remove_num = max(2, round(g.degree[n] * remove_per))
        remove = np.random.choice(list(g.neighbors(n)), remove_num, replace=False)
    
        for nb in remove:
            remove_edges.append((n, nb))
    
        for node_pair in combinations(remove, 2):
            pos_ds.append((node_pair, (1, n)))

    # -- 去除边
    g.remove_edges_from(remove_edges)

    # -- 负对生成
    access_2step = adj @ adj
    candi_mat = access_2step
    # 将度小于min_degree的节点排除
    exclude_nodes = np.argwhere(access_2step.diagonal() < neg_min_degree).squeeze(-1)
    candi_mat = set_rows(candi_mat, exclude_nodes, 1)
    candi_mat = set_columns(candi_mat, exclude_nodes, 1)

    neg_size = min(len(pos_ds), int(candi_mat.count_nonzero()*0.5))
    neg_node_pairs = sample_zeros(candi_mat, neg_size)
    neg_ds = [(pair, (0, -1)) for pair in neg_node_pairs]

    # 数据量上限
    if limit_size:
        idxs = np.arange(len(pos_ds))
        np.random.shuffle(idxs)
        pos_ds = [pos_ds[idx] for idx in idxs[:limit_size]]
        idxs = np.arange(len(neg_ds))
        np.random.shuffle(idxs)
        neg_ds = [neg_ds[idx] for idx in idxs[:limit_size]]
    return pos_ds, neg_ds, g


class MultiLayerNeighborSampler(dgl.dataloading.Sampler):
    def __init__(self, hop_widths): #接收hop_widths列表作为参数，表示每个节点需要采样的邻居层数或跳数
        super().__init__()
        self.fanouts = hop_widths

    def sample(self, g, seed_nodes):
        output_nodes = seed_nodes
        subgs = []
        for fanout in reversed(self.fanouts):
            # Sample a fixed number of neighbors of the current seed nodes.
            sg = dgl.sampling.sample_neighbors(g, seed_nodes, fanout, replace=False)
            # Convert this subgraph to a message flow graph.
            sg = dgl.to_block(sg, seed_nodes)
            seed_nodes = sg.srcdata[dgl.NID]
            subgs.insert(0, sg)
        input_nodes = seed_nodes
        return input_nodes, output_nodes, subgs


class NPDataLoader(DataLoader):
    def __init__(self, g, dataset, batch_size, node_feats=None, feat_key='__feat__',
                 hop_widths=(-1, -1), shuffle=True):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self._collate)

        #g = dgl.from_networkx(g, idtype=torch.long)
        self.g = g.add_self_loop()  # 添加self_loop

        # 节点特征
        if node_feats is None:
            self.g.ndata[feat_key] = self.g.nodes()       # 节点id作为其特征（作为one-hot使用）
        else:
            self.g.ndata[feat_key] = torch.tensor(node_feats, dtype=torch.float64)

        self.block_sampler = MultiLayerNeighborSampler(hop_widths)   # DGL邻域采样器

    def _collate(self, batch_ds):
        batch = np.array(batch_ds)
        pairs, labels = batch[:,0], batch[:,1]
        result_dict = {}
        negative_counter = -1
        for pair, label in zip(pairs, labels):
            n_value = label[1]
            if n_value == -1:
                negative_counter -=1
                new_negative_value = negative_counter
                if new_negative_value not in result_dict:
                    result_dict[new_negative_value] = []
                result_dict[new_negative_value].append(pair)
            else:
                if n_value not in result_dict:
                    result_dict[n_value] = []
                result_dict[n_value].append(pair)
        
        seeds_result = {}
        blocks_list = []
        for n_value, pairs_list in result_dict.items():
            seeds = np.unique(np.array(pairs_list).flatten())
            seeds_ =  torch.tensor(seeds, dtype=torch.long)
            blocks = self.block_sampler.sample(self.g, seeds_)[-1]
            block_info = (n_value, blocks)
            label = block_info[0]
            if label not in seeds_result:
                seeds_result[label] = []
            seeds_result[label].append(block_info[1])
            blocks_list.append(blocks)
        #for i, (n_value, blocks) in enumerate(seeds_result):
            #print(f"第'{n_value}'产生的'blocks'(迭代{i+1}):", blocks)
        #seeds = np.unique(pairs.flatten())  #对元素去重
        #seeds_ =  torch.tensor(seeds, dtype=torch.long)
        #blocks = self.block_sampler.sample(self.g, seeds_)[-1]        # 利用BlockSampler采样

        # 构造节点对在block中的id
        pair_id = index_of(pairs, seeds)             # 对pos_nps中的节点重新编号

        if len(batch_ds) < 4:
            raise StopIteration()
        
        return blocks_list, seeds_result
        #return blocks, torch.tensor(pair_id, dtype=torch.int64), torch.tensor(labels, dtype=torch.int32)

        
import matplotlib
import random
import matplotlib.pyplot as plt

if __name__ == '__main__':
    es = np.array([(0, 1), (0, 2), (0, 3), (0, 4),
                   (1, 4), (1, 5), (1, 6),
                   (2, 3),  (2, 7), (2, 8),
                   (3, 8), (3, 9), (3, 10),
                   (4, 5), (4, 10), (4, 11),
                   (5, 11), (5, 12), (5, 13),
                   (6, 13), (6, 14),
                   (7, 8), (7, 15), (7, 16),
                   (8, 16), (8, 17),
                   (9, 10), (9, 17), (9, 18),
                   (10, 18), (10, 19),
                   (11, 12), (11, 19), (11, 20),
                   (12, 20), (12, 21),
                   (13, 14), (13, 21), (13, 22),
                   (14, 22)
                   ])   #es是用来存储边信息的二维数组，形状是（41,2）

    h=nx.Graph()
    h.add_edges_from(es)
    

    src, dst = es.transpose(1, 0)  #将边的数组转置，得到起点数组src和终点数组dst
    #r = gen_datasets(es)
    #print(r)
    pos_ds, neg_ds, g=gen_datasets(es)
    #print(pos_ds, neg_ds)
    plt.figure(figsize=(10,5))
    nx.draw(h, with_labels=True)
    plt.show()

    h=dgl.from_networkx(h)
    h = h.add_self_loop()
    num_nodes = h.number_of_nodes()
    h = dgl.to_homogeneous(h)
    num_seed_nodes=2
    seed_nodes=np.random.choice(num_nodes, num_seed_nodes, replace=False)
    sampler = MultiLayerNeighborSampler([-1,-1])
    input_nodes, output_nodes, subgs = sampler.sample(h, seed_nodes)

    dl = NPDataLoader(g, pos_ds, batch_size=20, node_feats=None, hop_widths=[-1, -1])

    print('输入节点是:', input_nodes)
    print('输出节点是:', output_nodes)
    print('图列表:', subgs)
    nx.draw(h.to_networkx(), with_labels=True)
    plt.show()
    print('-------------------')
    for blocks, __ in dl:
        print(blocks)

    
    


