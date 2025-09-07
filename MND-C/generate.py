import random as rand
import torch
from data_load import load_graph
from data_prepare import gen_datasets, split_dataset
from data_prepare import NPDataLoader
import numpy as np
import pickle
import networkx as nx
import os

rand.seed(50)
np.random.seed(50)
torch.manual_seed(50)


if not os.path.exists('./data'):
    os.mkdir('./data')


# cora数据
data_name = 'ENZYMES213'

edges, feat_dict = load_graph('G:/YJSXX/documents/detection of missing nodes/daimalianxi/01/ENZYMES', head=False)
pos_ds, neg_ds, g = gen_datasets(edges, feat_dict, remove_per=0.4, limit_size=600)
train_ds, test_ds = split_dataset(pos_ds, train_per=0.7, test_per=0.3)

data = {
    'feats': nx.get_node_attributes(g, '__feat__'),
    'adj': nx.to_dict_of_lists(g),
    'train_ds': train_ds,
    'test_ds': test_ds
}

with open(f'./data/{data_name}.pkl', 'wb') as f:
    pickle.dump(data, f)

# ENZYMES  PROTEINS   IMDB-BINARY  IMDB-MULTI
# G:/YJSXX/documents/detection of missing nodes/daimalianxi/datasets/IMDB-MULTI
# G:/YJSXX/documents/detection of missing nodes/daimalianxi/IMDB-MULTI/output
# G:/YJSXX/documents/detection of missing nodes/daimalianxi/datasets/IMDB-BINARY