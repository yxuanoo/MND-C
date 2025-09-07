#数据加载
import networkx as nx
import warnings
from pathlib import Path


def load_edges(path='edges.txt', delimiter=',', head=False):
    """
    从边文件中加载网络的边。文件格式如下（#表示注释符号）：
        # 表示注释
        a, b  # 边(a,b)
        a, c  # 边(a,c)
        ...
    Args:
        path: 存储边的文件名，默认为'edges'
        delimiter: 分割符，默认为','
        head: 是否包含标题行
    Return: networkx.Graph
    """
    print('Loading edge data ...')
    edges = []
    with open(path) as f:
        for line in f:
            line = line.split('#', 1)[0].strip()
            if not line.strip():
                continue
            endpoints = line.split(delimiter)  #将字符串line按照指定的分隔符进行拆分，并将结果储存在列表中
            assert len(endpoints) == 2, 'Invalid Format! Each edge must contains two nodes!'
            edges.append(tuple(endpoints))
    if head:
        edges = edges[1:]
    return edges


def load_features(path='features.txt', delimiters=(':', ','), head=False):
    """
    文件格式：
        # 表示注释
        node1: f11, f12, f13, ...   # 节点node1的特征
        node2: f21, f22, f23, ...
        ...
    Args:
        path: 文件名
        delimiters: 节点-特征分割符，特征分割符
    Return: 特征字典，key为节点id，value为特征
    """
    print('Loading feature data ...')
    features = dict()
    skip_head = 0
    with open(path) as f:
        for line in f:
            line = line.split('#', 1)[0].strip()
            if not line.strip():
                continue

            if head and skip_head == 0:  # skip head line
                skip_head += 1
                continue

            node_feat = line.split(delimiters[0], 1)
            assert len(node_feat) == 2, 'Invalid Format! Each line must contains two patrs of node and feature.'
            features[node_feat[0]] = [float(f) for f in node_feat[1].split(delimiters[1])]

    feature_lens = {len(feat) for _, feat in features.items()}
    if len(feature_lens) > 1:
        warnings.warn('Features have different lengthes!')

    return features


def load_graph(path, g_delimiter=',', feat_delimiters=(':', ','), head=False):
    """
    从文件中加载图数据。
    Args:
        path: 两种取值，a)文件夹地址，此时默认的边文件名为edges，特征文件名为features；b)取值为元组(边文件地址, 特征文件地址)
        g_delimiter: 边文件分割符
        feat_delimiters: 特征文件分割符
        head: 首行是否标题
    Return:
        edges: [(srcs, dsts)]，graph edges
        feats_dict: 节点特征字典
    """
    if type(path) in [list, tuple]:
        assert len(path) == 2, '需提供(边索引文件, 特征文件)'
        path_edges, path_feats = path
    else:
        path = Path(path)
        path_edges, path_feats = path/'edges.txt', path/'features.txt'

    edges = load_edges(path_edges, g_delimiter, head)

    feat_dict = load_features(path_feats, feat_delimiters, head) if path_feats else None
    return edges, feat_dict

