"""
input_file_path = "G:\YJSXX\documents\detection of missing nodes\daimalianxi\datasets\PROTEINS\PROTEINS\PROTEINS_node_attributes.txt"
output_file_path = "G:\YJSXX\documents\detection of missing nodes\daimalianxi\datasets\PROTEINS\PROTEINS\eatures.txt"

with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
    for index, line in enumerate(input_file, start=1):
        # 添加节点ID
        modified_line = f"{index}: {line.strip()}"
        # 写入到输出文件
        output_file.write(modified_line + '\n')
"""

"""
input_file_path = "G:\YJSXX\documents\detection of missing nodes\daimalianxi\datasets\IMDB-MULTI\IMDB-MULTI\IMDB-MULTI_A.txt"
output_file_path = "G:\YJSXX\documents\detection of missing nodes\daimalianxi\datasets\IMDB-MULTI\IMDB-MULTI\edges.txt"

# 使用集合来追踪已经出现过的边
unique_edges = set()

with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
    for line in input_file:
        # 去掉行两端的空格和换行符
        edge = line.strip()
        # 将边按逗号分割成两个节点
        nodes = edge.split(',')
        # 将边的两个节点排序后，作为已经处理过的边的标志
        sorted_edge = ','.join(sorted(nodes))
        # 如果这条边的反向边还没有出现过，就记录这条边
        if sorted_edge not in unique_edges:
            unique_edges.add(sorted_edge)
            output_file.write(edge + '\n')
"""
"""
import numpy as np

edges_file_path = "G:\YJSXX\documents\detection of missing nodes\daimalianxi\datasets\IMDB-BINARY\IMDB-BINARY\edges.txt"
features_file_path = "G:\YJSXX\documents\detection of missing nodes\daimalianxi\datasets\IMDB-BINARY\IMDB-BINARY\eatures.txt"

# 读取 edges 文件，获取节点列表
nodes_set = set()
with open(edges_file_path, 'r') as edges_file:
    for line in edges_file:
        node1, node2 = map(int, line.strip().split(','))
        nodes_set.add(node1)
        nodes_set.add(node2)

# 生成随机初始化特征
num_features = 18
random_features = {node: np.random.rand(num_features).tolist() for node in nodes_set}

# 保存特征到 features 文件
with open(features_file_path, 'w') as features_file:
    for node, feature_list in random_features.items():
        feature_str = ','.join(map(str, feature_list))
        features_file.write(f"{node}: {feature_str}\n")
"""