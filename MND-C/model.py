import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 
import dgl
from dgl.nn.pytorch import GraphConv


#GCN的层数即hidden_dims的个数取决于hop_width也就是采样的阶数
class GCN(nn.Module):
    def __init__(self, feature_dim, hidden_dims, node_number_for_embedding=None, feat_key='__feat__'):
        super().__init__()
        self.feat_key = feat_key

        if node_number_for_embedding:
            self.node_embed = nn.Embedding(node_number_for_embedding, feature_dim)
        else:
            self.node_embed = None

        self.conv_layers = nn.ModuleList()
        self.gcn_dims = hidden_dims
        in_dim = feature_dim
        for out_dim in hidden_dims:
            self.conv_layers.append(GraphConv(in_dim, out_dim))
            in_dim = out_dim
    def forward(self, blocks, feats):
        gcn_feat = feats.to(torch.float32)
        for (block, layer) in zip(blocks, self.conv_layers):
            gcn_feat = F.relu(layer(block, gcn_feat))
        return gcn_feat
    
        
        """
        self.conv1 = GraphConv(feature_dim, hidden_dims)
        self.conv2 = GraphConv(hidden_dims, out_dims)
        
    def forward(self, blocks, feats):
        #gcn_feat=feats[:blocks[0].number_of_dst_nodes()]
        #gcn_feat = feats[block.srcdata[dgl.NID]]
        feats = feats.to(torch.float32)
        #if self.node_embed:
            #gcn_feat = self.node_embed(gcn_feat)
    
        gcn_feat = F.relu(self.conv1(blocks[0], feats))
        gcn_feat = F.relu(self.conv2(blocks[1], gcn_feat))
        return gcn_feat
        """


def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=64):
        super().__init__()
        
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            #nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            #nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            #nn.BatchNorm1d(hidden_dim)
        )
        self.num_layers = 3
    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x 

class prediction_MLP(nn.Module):
    def __init__(self, in_dim=64, hidden_dim=32, out_dim=64): # bottleneck structure
        super().__init__()
        
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 



class SimSiam(nn.Module):
    def __init__(self, backbone=GCN(feature_dim=18, hidden_dims=[32,16])):
        super().__init__()
        
        self.backbone = backbone
        self.projector = projection_MLP(in_dim=16)

        #self.encoder = nn.Sequential(
            #self.backbone,
            #self.projector
        #)
        self.predictor = prediction_MLP()
    
    def forward(self, blocks, feats):
        
        #f, h = self.encoder, self.predictor
        x = self.backbone(blocks, feats)
        y = self.projector(x)
        indices = np.random.choice(y.shape[0], size=2, replace=False) 
        #z1, z2 = y[0:1, :], y[1:2, :]
        z1 = y[indices[0]:indices[0]+1, :]
        z2 = y[indices[1]:indices[1]+1, :]
        z1 = torch.cat([z1, z1], dim = 0)
        z2 = torch.cat([z2, z2], dim = 0)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        L = D(p1, z2) / 2 + D(p2, z1) / 2
        return {'loss': L, 'embed1': z1, 'embed2': z2}
        

class Fc(nn.Module):
    def __init__(self, in_dim, embed_dim):
        super().__init__()
        in_dim = in_dim*2
        self.mlps = nn.Linear(in_dim, embed_dim)
    
    def forward(self, input1, input2):
        input = torch.cat([input1, input2], 1)
        return self.mlps(input)

class Classifier(nn.Module):
    def __init__(self, embed_dim, mlp_dims):  #将fc的输出作为Classifier的输入
        super().__init__()
        self.mlps = nn.Sequential()
        in_dim = embed_dim
        for i, out_dim in enumerate(mlp_dims):
            self.mlps.add_module(f'c_mlp{i}', nn.Linear(in_dim, out_dim))
            #self.mlps.add_module(f'c_relu{i}', nn.ReLU())
            in_dim = out_dim
        self.mlps.add_module('c_out', nn.Linear(in_dim, 1))
        self.mlps.add_module('sigmoid', nn.Sigmoid())  #输出值映射在0到1之间

    def forward(self, inputs):
        return self.mlps(inputs)

# 完整模型
class CompleteModel(nn.Module):
    def __init__(self, fc, classifier):
        super().__init__()
        self.linear_layer = fc
        self.classifier = classifier

    def forward(self, input1, input2):
        linear_outputs = self.linear_layer(input1, input2)
        linear_outputs = F.normalize(linear_outputs, p=1, dim=1)
        preds = self.classifier(linear_outputs)
        return preds


