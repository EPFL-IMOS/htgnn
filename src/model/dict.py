import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from torch_geometric.nn import (    
    GCN, GAT, GIN,
    GCNConv,
    GATv2Conv,
    GINEConv,
    LayerNorm,
    BatchNorm,
    GraphNorm,
    global_add_pool,
    global_mean_pool,
    global_max_pool,
)

from ..utils.preprocessing import CustomMinMaxScaler


SCALER_DICT = {
    'cminmax': CustomMinMaxScaler,
    'minmax': MinMaxScaler,
    'robust': RobustScaler,
    'standard': StandardScaler
}


RNN_LAYER_DICT = {
    'gru': nn.GRU,
    'lstm': nn.LSTM,
}

ACTIVATION_DICT = {
    'leakyrelu': nn.LeakyReLU,
    'relu': nn.ReLU,
    'silu': nn.SiLU, 
    'tanh': nn.Tanh,
    'softplus': nn.Softplus,
    'sigmoid': nn.Sigmoid,
    'identity': nn.Identity,
}

NORM_LAYER_DICT = {
    'layer': LayerNorm,
    'batch': BatchNorm,
    'graph': GraphNorm,
    'identity': nn.Identity,
}

GNN_DICT = {
    'gat': GAT,
    'gcn': GCN,
    'gin': GIN,
}

GNN_CONV_LAYER_DICT = {
    'gat': GATv2Conv,
    'gcn': GCNConv,
    'gin': GINEConv,
}

GRAPH_POOL_DICT = {
    'mean': global_mean_pool,
    'max': global_max_pool,
    'add': global_add_pool,
}