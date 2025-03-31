"""
Implementation of the Hetereogeneous Temporal Graph Neural Network (HTGNN) model.

The heteogenty comes from the fact that the model processes two different modalities of signals: low frequency signals and high frequency signals. 
The model is designed for graph level regression tasks

Author: Mengjie Zhao (mengjie.zhao@outlook.de)
Date: June 2024
"""

import torch.nn as nn
from torch_geometric.nn import HeteroConv, GCNConv, GATv2Conv

from ..model.encoder import ENCODER_DICT
from ..model.head import GraphHead
from .dict import ACTIVATION_DICT, NORM_LAYER_DICT
from ..utils.init import init_weights
from ..config import cfg


class HTGNN(nn.Module):
    def __init__(
        self,
        output_dim,
        aug_exovar=True,
        do_encoder_norm=False,
        node_embed_dim=10,
        exovar_embed_dim=10,
        gnn_embed_dim=32,
        num_gnn_layers=2,
        do_gnn_norm=True,
        gnn_norm_type='batch',
        head_embed_dim=32,
        num_head_layers=2,
        head_rnn=True,
        head_rnn_type='gru',
        head_rnn_bi=True,
        do_head_norm=True,
        head_norm_type='batch',
        gating_cnn=False,
        cnn_channel_sizes=[2, 2, 1],
        cnn_kernel_sizes=[3, 5],
        cnn_dilations=[1, 1],
        cnn_concat=True,
        dropout_prob=0.5, 
        activation='relu',
        hetero_agg='sum',
        **kwargs,
    ):
        """HGNN for the bearing dataset"""
        super().__init__()
        self.act = ACTIVATION_DICT[activation]()
        self.aug_exovar = aug_exovar
        self.lowf_encoder = None
        self.highf_encoder = None
        self.dropout = nn.Dropout(p=dropout_prob)
        
        self.lowf_encoder = ENCODER_DICT['gru'](
            in_dim=cfg.dataset.feat_dim_lowf,
            out_dim=node_embed_dim,
            n_nodes=cfg.dataset.n_nodes_lowf,
            final_activation=activation,
            norm_func=encoder_norm_type if do_encoder_norm else None, # type: ignore
        )
    
        self.highf_encoder = ENCODER_DICT['cnn'](
            in_dim=cfg.dataset.feat_dim_highf,
            n_nodes=cfg.dataset.n_nodes_highf,
            out_dim=node_embed_dim,
            activation='sigmoid',
            kernel_sizes=cnn_kernel_sizes,
            dilations=cnn_dilations,
            channel_sizes=cnn_channel_sizes,
            gated=gating_cnn,
            concat=cnn_concat,
            context_dim=node_embed_dim,
            norm_func='batch' if do_encoder_norm else None, # type: ignore
        )
        
        self.exo_encoder = ENCODER_DICT['emb'](
            in_dim=1, n_layers=2,
            hidden_dim=node_embed_dim,
            out_dim=exovar_embed_dim, 
            norm_func='batch' if do_encoder_norm else None, # type: ignore            
            act=activation
        )
            
        self.gnn_layers = nn.ModuleList()
        self.lowf_norm_layers = nn.ModuleList()
        self.highf_norm_layers = nn.ModuleList()
        
        if not do_gnn_norm:
            gnn_norm_type = 'identity'

        for _ in range(num_gnn_layers):
            conv = HeteroConv({
                ('lowf', 'to', 'lowf'): GCNConv(-1, gnn_embed_dim),
                ('highf', 'to', 'highf'): GCNConv(-1, gnn_embed_dim),
                ('highf', 'to', 'lowf'): GATv2Conv((-1, -1), gnn_embed_dim, add_self_loops=False),
                ('lowf', 'to', 'highf'): GATv2Conv((-1, -1), gnn_embed_dim, add_self_loops=False),
                }, aggr=hetero_agg)
            self.gnn_layers.append(conv)
            self.lowf_norm_layers.append(NORM_LAYER_DICT[gnn_norm_type](gnn_embed_dim))
            self.highf_norm_layers.append(NORM_LAYER_DICT[gnn_norm_type](gnn_embed_dim))
        
        self.head = GraphHead(
            dim_in=gnn_embed_dim,
            dim_out=output_dim, 
            exovar_indim=exovar_embed_dim,
            aug_exovar=not aug_exovar, # for the ablation study
            n_layers=num_head_layers,
            hidden_dim=head_embed_dim,
            do_norm=do_head_norm,
            norm_func=head_norm_type,
            final_activation='none',
            use_rnn=head_rnn,
            rnn_type=head_rnn_type,
            bidirectional=head_rnn_bi,
            activation=activation,
        )
        
        # Regression Output
        self.apply(init_weights)
        
    def forward(self, batch):
        # x: low frequency signals, shape (b, w, k): b - batch size, w - window size, k - number of features        
        b = batch['highf'].batch[-1] + 1
        x_dict = batch.x_dict
        edge_index_dict = {k: v['edge_index'] for k, v in batch.edge_index_dict.items()}
        
        exo_hid = self.exo_encoder(batch['exo_var'].x) 
        
        if self.lowf_encoder is not None:
            # low frequency node representation conditioned on rpm
            x_lowf_exo = self.lowf_encoder(batch['lowf'].x, exo_hid if self.aug_exovar else None)
            x_dict['lowf'] = x_lowf_exo
        
        if self.highf_encoder is not None:
            # high frequency feature extraction
            x_highf_exo = self.highf_encoder(batch['highf'].x, exo_hid if self.aug_exovar else None)
            x_dict['highf'] = x_highf_exo
        
        for i, conv in enumerate(self.gnn_layers):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict['lowf'] = self.dropout(self.act(self.lowf_norm_layers[i](x_dict['lowf'])))
            x_dict['highf'] = self.dropout(self.act(self.highf_norm_layers[i](x_dict['highf'])))

        #  for the ablation study
        out = self.head(x_dict['lowf'], x_dict['highf'], b, exo_hid if not self.aug_exovar else None)
        
        return out