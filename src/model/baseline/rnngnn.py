"""This is the implementation of Homogeneous Graph Neural Network as an ablation study of HTGNN"""


import torch
import torch.nn as nn
from ..dict import ACTIVATION_DICT, GNN_CONV_LAYER_DICT, NORM_LAYER_DICT
from ...utils.init import init_weights
from ...config import cfg
from ..encoder import ENCODER_DICT
from ..head import GraphHead


class RNNGNN(nn.Module):
    def __init__(
        self,
        input_dim,
        enc_type='gru',
        output_dim=None,
        aug_exovar=True,
        first_low=True,
        exovar_embed_dim=5,
        do_encoder_norm=True,
        encoder_norm_type='layer',
        node_embed_dim=10,
        gnn_type='gcn',
        gnn_embed_dim=32,
        num_gnn_layers=2,
        do_gnn_norm=True,
        gnn_norm_type='batch',
        head_embed_dim=32,
        num_head_layers=2,
        do_head_norm=True,
        head_norm_type='batch',
        head_rnn=True,
        head_rnn_type='gru',
        head_rnn_bi=True,
        gating_cnn=False,
        cnn_channel_sizes=[2, 2, 1],
        cnn_kernel_sizes=[3, 5],
        cnn_dilations=[1, 1],
        cnn_concat=True,
        dropout_prob=0.5, 
        activation='relu',
        **kwargs,
    ):
        """HGNN for the bridge dataset"""
        super().__init__()
        self.input_dim = input_dim
        self.act = ACTIVATION_DICT[activation]()
        self.aug_exovar = aug_exovar
        self.first_low = first_low
        self.gnn_type = gnn_type
        self.node_embed_dim = node_embed_dim
        self.gnn_to_head = torch.nn.Linear(gnn_embed_dim, head_embed_dim) if gnn_embed_dim != head_embed_dim else torch.nn.Identity()
        self.dropout = nn.Dropout(p=dropout_prob)
        
        args = dict(
            in_dim=1,
            n_nodes=cfg.dataset.n_nodes,
            out_dim=exovar_embed_dim,
            final_activation=activation,
            activation=activation,
            normfunc=encoder_norm_type if do_encoder_norm else None, # type: ignore
        )
        if enc_type == 'cnn':
            args['channel_sizes'] = cnn_channel_sizes
            args['kernel_sizes'] = cnn_kernel_sizes
            args['dilations'] = cnn_dilations
            args['gated'] = gating_cnn
            args['concat'] = cnn_concat
            args['context_dim'] = exovar_embed_dim
            
        self.dyn_encoder = ENCODER_DICT[enc_type](**args)
        
        if self.aug_exovar:
            self.exo_encoder = ENCODER_DICT['emb'](
                in_dim=1, n_layers=2,
                hidden_dim=node_embed_dim,
                out_dim=exovar_embed_dim, 
                norm_func='batch' if do_encoder_norm else None, # type: ignore            
                act=activation
        )

        self.gnn_layers = nn.ModuleList()
        self.node_norm_layers = nn.ModuleList()
        
        if not do_gnn_norm:
            gnn_norm_type = 'identity'

        gnn_in_channels = node_embed_dim
        for _ in range(num_gnn_layers):
            gnn_out_channels = gnn_embed_dim
            # gnn_out_channels = gnn_embed_dim if i != num_gnn_layers - 1 else head_embed_dim
            if self.gnn_type == 'gin':
                self.gnn_layers.append(GNN_CONV_LAYER_DICT[gnn_type](
                    nn=nn.Sequential(
                        nn.Linear(gnn_in_channels, gnn_embed_dim),
                        nn.BatchNorm1d(gnn_embed_dim),
                        self.act,
                        nn.Linear(gnn_embed_dim, gnn_out_channels),
                        self.act,
                    ),
                ))
            else:
                self.gnn_layers.append(GNN_CONV_LAYER_DICT[gnn_type](
                    in_channels=gnn_in_channels,
                    out_channels=gnn_out_channels,
                ))
                
                self.node_norm_layers.append(NORM_LAYER_DICT[gnn_norm_type](gnn_out_channels))

            gnn_in_channels = gnn_out_channels

        self.gnn_embed_dim = gnn_embed_dim
        self.head = GraphHead(
            dim_in=gnn_embed_dim,
            dim_out=output_dim, 
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
        b = batch.batch[-1] + 1
        
        exo_hid = self.exo_encoder(batch.exo_var) if self.aug_exovar else None
        # exo_hid_highf = exo_hid.repeat_interleave(cfg.dataset.n_nodes_highf, 0) if self.aug_exovar else None
        x = self.dyn_encoder(batch.x, exo_hid)
        
        for i, conv in enumerate(self.gnn_layers):
            x = conv(x, edge_index=batch.edge_index)
            x = self.dropout(self.act(self.node_norm_layers[i](x)))
        
        x_lowf = x.view(b, -1, self.gnn_embed_dim)[:, :cfg.dataset.n_nodes_lowf, :].reshape(-1, self.gnn_embed_dim)
        x_highf = x.view(b, -1, self.gnn_embed_dim)[:, cfg.dataset.n_nodes_lowf:, :].reshape(-1, self.gnn_embed_dim)

        out = self.head(x_lowf, x_highf, b)
        
        return out
