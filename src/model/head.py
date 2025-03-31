import torch
import torch.nn as nn

from ..model.baseline.mlp import MLP
from ..config import cfg
from ..model.dict import ACTIVATION_DICT
from ..utils.init import init_weights


class GraphHead(nn.Module):
    """
    Graph Pooling.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """
    def __init__(
        self, 
        dim_in, 
        aug_exovar=False,
        exovar_indim=None,
        dim_out=1,
        hidden_dim=50,
        n_layers=2,
        do_norm=False,
        norm_func='batch', 
        activation='relu',
        final_activation='silu',
        bidirectional=True,
        use_rnn=False,
        rnn_type='gru',
        dropout_prob=0.2,
    ):
        super(GraphHead, self).__init__()
        
        self.use_rnn = use_rnn
        self.hidden_dim = hidden_dim
        
        self.lin_temp = nn.Linear(dim_in, hidden_dim)
        self.lin_vib = nn.Linear(dim_in, hidden_dim)
        
        self.aug_exovar = aug_exovar
        
        n_nodes = cfg.dataset.n_nodes
        
        if aug_exovar:
            self.lin_exovar = nn.Linear(exovar_indim, hidden_dim)
            n_nodes += cfg.dataset.n_exovars
        
        in_dim = hidden_dim*n_nodes
        if use_rnn:
            self.rnn = nn.GRU(n_nodes, hidden_dim, 1, 
                               batch_first=True,
                               dropout=dropout_prob, 
                               bidirectional=bidirectional) if rnn_type == 'gru' else nn.LSTM(n_nodes, hidden_dim, 1, 
                                                                                              batch_first=True, 
                                                                                              dropout=dropout_prob,
                                                                                              bidirectional=bidirectional)
            in_dim = hidden_dim
            if bidirectional:
                in_dim *= 2

        self.mlp = MLP(
            input_dim=in_dim,
            output_dim=dim_out, 
            hidden_dims=[hidden_dim] * n_layers,
            norm_func=norm_func,
            do_norm=do_norm,
            activation=activation,
            dropout_prob=0.
            )

        
        self.final_act = ACTIVATION_DICT[final_activation]() if final_activation != 'none' else nn.Identity()
        self.apply(init_weights)
        
    def forward(
        self, 
        x_lowf, 
        x_highf, 
        batch_size,
        c=None,
    ):
        x_lowf = self.lin_temp(x_lowf)
        x_highf = self.lin_vib(x_highf)
        
        b = batch_size
        x_lowf = x_lowf.view(b, -1, self.hidden_dim)
        x_highf = x_highf.view(b, -1, self.hidden_dim)
        x = torch.cat([x_lowf, x_highf], dim=1)
        
        if self.aug_exovar and c is not None:
            c = self.lin_exovar(c).unsqueeze(1)
            x = torch.cat([x, c], dim=1)
        
        if self.use_rnn:
            x = x.permute(0, 2, 1)
            out, _ = self.rnn(x)
            x = out[:, -1, :]
            # x = out.squeeze(0)
        else:
            x = x.flatten(1)
        outs = self.mlp(x)
        outs = self.final_act(outs)
        if not self.use_rnn:
            outs = outs.squeeze(1)
        return outs
