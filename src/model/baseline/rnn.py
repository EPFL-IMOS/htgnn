"""
Implementation of (LSTM-AD) Long Short Term Memory Networks for Anomaly Detection in Time Series
"""

import torch
import torch.nn as nn

from ...model.baseline.mlp import MLP
from ...model.dict import (
    ACTIVATION_DICT, 
    NORM_LAYER_DICT, 
    RNN_LAYER_DICT
)
from ...utils.init import init_weights


class RNN(nn.Module):
    '''
    A RNN model
    '''
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        rnn_type='gru',
        activation='relu',
        n_layers=1,
        n_head_layers=1,
        head_hidden_dim=None,
        bidirectional=False,
        norm_func=None,
        dropout_prob=0.,
    ):
        super(RNN, self).__init__()
        if head_hidden_dim is None:
            head_hidden_dim = hidden_dim
        
        self.rnn_type = rnn_type
        self.input_dim = input_dim
        self.rnn = RNN_LAYER_DICT[rnn_type](
            input_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.norm = NORM_LAYER_DICT[norm_func](hidden_dim*2 if bidirectional else hidden_dim) if norm_func is not None else None
        self.act = ACTIVATION_DICT[activation]()
        self.dropout = nn.Dropout(p=dropout_prob)
        
        self.mlp = MLP(
            input_dim=hidden_dim*2 if bidirectional else hidden_dim,
            output_dim=output_dim, 
            hidden_dims=[head_hidden_dim] * n_head_layers,
            norm_func=norm_func,
            do_norm=False,
            activation=activation,
            dropout_prob=0.
            )
        
        self.apply(init_weights)
        
    def forward(self, batch, **kwargs):
        if not isinstance(batch, torch.Tensor):
            x = batch.x  # (b * n_nodes, window_size)
            b = batch.batch[-1] + 1
        
            # Change input shape from (b * n_nodes, window_size) to (b, n_nodes, window_size)
            x_input = x.view(b, -1, x.shape[1])  # (b, n_nodes, window_size)
            
            if hasattr(batch, 'exo_var'):
                exo_var = batch.exo_var
                # concat exo_var to x
                exo_var = exo_var.unsqueeze(1)
                x_input = torch.cat([x_input, exo_var], dim=1)
            
        else:
            x_input = batch
        
        x_input = x_input.permute(0, 2, 1) # (b, window_size, n_nodes)
        
        # RNN layer
        out, _ = self.rnn(x_input) # (b, window_size, n_nodes)
        
        # Extracting from last layer # (b, window_size, D * hidden_dim)
        out = out[:, -1, :]
        
        # Apply activation function
        out = self.act(out)
        
        # Apply normalization if specified
        if self.norm is not None:
            out = self.norm(out)
        
        # Apply dropout if specified
        out = self.dropout(out)
        
        # Fully connected layer
        out = self.mlp(out)
        
        return out



if __name__ == '__main__':
    from torchinfo import summary
    seq_length = 50
    n_nodes = 10
    hidden_size = 20
    batch_size = 2
    rnn = RNN(input_dim=n_nodes, hidden_dim=hidden_size, output_dim=1, dropout_prob=0.2, rnn_type='lstm', norm_func='layer')
    summary(rnn, (batch_size, n_nodes, seq_length))

