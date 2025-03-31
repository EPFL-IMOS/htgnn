import torch
import torch.nn as nn


from ..model.baseline.mlp import MLP
from ..utils.init import init_weights
from .dict import ACTIVATION_DICT, NORM_LAYER_DICT


class GatedConv1d(nn.Module):
    """A gated convolutional layer using a separate gating coefficient generated from contextual input."""
    def __init__(self, in_channels, out_channels, context_dim, kernel_size, stride, padding, dilation):
        super(GatedConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.gate = nn.Linear(context_dim, out_channels)  # Assuming `c` is a scalar for simplicity

    def forward(self, x, c):
        gate_coefficients = torch.sigmoid(self.gate(c))  # Shape: (batch_size, out_channels)
        gate_coefficients = gate_coefficients.unsqueeze(2)  # Adjust shape for broadcasting
        return self.conv(x) * gate_coefficients  # Apply gating by element-wise multiplication


class EmbeddingEncoder(nn.Module):
    """Embedding Encoder"""
    
    def __init__(
        self,
        in_dim,
        out_dim=10,
        hidden_dim=10,
        n_layers=2,
        act='silu',
        **kwargs
    ):
        super(EmbeddingEncoder, self).__init__()
        self.mlp = MLP(
            input_dim=in_dim,
            output_dim=out_dim, 
            hidden_dims=[hidden_dim] * n_layers,
            activation=act,
            dropout_prob=0.,
            activation_after_last_layer=True
        )
        
        self.apply(init_weights)
    
    def forward(self, x):
        x = x.mean(dim=1)
        if x.dim() == 1:
            x = x.unsqueeze(1)
        out = self.mlp(x)
        out = out.view(x.shape[0], -1)
        return out


class ConstantEncoder(nn.Module):
    """Embedding Encoder"""
    
    def __init__(
        self,
        out_dim=10,
        **kwargs
    ):
        super(ConstantEncoder, self).__init__()
        self.dim_out = out_dim

    
    def forward(self, x):
        x = x.squeeze(-1)
        out = x[:, :self.dim_out]
        return out


class AverageEncoder(nn.Module):
    """Embedding Encoder"""
    
    def __init__(
        self,
        out_dim=10,
        **kwargs
    ):
        super(AverageEncoder, self).__init__()
        self.dim_out = out_dim

    
    def forward(self, x):
        x = x.squeeze(-1)
        out = x.mean(dim=1)
        out = out.repeat(self.dim_out, 1)
        out = out.permute(1, 0)
        return out


class CNNEncoder(nn.Module):
    """Extended Convolutional Neural Network (CNN) Layer to support multiple layers with dependencies in channel and kernel sizes, including gating mechanism for dual-path processing."""

    def __init__(
        self,
        in_dim,
        n_nodes,
        out_dim,
        channel_sizes=[1, 1, 1],
        kernel_sizes=[3, 5],
        dilations=[1, 1],
        context_dim=1,
        activation='relu',
        norm_func='batch',
        high_freq_path=True,
        gated=False,
        concat=False,
        **kwargs
    ):
        super(CNNEncoder, self).__init__()
        n_layers = len(channel_sizes)
        self.concat = concat
        self.n_nodes = n_nodes
        self.high_freq_path = high_freq_path
        self.low_freq_path = not high_freq_path or gated  # Enable low-frequency path if gated is True

        self.high_freq_layers = nn.ModuleList()
        self.low_freq_layers = nn.ModuleList()
        
        k1, k2 = kernel_sizes
        d1, d2 = dilations
        
        # Create layers for high-frequency details
        if self.high_freq_path:
            for i in range(n_layers):
                out_channels = channel_sizes[i]
                args = dict(
                    in_channels=in_dim if i == 0 else channel_sizes[i - 1],
                    out_channels=out_channels,
                    kernel_size=k1,
                    stride=1,
                    padding='valid',
                    dilation=d1,
                )
                if gated:
                    args['context_dim'] = context_dim
                layer = GatedConv1d(**args) if gated else nn.Conv1d(**args)
                self.high_freq_layers.append(layer)
                    
                # Add normalization if specified
                if norm_func is not None:
                    self.high_freq_layers.append(NORM_LAYER_DICT[norm_func](out_channels))
                
                # Add activation after each conv layer except the last one
                if i < n_layers-1:
                    self.high_freq_layers.append(ACTIVATION_DICT[activation]())

        # Create layers for low-frequency details
        if self.low_freq_path:
            for i in range(n_layers):
                out_channels = channel_sizes[i]
                args = dict(
                    in_channels=in_dim if i == 0 else channel_sizes[i - 1],
                    out_channels=out_channels,
                    kernel_size=k2,
                    stride=1,
                    padding='valid',
                    dilation=d2,
                )
                if gated:
                    args['context_dim'] = context_dim
                layer = GatedConv1d(**args) if gated else nn.Conv1d(**args)
                self.low_freq_layers.append(layer)
                
                # Add normalization if specified
                if norm_func is not None:
                    self.low_freq_layers.append(NORM_LAYER_DICT[norm_func](out_channels))
                
                # Add activation after each conv layer except the last one
                if i < n_layers-1:
                    self.low_freq_layers.append(ACTIVATION_DICT[activation]())
        if concat:
            out_dim = out_dim//2
        self.final_pool = nn.AdaptiveAvgPool1d(out_dim)

    def forward(self, x, c=None):
        if x.dim() == 2:
            x = x.unsqueeze(2)
        
        x = x.permute(0, 2, 1) # (b, w, k) -> (b, k, w)
        
        if c is not None:
            c = c.repeat_interleave(self.n_nodes, 0)
        
        high_freq_output = x.clone()
        low_freq_output = x.clone()

        # Process high-frequency path
        if self.high_freq_path:
            for layer in self.high_freq_layers:
                if isinstance(layer, GatedConv1d) and c is not None:
                    high_freq_output = layer(high_freq_output, c)
                else:
                    high_freq_output = layer(high_freq_output)
            output = self.final_pool(high_freq_output)

        # Process low-frequency path
        if self.low_freq_path:
            for layer in self.low_freq_layers:
                if isinstance(layer, GatedConv1d) and c is not None:
                    low_freq_output = layer(low_freq_output, c)
                else:
                    low_freq_output = layer(low_freq_output)
            low_freq_output = self.final_pool(low_freq_output)

            if self.concat:
                # Combine the outputs from both paths
                combined_output = torch.concat([output, low_freq_output], dim=2)
            else:
                combined_output = output + low_freq_output
            output = combined_output
        
        output = output.squeeze(1)  # Remove original last dimension
        return output
    

class GRUEncoder(nn.Module):
    """Gated Recurrent Unit (GRU) Layer
    :param in_dim: number of input features
    :param hid_dim: hidden size of the GRU
    :param n_layers: number of layers in GRU
    :param dropout: dropout rate
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        n_nodes,
        final_activation='silu',
        norm_func='batch',
        bidirectional=False,
        **kwargs
    ):
        super(GRUEncoder, self).__init__()
        self.final_act = ACTIVATION_DICT[final_activation]() if final_activation != 'none' else nn.Identity()
        self.n_nodes = n_nodes
        self.do_norm = norm_func is not None
        self.in_channels = in_dim
        self.gru = nn.GRU(
            in_dim,
            out_dim,
            bidirectional=bidirectional,
            batch_first=True
        )
        if self.do_norm:
            self.feat_norm = NORM_LAYER_DICT[norm_func](out_dim)
        self.apply(init_weights)

    def forward(self, x, c=None, return_all=False):
        # condition sequential model on exogenous variable
        if x.dim() == 2:
            x = x.unsqueeze(2)  # (b, w, 1), last dim is for channel
        h0 = None
        if c is not None:
            n_nodes = self.n_nodes
            h0 = c.repeat_interleave(n_nodes, 0).unsqueeze(0)  # (1, b*n_nodes, hidden) 
            out, h = self.gru(x, h0)
        else:
            out, h = self.gru(x)
        out, h = out[:, -1, :], h[-1, :, :]  # Extracting from last layer # (b, w, k)
    
        if self.do_norm:
            h = self.feat_norm(h)
        
        h = self.final_act(h)
        
        if return_all:
            return out, h
        return h


ENCODER_DICT = {
    'gru': GRUEncoder,
    'cnn': CNNEncoder,
    'emb': EmbeddingEncoder,
    'avg': AverageEncoder,
    'const': ConstantEncoder,
}
