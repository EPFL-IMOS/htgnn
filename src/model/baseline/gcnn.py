import torch
import torch.nn as nn


from ..dict import ACTIVATION_DICT, NORM_LAYER_DICT
from ...model.encoder import ENCODER_DICT, GatedConv1d
from ...model.baseline.mlp import MLP
from ...utils.init import init_weights




class GCNN(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim=1,
        hid_dim=10,
        enc_type='cnn',
        enc_kernel_size=5,
        channel_sizes=[1, 1, 1],
        kernel_sizes=[3, 5],
        dilations=[1, 1],
        context_dim=1,
        high_freq_path=True,
        gated=False,
        concat=False,
        padding='valid',
        n_hidden=50,
        n_head_layers=1,
        dropout_prob=0.5, 
        activation='relu',
        norm_func='batch',
        **kwargs
    ):
        super(GCNN, self).__init__()
        n_layers = len(channel_sizes)
        self.input_dim = input_dim
        self.concat = concat
        self.do_norm = norm_func != 'none'
        self.gated = gated
        
        if self.gated:
            self.contr_encoder = ENCODER_DICT[enc_type](
                in_dim=1, 
                out_dim=context_dim, 
                kernel_sizes=[enc_kernel_size, enc_kernel_size],
                norm_func='batch' if self.do_norm else None, # type: ignore            
                activation='sigmoid',
                gated=False,
                final_activation='silu',
            )
    
        self.high_freq_path = high_freq_path
        self.low_freq_path = not high_freq_path or gated  # Enable low-frequency path if gated is True

        self.high_freq_layers = nn.ModuleList()
        self.low_freq_layers = nn.ModuleList()
        
        assert len(kernel_sizes) == 2, 'kernel_sizes must have length 2 for high and low frequency paths'
        assert len(dilations) == 2, 'dilations must have length 2 for high and low frequency paths'
        
        k1, k2 = kernel_sizes
        d1, d2 = dilations
        
        # Create layers for high-frequency details
        if self.high_freq_path:
            for i in range(n_layers):
                out_channels = channel_sizes[i]
                args = dict(
                    in_channels=input_dim if i == 0 else channel_sizes[i - 1],
                    out_channels=out_channels,
                    kernel_size=k1,
                    stride=1,
                    padding=padding,
                    dilation=d1,
                )
                if gated:
                    args['context_dim'] = context_dim
                layer = GatedConv1d(**args) if gated else nn.Conv1d(**args)
                self.high_freq_layers.append(layer)
                    
                # Add normalization if specified
                if self.do_norm:
                    self.high_freq_layers.append(NORM_LAYER_DICT[norm_func](out_channels))
                
                if i!= 0 and dropout_prob > 0.:
                     self.high_freq_layers.append(nn.Dropout(dropout_prob))
                
                self.high_freq_layers.append(ACTIVATION_DICT[activation]())

        # Create layers for low-frequency details
        if self.low_freq_path:
            for i in range(n_layers):
                out_channels = channel_sizes[i]
                args = dict(
                    in_channels=input_dim if i == 0 else channel_sizes[i - 1],
                    out_channels=out_channels,
                    kernel_size=k2,
                    stride=1,
                    padding=padding,
                    dilation=d2,
                )
                if gated:
                    args['context_dim'] = context_dim
                layer = GatedConv1d(**args) if gated else nn.Conv1d(**args)
                self.low_freq_layers.append(layer)
                
                # Add normalization if specified
                if self.do_norm:
                    self.low_freq_layers.append(NORM_LAYER_DICT[norm_func](out_channels))
                
                if i!= 0 and dropout_prob > 0.:
                     self.low_freq_layers.append(nn.Dropout(dropout_prob))
                
                self.low_freq_layers.append(ACTIVATION_DICT[activation]())
        
        self.final_pool = nn.AdaptiveAvgPool1d(hid_dim//2 if concat else hid_dim)
        
        self.mlp = MLP(
            input_dim=hid_dim,
            output_dim=output_dim, 
            hidden_dims=[n_hidden] * n_head_layers,
            norm_func=norm_func,
            do_norm=False,
            activation=activation,
            dropout_prob=0.
        )
        
        self.apply(init_weights)

    def forward(self, batch):
        if not isinstance(batch, torch.Tensor):
            x = batch.x  # (b * n_nodes, window_size)
        
            # Change input shape from (b * n_nodes, window_size) to (b, n_nodes, window_size)
            x = x.view(-1, self.input_dim, x.shape[-1])  # (b, n_nodes, window_size)
        else:
            x = batch
        
        if self.gated:
            c = self.contr_encoder(batch.exo_var)        
        
        high_freq_output = x.clone()
        low_freq_output = x.clone()

        # Process high-frequency path
        if self.high_freq_path:
            for layer in self.high_freq_layers:
                if isinstance(layer, GatedConv1d) and c is not None:
                    high_freq_output = layer(high_freq_output, c)
                else:
                    high_freq_output = layer(high_freq_output)
            out = self.final_pool(high_freq_output)

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
                combined_output = torch.concat([out, low_freq_output], dim=2)
            else:
                combined_output = out + low_freq_output
            out = combined_output
        
        out = out.squeeze(1) # (b, hidden)
        out = self.mlp(out)
        return out


if __name__ == '__main__':
    from torchinfo import summary
    seq_length = 50
    n_nodes = 5
    batch_size = 2
    model = GCNN(
        input_dim=n_nodes, 
        window_size=seq_length,
        n_channels=10,
        n_layers=4,
        output_dim=1, 
        kernel_size=5, 
        padding='same',
        norm_func='batch',
        dropout_prob=0.3
    )
    print(model)
    summary(model, (batch_size, n_nodes, seq_length))
