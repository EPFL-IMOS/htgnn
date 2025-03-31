
from ..config import cfg
from .htgnn import HTGNN
from .baseline.rnn import RNN
from .baseline.cnn import CNN
from .baseline.gcnn import GCNN
from .baseline.rnngnn import RNNGNN
from .baseline.mtgat import MTGAT
from .baseline.timemixer import TimeMixer


BASELINE_MODELS = {
    'cnn': CNN,
    'gcnn': GCNN,
    'rnn': RNN,
    'rnngnn': RNNGNN,
    'mtgat': MTGAT,
    'timemixer': TimeMixer,
    'htgnn': HTGNN,
}


def create_model(model_type=None):
    common_args = dict(
        dropout_prob=cfg.model.dropout,
        activation=cfg.model.activation,
        norm_func=cfg.model.norm_func if cfg.model.norm_func != 'none' else None,
        output_dim=cfg.model.output_dim if cfg.model.output_dim > 0 else None,
    ) # type: ignore
    
    if model_type in ['htgnn', 'rnngnn']:        
        model = BASELINE_MODELS[model_type](
            **common_args,
            input_dim=cfg.dataset.n_nodes,
            aug_exovar=cfg.model.htgnn.aug_exovar,
            gating_cnn=cfg.model.htgnn.cnn.gating,
            cnn_concat=cfg.model.htgnn.cnn.emb_concat,
            cnn_channel_sizes=cfg.model.htgnn.cnn.channel_sizes,
            cnn_kernel_sizes=cfg.model.htgnn.cnn.kernel_sizes,
            cnn_dilations=cfg.model.htgnn.cnn.dilations,
            node_embed_dim=cfg.model.htgnn.node_embed_dim,
            exovar_embed_dim=cfg.model.htgnn.exovar_embed_dim,
            do_encoder_norm=cfg.model.htgnn.do_encoder_norm,
            gnn_embed_dim=cfg.model.htgnn.gnn_embed_dim,
            num_gnn_layers=cfg.model.htgnn.num_gnn_layers,
            do_gnn_norm=cfg.model.htgnn.do_gnn_norm,
            gnn_norm_type=cfg.model.htgnn.gnn_norm_type,
            num_head_layers=cfg.model.htgnn.head.num_layers,
            head_embed_dim=cfg.model.htgnn.head.embed_dim,
            head_rnn=cfg.model.htgnn.head.use_rnn,
            head_rnn_type=cfg.model.htgnn.head.rnn_type,
            head_rnn_bi=cfg.model.htgnn.head.rnn_bidirectional,
            do_head_norm=cfg.model.htgnn.head.do_norm,
            head_norm_type=cfg.model.htgnn.head.norm_type,
            hetero_agg=cfg.model.htgnn.hetero_agg,
            # for ablation study
            gnn_type=cfg.model.htgnn.gnn_type,
            enc_type=cfg.model.htgnn.enc_type,
        ) 
    elif model_type == 'mtgat':
        model = BASELINE_MODELS[model_type](
            **common_args,
            input_dim=cfg.dataset.n_nodes + cfg.dataset.n_exovars,
            window_size=cfg.dataset.window_size,
            encode=cfg.model.mtgat.encoder.encode,
            kernel_size=cfg.model.mtgat.encoder.kernel_size,
            use_gatv2=cfg.model.mtgat.use_gatv2,
            feat_gat_embed_dim=cfg.model.mtgat.feat_gat_embed_dim,
            time_gat_embed_dim=cfg.model.mtgat.time_gat_embed_dim,
            gru_n_layers=cfg.model.mtgat.gru_n_layers,
            gru_hid_dim=cfg.model.mtgat.gru_hid_dim,
            head_n_layers=cfg.model.mtgat.head_n_layers,
            head_hid_dim=cfg.model.mtgat.head_hid_dim,
        )   
    elif model_type == 'gcnn':
        model = BASELINE_MODELS[model_type](
            **common_args,
            input_dim=cfg.dataset.n_nodes,
            gated=cfg.model.gcnn.gated,
            concat=cfg.model.gcnn.concat,
            enc_type=cfg.model.gcnn.enc_type,
            enc_kernel_size=cfg.model.gcnn.enc_kernel_size,
            context_dim=cfg.model.gcnn.context_dim,
            stride=cfg.model.gcnn.stride,
            channel_sizes=cfg.model.gcnn.channel_sizes,
            dilations=cfg.model.gcnn.dilations,
            kernel_sizes=cfg.model.gcnn.kernel_sizes,
            hid_dim=cfg.model.gcnn.hidden_dim,
            n_head_layers=cfg.model.gcnn.n_head_layers
        )
    elif model_type == 'cnn':
        model = BASELINE_MODELS[model_type](
            **common_args,
            input_dim=cfg.dataset.n_nodes + cfg.dataset.n_exovars,
            window_size=cfg.dataset.window_size,
            n_layers=cfg.model.cnn.n_layers,
            n_channels=cfg.model.cnn.n_channels,
            kernel_size=cfg.model.cnn.kernel_size,
            stride=cfg.model.cnn.stride,
            n_hidden=cfg.model.cnn.hidden_dim,
            n_head_layers=cfg.model.cnn.n_head_layers
        )
    elif model_type == 'rnn':
        model = BASELINE_MODELS[model_type](
            **common_args,
            input_dim=cfg.dataset.n_nodes + cfg.dataset.n_exovars,
            hidden_dim=cfg.model.rnn.hidden_dim,
            n_layers=cfg.model.rnn.n_layers,
            rnn_type=cfg.model.rnn.rnn_type,
            bidirectional=cfg.model.rnn.bidirectional,
            n_head_layers=cfg.model.rnn.n_head_layers,
            head_hidden_dim=cfg.model.rnn.head_hidden_dim,
        )
    elif model_type == 'timemixer':
        model = BASELINE_MODELS[model_type](
            **common_args,
            enc_in=cfg.dataset.n_nodes + cfg.dataset.n_exovars,
            seq_len=cfg.dataset.window_size,
            decomp_method=cfg.model.timemixer.decomp_method,
            moving_avg_len=cfg.model.timemixer.moving_avg_len,
            e_layers=cfg.model.timemixer.e_layers,
            d_model=cfg.model.timemixer.d_model,
            d_ff=cfg.model.timemixer.d_ff,
            topk=cfg.model.timemixer.topk,
            use_norm=cfg.model.timemixer.use_norm,
            down_sampling_window=cfg.model.timemixer.down_sampling_window,
            num_down_sampling_layers=cfg.model.timemixer.num_down_sampling_layers,
            down_sampling_method=cfg.model.timemixer.down_sampling_method,
            num_regression_layers=cfg.model.timemixer.num_regression_layers,
        )
    else:
        raise ValueError(f'Model {model_type} not supported. Should be one of {BASELINE_MODELS.keys()}')
          
    model.to(cfg.device)
    return model
