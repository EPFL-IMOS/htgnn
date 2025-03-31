import os, stat
import shutil
import yaml

from yacs.config import CfgNode as CN


# Global config object
cfg = CN()


def set_cfg(cfg):
    r'''
    This function sets the default config value.
    1) Note that for an experiment, only part of the arguments will be used
    The remaining unused arguments won't affect anything.
    So feel free to register any argument in graphgym.contrib.config
    2) We support *at most* two levels of configs, e.g., cfg.dataset.name
    :return: configuration use by the experiment.
    '''
    if cfg is None:
        return cfg

    # ----------------------------------------------------------------------- #
    # Basic options
    # ----------------------------------------------------------------------- #
    
    # Set print destination: stdout / file / both
    cfg.print = 'both'
    
    cfg.metric_agg = 'argmin'
    cfg.metric_best = 'mae'
    
    # Select device: 'cpu', 'cuda:0', 'auto'
    cfg.device = 'cpu'
    
    # Output directory
    cfg.out_dir = 'run/results'
    
    cfg.fname = ''
    
    cfg.case = ''
    
    cfg.cfg_dest = 'config.yaml'
    
    # Random seed
    cfg.seed = 0

    # If get GPU usage
    cfg.gpu_mem = False

    # Print rounding
    cfg.round = 5

    # Additional num of worker for data loading
    cfg.num_workers = 0

    # Max threads used by PyTorch
    cfg.num_threads = 1

    # ----------------------------------------------------------------------- #
    # Debug
    # ----------------------------------------------------------------------- #

    cfg.tensorboard_each_run = True
    cfg.tensorboard_iter = False
    cfg.tensorboard_agg = True
    
    cfg.neptune_agg = False
    cfg.neptune_each_run = True
    
    cfg.draw_learned_graph = False
    
    # ----------------------------------------------------------------------- #
    # Dataset options
    # ----------------------------------------------------------------------- #
    cfg.dataset = CN()
    
    # Dir to load the dataset. If the dataset is downloaded, this is the
    # cache dir
    cfg.dataset.dir = ''
    # normalize data input
    cfg.dataset.normalize = True
    
    cfg.dataset.num_classes = 1
    cfg.dataset.train_file = 'train.csv'
    cfg.dataset.test_file = 'test.csv'
    cfg.dataset.stride = 1
    cfg.dataset.scaler_type = 'minmax'
    cfg.dataset.window_size = 30        
    cfg.dataset.val_split = 0.2
    
    # Sequence length
    cfg.dataset.window_size = 50    
    
    cfg.dataset.use_hetero = False
    cfg.dataset.include_lowf = True
    cfg.dataset.include_highf = True
    cfg.dataset.include_exo = True
    cfg.dataset.aug_lowf_diff = False
    cfg.dataset.only_lowf_diff = False

    # bearing
    cfg.dataset.y_var = ['Fx PR', 'Fz PR']
    
    # ----------------------------------------------------------------------- #
    # Training options
    # ----------------------------------------------------------------------- #
    cfg.train = CN()

    # Total graph mini-batch size
    cfg.train.batch_size = 512
    
    # Whether to shuffle the training data
    cfg.train.shuffle = True

    # Save model checkpoint every checkpoint period epochs
    cfg.train.ckpt_period = 5
    cfg.train.draw_period = 5

    # Resume training from the latest checkpoint in the output directory
    cfg.train.auto_resume = False

    # The epoch to resume. -1 means resume the latest epoch.
    cfg.train.epoch_resume = -1

    # Clean checkpoint: only keep the last ckpt
    cfg.train.ckpt_clean = True
    
    # Gradient clipping
    cfg.train.clip_grad = False
    
    cfg.train.max_grad_norm = 1
    
    cfg.train.load_pretrained = False
    
    cfg.train.pretrained_model_path = ''
    
    # early stop patience till triggered
    cfg.train.early_stop_patience = 100   
    
    # early stop min steps 
    cfg.train.early_stop_min = 100
    
    # ----------------------------------------------------------------------- #
    # Optimizer options
    # ----------------------------------------------------------------------- #
    cfg.optim = CN()

    cfg.optim.criterion = 'rmse'

    # optimizer: sgd, adam
    cfg.optim.optimizer = 'adam'

    # Base learning rate / warmup target learning rate
    cfg.optim.base_lr = 0.01

    # L2 regularization
    cfg.optim.weight_decay = 1e-5

    # SGD momentum
    cfg.optim.momentum = 0.9

    # Scheduler: none, steps, plateau
    cfg.optim.scheduler = 'plateau'
    
    # Linear warmup for learning rate
    cfg.optim.warmup = True
    
    # Warmup epochs
    cfg.optim.warmup_iters = 1000
    
    # Warmup initial learning rate
    cfg.optim.warmup_initial_lr = 1e-4

    # Steps for 'steps' policy (in epochs)
    cfg.optim.steps = [30, 60, 90]

    # Learning rate multiplier for 'steps' policy
    cfg.optim.lr_decay = 0.1

    # Maximal number of epochs
    cfg.optim.max_epochs = 30
    
    # Learning rate multiplier for 'plateau' policy
    cfg.optim.factor = 0.95
    
    cfg.optim.patience = 2
    
    # Minimum learning rate
    cfg.optim.min_lr = 1e-4
    
    # ----------------------------------------------------------------------- #
    # Shared model paramters
    # ----------------------------------------------------------------------- #
    cfg.model = CN()
    
    cfg.model.type = ''    
    
    # ----------------------------------------------------------------------- #
    # Shared model paramters
    # ----------------------------------------------------------------------- #
    cfg.model.activation = 'relu'
    cfg.model.output_dim = 0
    cfg.model.do_norm = True    
    cfg.model.norm_func = 'batch'  
    cfg.model.dropout = 0.
    
    # ----------------------------------------------------------------------- #
    # RNN model paramters
    # ----------------------------------------------------------------------- #
    cfg.model.rnn = CN()
    cfg.model.rnn.rnn_type = 'gru'
    cfg.model.rnn.n_layers = 1
    cfg.model.rnn.hidden_dim = 50
    cfg.model.rnn.bidirectional = False
    cfg.model.rnn.head_hidden_dim = 50
    cfg.model.rnn.n_head_layers = 1
    
    # ----------------------------------------------------------------------- #
    # 1DCNN model paramters
    # ----------------------------------------------------------------------- #
    cfg.model.cnn = CN()
    cfg.model.cnn.n_channels = 10
    cfg.model.cnn.kernel_size = 5
    cfg.model.cnn.n_layers = 3
    cfg.model.cnn.hidden_dim = 50
    cfg.model.cnn.stride = 1
    cfg.model.cnn.n_head_layers = 1
    
    # ----------------------------------------------------------------------- #
    # GCNN model paramters
    # ----------------------------------------------------------------------- #
    cfg.model.gcnn = CN()
    cfg.model.gcnn.gated = True
    cfg.model.gcnn.concat = True
    cfg.model.gcnn.context_dim = 1
    cfg.model.gcnn.enc_type = 'cnn'
    cfg.model.gcnn.enc_kernel_size = 5
    cfg.model.gcnn.channel_sizes = [2, 2, 1]
    cfg.model.gcnn.kernel_sizes = [3, 5],
    cfg.model.gcnn.dilations = [1, 1]
    cfg.model.gcnn.hidden_dim = 50
    cfg.model.gcnn.stride = 1
    cfg.model.gcnn.n_head_layers = 1
    
    # ----------------------------------------------------------------------- #
    # TimeMixer model paramters
    # ----------------------------------------------------------------------- #
    cfg.model.timemixer = CN()
    cfg.model.timemixer.e_layers = 6
    cfg.model.timemixer.decomp_method = 'moving_avg' # 'moving_avg', 'dft_decomp'
    cfg.model.timemixer.moving_avg_len = 25
    cfg.model.timemixer.num_down_sampling_layers = 3
    cfg.model.timemixer.down_sampling_window = 2
    cfg.model.timemixer.down_sampling_method = 'avg' # 'avg', 'max', 'conv'
    cfg.model.timemixer.d_model = 64
    cfg.model.timemixer.num_regression_layers = 1
    cfg.model.timemixer.use_norm = 1
    cfg.model.timemixer.topk = 3
    cfg.model.timemixer.d_ff = 64
    
    # ----------------------------------------------------------------------- #
    # HTGNN paramters
    # ----------------------------------------------------------------------- #
    cfg.model.htgnn = CN()
    cfg.model.htgnn.aug_exovar = True
    cfg.model.htgnn.exo_kernel_size = 5
    cfg.model.htgnn.node_embed_dim = 5
    cfg.model.htgnn.exovar_embed_dim = 5
    cfg.model.htgnn.gnn_embed_dim = 64
    cfg.model.htgnn.gnn_type = 'gcn'
    cfg.model.htgnn.num_gnn_layers = 2
    cfg.model.htgnn.do_encoder_norm = False
    cfg.model.htgnn.do_gnn_norm = False
    cfg.model.htgnn.gnn_norm_type = 'graph'
    cfg.model.htgnn.hetero_agg = 'sum'

    # htgnn encoder
    cfg.model.htgnn.cnn = CN()
    cfg.model.htgnn.cnn.gating = False
    cfg.model.htgnn.cnn.emb_concat = False
    cfg.model.htgnn.cnn.channel_sizes = [2, 2, 1]
    cfg.model.htgnn.cnn.kernel_sizes = [3, 5]
    cfg.model.htgnn.cnn.dilations = [1, 1]

    # htgnn head
    cfg.model.htgnn.head = CN()
    cfg.model.htgnn.head.embed_dim = 32
    cfg.model.htgnn.head.num_layers = 2
    cfg.model.htgnn.head.use_rnn = True
    cfg.model.htgnn.head.rnn_type = 'lstm'
    cfg.model.htgnn.head.rnn_bidirectional = True
    cfg.model.htgnn.head.norm_type = 'batch'
    cfg.model.htgnn.head.do_norm = True
    
    # htgnn ablation study
    cfg.model.htgnn.gnn_type = 'gcn'
    cfg.model.htgnn.enc_type = 'cnn'
    
    # ----------------------------------------------------------------------- #
    # MTGAT paramters
    # ----------------------------------------------------------------------- #
    cfg.model.mtgat = CN()

    cfg.model.mtgat.encoder = CN()
    cfg.model.mtgat.encoder.encode = True
    cfg.model.mtgat.encoder.in_depth = False
    cfg.model.mtgat.encoder.conv_pad = True
    cfg.model.mtgat.encoder.padding = 0
    cfg.model.mtgat.encoder.stride = 1
    cfg.model.mtgat.encoder.dilation = 1
    cfg.model.mtgat.encoder.kernel_size = 5

    cfg.model.mtgat.use_gatv2 = True

    cfg.model.mtgat.num_feat_layers = 1
    cfg.model.mtgat.num_temp_layers = 1
    
    cfg.model.mtgat.feat_gat_embed_dim = 100
    cfg.model.mtgat.time_gat_embed_dim = 100
    
    cfg.model.mtgat.encode_edge_attr = True
    cfg.model.mtgat.edge_embed_dim = 3
    cfg.model.mtgat.num_edge_types = 4
        
    cfg.model.mtgat.gru_n_layers = 1
    cfg.model.mtgat.gru_hid_dim = 100
    
    cfg.model.mtgat.head_n_layers = 3
    cfg.model.mtgat.head_hid_dim = 100
    
    cfg.model.mtgat.recon_n_layers = 1
    cfg.model.mtgat.recon_hid_dim = 100
    
    cfg.model.mtgat.decoder_hidden = 100
    
    cfg.model.mtgat.forecast = True
    cfg.model.mtgat.recons = True
    
    # ----------------------------------------------------------------------- #
    # Tasks
    # ----------------------------------------------------------------------- #
    cfg.task = CN()
    
    # Select task type: 'bearing', 'bridge', 
    cfg.task.type = 'bearing'
    
    cfg.task.track_graph = False
    
    # Select task level: 'graph', 'node'
    cfg.task.level = 'graph'
    
    # Select task training type: 'forecast', 'reconst', 'mapping'
    cfg.task.train_type = 'mapping' 


def dump_cfg(cfg, cfg_file=None):
    r"""
    Dumps the config to the output directory specified in
    :obj:`cfg.out_dir`
    Args:
        cfg (CfgNode): Configuration node
    """
    if cfg_file is None:
        os.makedirs(cfg.out_dir, exist_ok=True)
        cfg_file = os.path.join(cfg.out_dir, cfg.cfg_dest)
    with open(cfg_file, 'w') as f:
        cfg.dump(stream=f)


def load_yaml(yaml_path):
    with open(yaml_path) as file:
        return yaml.safe_load(file)


def load_cfg(cfg_file):
    r"""
    Load configurations from file path
    Args:
        cfg_file (string): config file path
    """
    cfg.merge_from_file(cfg_file)
    return cfg


def remove_readonly(func, path, _):
    "Clear the readonly bit and reattempt the removal"
    os.chmod(path, stat.S_IWRITE)
    func(path)
    

def makedirs_rm_exist(dir):
    dir = os.path.abspath(dir)
    if os.path.isdir(dir):
        shutil.rmtree(dir, onerror=remove_readonly)
    os.makedirs(dir, exist_ok=True)


def get_fname(fname):
    r"""
    Extract filename from file name path
    Args:
        fname (string): Filename for the yaml format configuration file
    """
    fname = fname.split('/')[-1]
    if fname.endswith('.yaml'):
        fname = fname[:-5]
    elif fname.endswith('.yml'):
        fname = fname[:-4]
    return fname


def get_out_dir(out_dir, fname):
    fname = get_fname(fname)
    cfg.fname = fname
    out_dir = f'run/{out_dir}/{cfg.task.type}/{cfg.case}/{fname}'
    return out_dir


def set_out_dir(out_dir, fname, exist_ok=True):
    r"""
    Create the directory for full experiment run
    Args:
        out_dir (string): Directory for output, specified in :obj:`cfg.out_dir`
        fname (string): Filename for the yaml format configuration file
    """
    cfg.out_dir = get_out_dir(out_dir, fname)
    # Make output directory
    if cfg.train.auto_resume or exist_ok:
        os.makedirs(cfg.out_dir, exist_ok=True)
    else:
        makedirs_rm_exist(cfg.out_dir)
    return cfg.out_dir


def set_run_dir(out_dir, seed=None):
    r"""
    Create the directory for each random seed experiment run
    Args:
        out_dir (string): Directory for output, specified in :obj:`cfg.out_dir`
        fname (string): Filename for the yaml format configuration file
    """
    if seed is None:
        seed = cfg.seed
    cfg.run_dir = f'{out_dir}/{seed}'
    # Make output directory
    if cfg.train.auto_resume:
        os.makedirs(cfg.run_dir, exist_ok=True)
    else:
        makedirs_rm_exist(cfg.run_dir)

set_cfg(cfg)