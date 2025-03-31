import logging
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData, Dataset, Data
from torch_geometric.utils import to_undirected

from ..model.dict import SCALER_DICT
from ..config import cfg


class BridgeDataset(Dataset):
    def __init__(self, 
                 dataframe, 
                 window_size=10, 
                 stride=1, 
                 include_lowf=True,
                 include_highf=True,
                 aug_lowf_diff=False,
                 only_lowf_diff=True,
                 use_hetero=False,
                 scaler=None,
                 include_exo=True,
                 y_var=['TrainLoad']
                 ):
        """Sliding window dataset

        Args:
            dataframe (pd.DataFrame): dataframe containing scenario descriptors and sensor reading
            window_size (int, optional): sequence window length. Defaults to 50.
            stride (int, optional): data stride length. Defaults to 1.
        """
        # assert one of include_lowf, include_highf is True
        assert include_lowf or include_highf, 'At least one of include_lowf or include_highf should be True'
        
        self.window_size = window_size
        self.stride = stride
        self.scaler = scaler
        self.use_hetero = use_hetero
        self.include_lowf = include_lowf
        self.include_highf = include_highf
        self.include_exo = include_exo
        self.highf_feature_names = [f'acc_{i+1}' for i in range(10)]
        self.lowf_feature_names = [f'disp_{i+1}' for i in range(10)]
        self.exovar_feature_names = ['Temperature']
        
        self.n_exovars = len( self.exovar_feature_names) if include_exo else 0
        self.n_lowf_nodes = len(self.lowf_feature_names)
        self.n_highf_nodes = len(self.highf_feature_names)
        self.feature_dim_lowf = 1
        self.feature_dim_highf = 1
        
        if aug_lowf_diff:
            diff_lowf_feature_names = [f'{col}_diff' for col in self.lowf_feature_names]
            if only_lowf_diff:
                self.lowf_feature_names = diff_lowf_feature_names
        self.n_lowf_features = len(self.lowf_feature_names)
        self.n_highf_features = len(self.highf_feature_names)
        all_col_names = self.highf_feature_names + self.lowf_feature_names + self.exovar_feature_names
    
        if scaler is None:
            args={}
            if cfg.dataset.scaler_type=='robust':
                args['quantile_range'] = (1, 99)
            if cfg.dataset.scaler_type=='minmax':
                 args['feature_range'] = (cfg.dataset.feature_range[0], cfg.dataset.feature_range[1])
            if cfg.dataset.scaler_type=='cminmax':
                if aug_lowf_diff:
                    args['indices_1'] = range(len(self.highf_feature_names)+len(self.lowf_feature_names))
                    args['indices_2'] = [_ for _ in range(len(self.highf_feature_names)+len(self.lowf_feature_names), len(all_col_names))]
                else:
                    args['indices_1'] = range(len(self.highf_feature_names))
                    args['indices_2'] = range(len(self.highf_feature_names), len(all_col_names))
            self.scaler = SCALER_DICT[cfg.dataset.scaler_type](**args)
            dataframe[all_col_names] = self.scaler.fit_transform(dataframe[all_col_names])
        else:
            self.scaler = scaler
            dataframe[all_col_names] = self.scaler.transform(dataframe[all_col_names])

        n_samples = len(dataframe)
        self.exovar = np.array(dataframe[self.exovar_feature_names]).astype(np.float32).reshape(n_samples, len(self.exovar_feature_names), 1)
        self.lowf_vars = np.array(dataframe[self.lowf_feature_names]).astype(np.float32).reshape(n_samples, self.n_lowf_nodes, -1)
        self.highf_vars = np.array(dataframe[self.highf_feature_names]).astype(np.float32).reshape(n_samples, self.n_highf_nodes, -1)
        self.y = np.array(dataframe[y_var]).astype(np.float32).reshape(-1, len(y_var), 1)
        self.y[:, 0] /= 40000
        if len(y_var) > 1:
            self.y[:, 1] /= 50
        if self.use_hetero:
            self._construct_graph()
        self._indices = self._get_indices(dataframe)

    def len(self) -> int:
        return len(self._indices)
    
    def get(self, idx):
        return self.__getitem__(idx)

    def _get_indices(self, dataframe):
        # get indicies where sample number is not continuous
        sample_num = dataframe['SampleNumber']
        sample_num_diff = sample_num.diff()
        break_idx = sample_num_diff[sample_num_diff != 0].index
        idx_starts = [0] + break_idx.to_list()
        
        idx_list = []
        for idx_start, idx_end in zip(idx_starts, break_idx):
            idx_list += [(i, i+self.window_size) for i in np.arange(idx_start, idx_end + 1, self.stride) if i+self.window_size < idx_end]
        
        return idx_list

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        i_start, i_stop = self._indices[idx]

        lowf_i = self.lowf_vars[i_start:i_stop, :, :] # [seq_len, n_nodes, n_f]
        highf_i = self.highf_vars[i_start:i_stop, :, :] # [seq_len, n_nodes, n_f]
        exo_i = self.exovar[i_start:i_stop, :, :] # [seq_len, 2, n_f]
        yi = self.y[i_stop].reshape(-1, 1)

        if self.use_hetero:
            g = HeteroData(
                lowf={'x': torch.from_numpy(lowf_i).permute(1, 0, 2)},
                highf={'x': torch.from_numpy(highf_i).permute(1, 0, 2)},
                exo_var={'x': torch.from_numpy(exo_i).permute(1, 0, 2)},
                y=torch.from_numpy(yi),
            )
            for k, v in self.edge_relations.items():
                g[k].edge_index = v
        else:
            data_args = dict(
                y=torch.from_numpy(yi),
            )
            xi = None
            if self.include_lowf:
                lowf_i = lowf_i.reshape(lowf_i.shape[0], -1).T
                xi = lowf_i
            if self.include_highf:
                highf_i = highf_i.reshape(highf_i.shape[0], -1).T
                xi = np.vstack([xi, highf_i]) if xi is not None else highf_i
            if self.include_exo:
                exo_i = exo_i.reshape(exo_i.shape[0], -1).T
                data_args['exo_var'] = torch.from_numpy(exo_i)
            data_args['x'] = torch.from_numpy(xi)
            g = Data(**data_args)

        return g

    def _construct_graph(self):
        edge_index_lowf = to_undirected(torch.tensor([
            [0, 1, 2, 3, 4, 5, 6, 7, 8], 
            [1, 2, 3, 4, 5, 6, 7, 8, 9]
        ]))
        edge_index_highf = to_undirected(torch.tensor([
            [0, 1, 2, 3, 4, 5, 6, 7, 8], # [0, 1, 2], 
            [1, 2, 3, 4, 5, 6, 7, 8, 9]  # [1, 2, 3]
        ]))
        # heterogenous relationships
        edge_index_highf_lowf = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], # [0, 0, 1, 1, 2, 2, 3, 3],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], # [0, 1, 1, 2, 2, 3, 3, 4], 
        ])
        edge_index_lowf_highf = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], # [0, 0, 1, 1, 2, 2, 3, 3],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # [0, 1, 1, 2, 2, 3, 3, 4], 
        ])
        self.edge_relations = {
            ('lowf', 'to', 'lowf'):{ 'edge_index': edge_index_lowf },
            ('highf', 'to', 'lowf'):{ 'edge_index': edge_index_highf_lowf },
            ('lowf', 'to', 'highf'):{ 'edge_index': edge_index_lowf_highf },
            ('highf', 'to', 'highf'): { 'edge_index': edge_index_highf }
        }
        self.n_lowf_edges = self.edge_relations[('lowf', 'to', 'lowf')]['edge_index'].shape[1]
        self.n_highf_edges = self.edge_relations[('highf', 'to', 'highf')]['edge_index'].shape[1]
        self.n_highf_lowf_edges = self.edge_relations[('highf', 'to', 'lowf')]['edge_index'].shape[1]
        self.n_lowf_highf_edges = self.edge_relations[('lowf', 'to', 'highf')]['edge_index'].shape[1]
      
    def __repr__(self):
        info = f'SlidingWindowDataset(window_size={self.window_size}, stride={self.stride})'
        info += f'\nNumber of samples: {len(self)}'
        info += f'\nDisplacement feature shape: {self.lowf_vars.shape[1:]}'
        info += f'\nAcceleration feature shape: {self.highf_vars.shape[1:]}'
        info += f'\nlowf_feature_names: {self.lowf_feature_names}'
        info += f'\nhighf_feature_names: {self.highf_feature_names}'
        return info


def load_data(
    dataset_dir, 
    window_size=50, 
    stride=1,
    y_var=None,
    include_lowf=True,
    include_highf=True,
    include_exo=True,
    aug_lowf_diff=False,
    only_lowf_diff=True,
    use_hetero=False,
):
    dataset_dir = f'run/{dataset_dir}'
    train_file_path = f'{dataset_dir}/{cfg.dataset.train_file}'
    test_file_path = f'{dataset_dir}/{cfg.dataset.test_file}'
    df_train =  pd.read_csv(train_file_path)
    df_test =  pd.read_csv(test_file_path)

    args = dict(
        window_size=window_size, 
        stride=stride,
        y_var=y_var,
        include_lowf=include_lowf,
        include_highf=include_highf,
        include_exo=include_exo,
        aug_lowf_diff=aug_lowf_diff,
        only_lowf_diff=only_lowf_diff,
        use_hetero=use_hetero,
    )

    train_dataset = BridgeDataset(df_train, scaler=None, **args)
    test_dataset = BridgeDataset(df_test, scaler=train_dataset.scaler, **args)
    
    datasets = [train_dataset, test_dataset]
    
    cfg.dataset.n_train = len(train_dataset)
    cfg.dataset.n_test = len(test_dataset)
    num_graphs = cfg.dataset.n_train + cfg.dataset.n_test
    cfg.dataset.feat_dim_lowf = train_dataset.feature_dim_lowf
    cfg.dataset.feat_dim_highf = train_dataset.feature_dim_highf
    
    cfg.dataset.n_nodes_lowf = train_dataset.n_lowf_nodes
    cfg.dataset.n_nodes_highf = train_dataset.n_highf_nodes
    cfg.dataset.n_exovars = train_dataset.n_exovars
    
    if use_hetero:
        disp_feature_size = test_dataset.feature_dim_lowf
        acc_feature_size = test_dataset.feature_dim_highf
        cfg.dataset.n_edges_lowf =  test_dataset.n_lowf_edges
        cfg.dataset.n_edges_highf =  test_dataset.n_highf_edges
        cfg.dataset.n_mixf_edges =  test_dataset.n_lowf_highf_edges
        cfg.dataset.n_nodes = test_dataset.n_lowf_nodes + test_dataset.n_highf_nodes
    else: 
        cfg.dataset.n_nodes = test_dataset[0].x.shape[0]
        feature_size = test_dataset[0].x.shape[1] 
    
    d_info = '\n'
    d_info += f'Loaded data from {dataset_dir}\n'
    d_info += f'------------ Basic -----------\n'
    d_info += f'# Samples: {num_graphs}\n'
    if use_hetero:
        d_info += f'# Nodes Disp.: {cfg.dataset.n_nodes_lowf}\n'
        d_info += f'# Nodes Acc.: {cfg.dataset.n_nodes_highf}\n'
        d_info += f'# Edges Disp.: {cfg.dataset.n_edges_lowf}\n'
        d_info += f'# Edges Acc.: {cfg.dataset.n_edges_highf}\n'
        d_info += f'# Edges Disp-Acc: : {cfg.dataset.n_mixf_edges}\n'
    else:
        d_info += f'# Nodes: {cfg.dataset.n_nodes}\n'
    d_info += f'# Sequence Length: {window_size}\n'
    
    d_info += f'------------ Feature -----------\n'
    if use_hetero:
        d_info += f'Disp. Feature Size: {disp_feature_size}\n'
        d_info += f'Acc. Feature Size: {acc_feature_size}\n'
    else:
        d_info += f'Feature Size: {feature_size}\n'

    d_info += f'------------ Scaling -----------\n'
    if cfg.dataset.scaler_type == 'minmax' or cfg.dataset.scaler_type == 'cminmax':
        d_info += f'{np.array2string(train_dataset.scaler.data_range_, precision=2,)}\n'
    elif cfg.dataset.scaler_type == 'robust':
        d_info += f'{np.array2string(train_dataset.scaler.center_, precision=2,)}\n'
        d_info += f'{np.array2string(train_dataset.scaler.scale_, precision=2,)}\n'
    elif cfg.dataset.scaler_type == 'standard':
        d_info += f'{np.array2string(train_dataset.scaler.mean_, precision=2,)}\n'

    d_info += f'------------ Split -----------\n'
    d_info += f'Train: {len(train_dataset)}/{num_graphs}\n'
    d_info += f' Test: {len(test_dataset)}/{num_graphs}\n'

    logging.info(d_info)
    return datasets, train_dataset.scaler
