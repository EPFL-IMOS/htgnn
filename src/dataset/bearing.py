import logging
import numpy as np
import pandas as pd
import torch

from torch_geometric.data import HeteroData, Dataset, Data
from torch_geometric.utils import to_undirected

from ..model.dict import SCALER_DICT
from ..config import cfg


class BearingDataset(Dataset):
    def __init__(self, 
                 dataframe, 
                 window_size=10, 
                 stride=1, 
                 diff_resol=5*60,
                 include_lowf=True,
                 include_highf=True,
                 use_hetero=False,
                 include_exo=False,
                 aug_lowf_diff=False,
                 only_lowf_diff=False,
                 scaler=None,
                 y_var=['Fx PR', 'Fz PR']):
        """Sliding window dataset

        Args:
            dataframe (pd.DataFrame): dataframe containing scenario descriptors and sensor reading
            window_size (int, optional): sequence window length. Defaults to 50.
            stride (int, optional): data stride length. Defaults to 1.
        """
        # assert one of include_lowf, include_highf is True
        assert include_lowf or include_highf, 'At least one of include_lowf or include_highf should be True'
        
        dataframe = dataframe
        self.window_size = window_size
        self.stride = stride
        self.scaler = scaler
        self.use_hetero = use_hetero
        self.include_lowf = include_lowf
        self.include_highf = include_highf
        self.include_exo = include_exo
        input_var = [col for col in dataframe.columns if col.startswith('TB')]
        # outering temperature features
        self.temp_feature_names = [col for col in input_var if 'T_OR' in col]
        # inner ring temperature features
        self.temp_feature_names += [col for col in input_var if 'T_IR' in col]
        self.vib_feature_names = [
            'TB1_VB_006_AX', 'TB2_VB_006_AX', 
            'TB1_VB_186_RA', 'TB2_VB_186_RA',
            'TB1_VB_006_RA', 'TB2_VB_006_RA'
        ]
        exovar_feature_name = ['nrot UUT Mot']
        
        self.n_exovars = len(exovar_feature_name) if include_exo else 0
        self.n_lowf_nodes = len(self.temp_feature_names)
        self.n_highf_nodes = len(self.vib_feature_names)
        self.feature_dim_lowf = 1
        self.feature_dim_highf = 1
        
        if aug_lowf_diff:
            diff_lowf_feature_names = [f'DIFF_{col}' for col in self.temp_feature_names]
            # check if the diff features are already in the dataframe
            if not all([col in dataframe.columns for col in diff_lowf_feature_names]):
                dataframe[diff_lowf_feature_names] = dataframe[self.temp_feature_names].diff(diff_resol)
                dataframe = dataframe.dropna()
            if only_lowf_diff:
                self.temp_feature_names = diff_lowf_feature_names
            else:
                self.temp_feature_names += diff_lowf_feature_names
                self.feature_dim_lowf = 2
    
        self.n_lowf_features = len(self.temp_feature_names)
        self.n_highf_features = len(self.vib_feature_names)
        all_col_names = self.temp_feature_names + self.vib_feature_names + exovar_feature_name
        
        if scaler is None:
            args={}
            if cfg.dataset.scaler_type=='robust':
                args['quantile_range'] = (1, 99)
            if cfg.dataset.scaler_type=='minmax':
                 args['feature_range'] = (cfg.dataset.feature_range[0], cfg.dataset.feature_range[1])
            if cfg.dataset.scaler_type=='cminmax':
                    args['indices_1'] = range(len(self.temp_feature_names))
                    args['indices_2'] = [_ for _ in range(len(self.temp_feature_names), len(all_col_names))]
            self.scaler = SCALER_DICT[cfg.dataset.scaler_type](**args)
            dataframe[all_col_names] = self.scaler.fit_transform(dataframe[all_col_names])
        else:
            self.scaler = scaler
            dataframe[all_col_names] = self.scaler.transform(dataframe[all_col_names])

        n_samples = len(dataframe)
        self.rot = np.array(dataframe[exovar_feature_name]).astype(np.float32).reshape(n_samples, 1, 1)
        self.temp_vars = np.array(dataframe[self.temp_feature_names]).astype(np.float32).reshape(n_samples, self.n_lowf_nodes, -1)
        self.vib_vars = np.array(dataframe[self.vib_feature_names]).astype(np.float32).reshape(n_samples, self.n_highf_nodes, -1)
        self.y = np.array(dataframe[y_var]).astype(np.float32).reshape(-1, len(y_var), 1)
        self.y[:, 0]/=8000. # scale Fx by 8000
        self.y[:, 1]/=1000. # scale Fz by 1000
        if self.use_hetero:
            self._construct_hetero_graph()
        else:
            self._construct_homo_graph()
        self._indices = self._get_indices(dataframe)

    def len(self) -> int:
        return len(self._indices)
    
    def get(self, idx):
        return self.__getitem__(idx)

    def _get_indices(self, dataframe):
        # get indicies where the time is not continous
        t = pd.to_datetime(dataframe.index)
        tn1 = t[1:]
        dt = (tn1 - t[:-1]).total_seconds().to_numpy()
        break_idx = [l[0] for l in np.argwhere(dt != 1)]
        
        idx_starts = [0] + break_idx
        
        idx_list = []
        for idx_start, idx_end in zip(idx_starts, break_idx):
            idx_list += [(i, i+self.window_size) for i in np.arange(idx_start, idx_end + 1, self.stride) if i+self.window_size < idx_end]
        
        return idx_list

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        i_start, i_stop = self._indices[idx]

        temp_i = self.temp_vars[i_start:i_stop, :, :] # [seq_len, n_nodes, n_f]
        vib_i = self.vib_vars[i_start:i_stop, :, :] # [seq_len, n_nodes, n_f]
        rot_i = self.rot[i_start:i_stop, :, :] # [seq_len, n_nodes, n_f]
        yi = self.y[i_stop].reshape(-1, 1)

        if self.use_hetero:
            g = HeteroData(
                lowf={'x': torch.from_numpy(temp_i).permute(1, 0, 2)},
                highf={'x': torch.from_numpy(vib_i).permute(1, 0, 2)},
                exo_var={'x': torch.from_numpy(rot_i).permute(1, 0, 2)},
                y=torch.from_numpy(yi),
            )
            for k, v in self.edge_relations.items():
                g[k].edge_index = v
        else:
            data_args = dict(
                y=torch.from_numpy(yi),
                edge_index=self.edge_index,
            )
            xi = None
            if self.include_lowf:
                temp_i = temp_i.reshape(temp_i.shape[0], -1).T
                xi = temp_i
            if self.include_highf:
                vib_i = vib_i.reshape(vib_i.shape[0], -1).T
                xi = np.vstack([xi, vib_i]) if xi is not None else vib_i
            if self.include_exo:
                rot_i = rot_i.reshape(rot_i.shape[0], -1).T
                data_args['exo_var'] = torch.from_numpy(rot_i)
            data_args['x'] = torch.from_numpy(xi)
            g = Data(**data_args)

        return g
    
    def _construct_homo_graph(self):
        if self.include_lowf:
            edge_list = [
                [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14, 16, 17, 16, 18, 0, 14, 6, 8, 1, 15, 7, 9], 
                [2, 4, 6, 8, 10, 12, 14, 0, 3, 5, 7, 9, 11, 13, 15, 1, 1, 3, 5, 7, 9, 11, 13, 15, 18, 19, 17, 19, 16, 16, 18, 18, 17, 17, 19, 19]
            ]
            if self.include_highf:
                edge_list = [
                    [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14, 16, 17, 16, 18, 0, 14, 6, 8, 1, 15, 7, 9, 20, 20, 24, 24, 22, 22, 21, 21, 25, 25, 23, 23], 
                    [2, 4, 6, 8, 10, 12, 14, 0, 3, 5, 7, 9, 11, 13, 15, 1, 1, 3, 5, 7, 9, 11, 13, 15, 18, 19, 17, 19, 16, 16, 18, 18, 17, 17, 19, 19, 0, 14, 0, 14, 6, 8, 1, 15, 1, 15, 7, 9]
                ]
        else:
            if self.include_highf:
                edge_list = [
                    [0, 2, 4, 1, 3, 5, 0, 2, 4], 
                    [2, 4, 0, 3, 5, 1, 1, 3, 5]
                ]
            else:
                raise ValueError('At least one of include_lowf or include_highf should be True')
        self.edge_index = to_undirected(torch.tensor(edge_list))

    def _construct_hetero_graph(self):
        edge_index_lowf = to_undirected(torch.tensor([
            [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14, 16, 17, 16, 18, 0, 14, 6, 8, 1, 15, 7, 9], 
            [2, 4, 6, 8, 10, 12, 14, 0, 3, 5, 7, 9, 11, 13, 15, 1, 1, 3, 5, 7, 9, 11, 13, 15, 18, 19, 17, 19, 16, 16, 18, 18, 17, 17, 19, 19]
        ]))
        edge_index_highf = to_undirected(torch.tensor([
            [0, 2, 4, 1, 3, 5, 0, 2, 4], 
            [2, 4, 0, 3, 5, 1, 1, 3, 5]
        ]))
        # heterogenous relationships
        edge_index_highf_lowf = torch.tensor([
            [0, 0, 4, 4, 2, 2, 1, 1, 5, 5, 3, 3],
            [0, 14, 0, 14, 6, 8, 1, 15, 1, 15, 7, 9], 
        ])
        edge_index_lowf_highf = torch.tensor([
            [0, 14, 0, 14, 6, 8, 1, 15, 1, 15, 7, 9], 
            [0, 0, 4, 4, 2, 2, 1, 1, 5, 5, 3, 3],
        ])
        self.edge_relations = {
            ('lowf', 'to', 'lowf'):{ 'edge_index': edge_index_lowf },
            ('highf', 'to', 'lowf'):{ 'edge_index': edge_index_highf_lowf },
            ('lowf', 'to', 'highf'):{ 'edge_index': edge_index_lowf_highf },
            ('highf', 'to', 'highf'): { 'edge_index': edge_index_highf }
        }
        self.n_lowf_edges = edge_index_lowf.shape[1]
        self.n_highf_edges = edge_index_highf.shape[1]
        self.n_highf_lowf_edges = edge_index_highf_lowf.shape[1]
        self.n_lowf_highf_edges = edge_index_lowf_highf.shape[1]
      
    def __repr__(self):
        info = f'SlidingWindowDataset(window_size={self.window_size}, stride={self.stride})'
        info += f'\nNumber of samples: {len(self)}'
        info += f'\nTemperature feature shape: {self.temp_vars.shape[1:]}'
        info += f'\nVibration feature shape: {self.vib_vars.shape[1:]}'
        info += f'\ntemp_feature_names: {self.temp_feature_names}'
        info += f'\nvib_feature_names: {self.vib_feature_names}'
        return info



def load_data(
    dataset_dir, 
    window_size=50, 
    stride=1,
    y_var=None,
    include_lowf=True,
    include_highf=True,
    use_hetero=False,
    include_exo=False,
    aug_lowf_diff=False,
    only_lowf_diff=False,
):
    dataset_dir = f'run/{dataset_dir}'
    train_file_path = f'{dataset_dir}/{cfg.dataset.train_file}'
    test_file_path = f'{dataset_dir}/{cfg.dataset.test_file}'
    df_train =  pd.read_csv(train_file_path, index_col=0)
    df_test =  pd.read_csv(test_file_path, index_col=0)

    args = dict(
        window_size=window_size, 
        stride=stride,
        y_var=y_var,
        include_lowf=include_lowf,
        include_highf=include_highf,
        use_hetero=use_hetero,
        include_exo=include_exo,
        aug_lowf_diff=aug_lowf_diff,
        only_lowf_diff=only_lowf_diff,
    )

    train_dataset = BearingDataset(df_train, scaler=None, **args)
    test_dataset = BearingDataset(df_test, scaler=train_dataset.scaler, **args)
    
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
        temp_feature_size = test_dataset.feature_dim_lowf
        vib_feature_size = test_dataset.feature_dim_highf
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
        d_info += f'# Nodes Temp.: {cfg.dataset.n_nodes_lowf}\n'
        d_info += f'# Nodes Vib.: {cfg.dataset.n_nodes_highf}\n'
        d_info += f'# Edges Temp: {cfg.dataset.n_edges_lowf}\n'
        d_info += f'# Edges Vib.: {cfg.dataset.n_edges_highf}\n'
        d_info += f'# Edges Temp-Vib: : {cfg.dataset.n_mixf_edges}\n'
    else:
        d_info += f'# Nodes: {cfg.dataset.n_nodes}\n'
    d_info += f'# Sequence Length: {window_size}\n'
    
    d_info += f'------------ Feature -----------\n'
    if use_hetero:
        d_info += f'Temp. Feature Size: {temp_feature_size}\n'
        d_info += f'Vib. Feature Size: {vib_feature_size}\n'
    else:
        d_info += f'Feature Size: {feature_size}\n'

    d_info += f'------------ Scaling -----------\n'
    if cfg.dataset.scaler_type in ['minmax', 'cminmax']:
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


