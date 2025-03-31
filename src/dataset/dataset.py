import logging
import numpy as np
from torch.utils.data import SubsetRandomSampler
from torch_geometric.loader import DataLoader

from ..config import cfg


from ..dataset.bearing import (
    load_data as load_data_bearing,
)

from ..dataset.bridge import (
    load_data as load_data_bridge,
)

class DataLoaderHelper:
    @staticmethod
    def get_data_split_indices(dataset, params: dict):
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(params['val_split'] * dataset_size))
        if params['shuffle']:
            np.random.shuffle(indices)
        return indices[split:], indices[:split]

    @staticmethod
    def get_train_val_data_loaders(d_train, params: dict):
        if params['val_split'] == 0.0:
            return DataLoader(d_train, batch_size=params['batch_size'], shuffle=params['shuffle']), None
        else:
            train_indices, val_indices = DataLoaderHelper.get_data_split_indices(d_train, params)
            args = dict(batch_size=params['batch_size'], num_workers=params['num_workers'])
            train_loader = DataLoader(d_train, **args,
                                      sampler=SubsetRandomSampler(train_indices) if params['shuffle'] else None)
            val_loader = DataLoader(d_train, **args,
                                    sampler=SubsetRandomSampler(val_indices) if params['shuffle'] else None)
            return train_loader, val_loader


def create_data_loaders(datasets, val_split, batch_size, num_workers, shuffle=True):
    d_train, d_test = datasets

    params = {
        'val_split': val_split,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'shuffle': shuffle,
    }
    train_loader, val_loader = DataLoaderHelper.get_train_val_data_loaders(d_train, params)
    test_loader = DataLoader(d_test, batch_size=batch_size, shuffle=False, follow_batch=['x_lowf', 'x_highf']) 
    return train_loader, val_loader, test_loader


def load_data_and_create_dataloader(task):

    if task == 'bearing':
        logging.info("Loading data for bearing load prediction task")
        graph_datasets, scaler = load_data_bearing(
            cfg.dataset.dir, 
            window_size=cfg.dataset.window_size,
            stride=cfg.dataset.stride,
            y_var=cfg.dataset.y_var,
            include_lowf=cfg.dataset.include_lowf,
            include_highf=cfg.dataset.include_highf,
            use_hetero=cfg.dataset.use_hetero,
            include_exo=cfg.dataset.include_exo,
            aug_lowf_diff=cfg.dataset.aug_lowf_diff,
            only_lowf_diff=cfg.dataset.only_lowf_diff,
        )
        loaders = create_data_loaders(
            graph_datasets, 
            val_split=cfg.dataset.val_split, 
            num_workers=cfg.num_workers,
            batch_size=cfg.train.batch_size,
            shuffle=cfg.train.shuffle
        )

    elif task == 'bridge':
        logging.info("Loading data for birdge load prediction task")
        graph_datasets, scaler = load_data_bridge(
            cfg.dataset.dir, 
            window_size=cfg.dataset.window_size,
            y_var=cfg.dataset.y_var,
            stride=cfg.dataset.stride,
            include_exo=cfg.dataset.include_exo,
            include_lowf=cfg.dataset.include_lowf,
            include_highf=cfg.dataset.include_highf,
            use_hetero=cfg.dataset.use_hetero,
            aug_lowf_diff=cfg.dataset.aug_lowf_diff,
            only_lowf_diff=cfg.dataset.only_lowf_diff,
        )
        loaders = create_data_loaders(
            graph_datasets, 
            val_split=cfg.dataset.val_split, 
            num_workers=cfg.num_workers,
            batch_size=cfg.train.batch_size,
            shuffle=cfg.train.shuffle
        )

    else:
        raise NotImplementedError(f"Cannot load data for unknown task {task}, must be one of 'bearing' or 'bridge'")
    
    return loaders, scaler