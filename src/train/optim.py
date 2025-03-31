import logging
import torch
from torch.optim.lr_scheduler import _LRScheduler

from ..config import cfg
from ..metric.loss import (
    masked_mae_loss, 
    masked_mse_loss, 
    masked_rmse_loss,
    mae_loss, 
    mse_loss, 
    rmse_loss,
    
)


class WarmUpScheduler(_LRScheduler):
    def __init__(self, 
                 optimizer, 
                 warmup_iters,
                 target_lr, 
                 initial_lr=0.0001, 
                 after_scheduler=None, 
                 last_epoch=-1):
        self.warmup_iters = warmup_iters
        self.target_lr = target_lr
        self.initial_lr = initial_lr
        self.after_scheduler = after_scheduler
        self.iter = -1
        
        super(WarmUpScheduler, self).__init__(optimizer, last_epoch)

    def state_dict(self):
        """Returns the state of the scheduler as a dictionary."""
        state = {'last_epoch': self.last_epoch,
                 'iteration': self.iter,
                 'warmup_iters': self.warmup_iters,
                 'target_lr': self.target_lr,
                 'initial_lr': self.initial_lr}
        if self.after_scheduler is not None:
            state['after_scheduler_state_dict'] = self.after_scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict):
        """Loads the scheduler's state."""
        self.iter = state_dict['iteration']
        self.last_epoch = state_dict['last_epoch']
        self.warmup_iters = state_dict['warmup_iters']
        self.target_lr = state_dict['target_lr']
        self.initial_lr = state_dict['initial_lr']
        if 'after_scheduler_state_dict' in state_dict and self.after_scheduler is not None:
            self.after_scheduler.load_state_dict(state_dict['after_scheduler_state_dict'])

    def get_lr(self):
        if self.iter <= self.warmup_iters:
            # Linear interpolation between initial_lr and target_lr over the warmup period
            return [self.initial_lr + (self.target_lr - self.initial_lr) * self.iter / self.warmup_iters
                  for _ in self.optimizer.param_groups]
        else:
            # After the warmup, we need to manually handle the transition to the after_scheduler
            if self.after_scheduler:
                return [group['lr'] for group in self.optimizer.param_groups]
            return [group['lr'] for group in self.optimizer.param_groups]

    def step(self, iter=None, epoch=None, **kwargs):
        """Override the step to increment based on iterations instead of epochs."""
        if iter is not None:
            self.iter = iter
        else:
            self.iter += 1
        is_new_epoch = epoch != self.last_epoch

        if epoch is not None:
            self.last_epoch = epoch
        else:
            self.last_epoch += 1
        
        if self.iter > self.warmup_iters:
            if self.after_scheduler and is_new_epoch:
                # Adjust after_scheduler's last_epoch to sync with iterations
                self.after_scheduler.step(**kwargs)
        else:
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.get_lr()[i]


def create_criterion(crit_type='rmse', mask_loss=False):
    if crit_type == 'rmse':
        criterion = masked_rmse_loss if mask_loss else rmse_loss
    elif crit_type == 'mse' or crit_type == 'l2':
        criterion = masked_mse_loss if mask_loss else mse_loss
    elif crit_type == 'mae' or crit_type == 'l1':
        criterion = masked_mae_loss if mask_loss else mae_loss
    else:
        raise ValueError(f'Criterion {crit_type} not supported')
    return criterion        


def create_scheduler(optimizer, 
                     use_warmup=True):
    optim = cfg.optim
    
    if optim.scheduler == 'none':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            gamma=optim.lr_decay,
            step_size=optim.max_epochs + 1)
    elif optim.scheduler  == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            gamma=optim.gamma,
            step_size=optim.step_size)
    elif optim.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=optim.factor,
            patience=optim.patience,
            min_lr=optim.min_lr,
        )
    elif optim.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=optim.max_epochs)
    else:
        raise ValueError(f'Scheduler {cfg.optim.scheduler} not supported')
    
    logging.info(f'Using {cfg.optim.scheduler} as base scheduler')
    
    if use_warmup:
        scheduler = WarmUpScheduler(
            optimizer, 
            warmup_iters=optim.warmup_iters, 
            target_lr=cfg.optim.base_lr, 
            initial_lr=optim.warmup_initial_lr,
            after_scheduler=scheduler
        )
        logging.info(f'Using WarmUpScheduler with warmup_iters={optim.warmup_iters}, target_lr={optim.base_lr}, initial_lr={optim.warmup_initial_lr}')
    return scheduler


def create_optimizer(params):
    optim = cfg.optim
    
    params = filter(lambda p: p.requires_grad, params)
    if cfg.optim.optimizer == 'adam':
        optimizer = torch.optim.AdamW(
            params,
            lr=optim.base_lr,
            weight_decay=cfg.optim.weight_decay)
    elif cfg.optim.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            params, 
            lr=optim.optimizer.base_lr,
            momentum=optim.optimizer.momentum,
            weight_decay=optim.optimizer.weight_decay)
    else:
        raise ValueError(f'Optimizer {cfg.optim.optimizer} not supported')
    return optimizer
