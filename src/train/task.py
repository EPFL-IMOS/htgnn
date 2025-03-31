import logging
import numpy as np
import torch
from sklearn.metrics import mean_squared_error

from ..config import cfg
from ..utils.plots import (
    plot_bearing_load, 
    plot_bridge_load,
)


def create_task(task_type, **task_args):
    if task_type == 'bearing':
        task = BearingLoadPredictionTask(**task_args)
    elif task_type == 'bridge':
        task = BridgeLoadPredictionTask(**task_args)
    else:
        raise ValueError(f"Invalid task type: '{task_type}', should be one of 'bearing', 'bridge'.")
    
    logging.info(f"Create {task.__class__.__name__}.")
    return task


class BaseTask:
    def __init__(self, 
        task_train_type='mapping',
        track_graph=False,
    ):
        self.task_train_type=task_train_type
        self.track_graph=track_graph
    
    def log_task_figures(self, loggers, **kwargs):
        raise NotImplementedError
    
    def eval_task_epoch(self, **kwargs):
        raise NotImplementedError


class BridgeLoadPredictionTask(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'bridge_laod'
        self.task_level = 'graph'
    
    def log_task_figures(self, loggers, cur_epoch, is_best=False):
        if is_best:
            preds = torch.cat(loggers['test']._pred).detach().cpu().numpy()
            trues = torch.cat(loggers['test']._true).detach().cpu().numpy()
            contr =  torch.cat(loggers['test']._custom_stats['exo_var']).detach().cpu().numpy()            
            f = plot_bridge_load(preds, trues, contr)
            if cfg.tensorboard_each_run:
                loggers['test'].tb_writer.add_figure('figures/damage', f, global_step=cur_epoch)
            if cfg.neptune_each_run:
                fname = f'best/damage'
                loggers['test'].neptune_writer[fname].upload(f)
            # save the figure locally for the best model
            f.savefig(f'{cfg.run_dir}/best_load.png')
        else:
            for split in ['eval', 'test']:
                preds = torch.cat(loggers[split]._pred).detach().cpu().numpy()
                trues = torch.cat(loggers[split]._true).detach().cpu().numpy()
                contr =  torch.cat(loggers[split]._custom_stats['exo_var']).detach().cpu().numpy()            
                f = plot_bridge_load(preds, trues, contr)
                if cfg.tensorboard_each_run:
                    loggers[split].tb_writer.add_figure('figures/damage', f, global_step=cur_epoch)
                if cfg.neptune_each_run:
                    fname = f'{loggers[split].name}/damage'
                    loggers[split].neptune_writer[fname].append(f, step=cur_epoch)
        

    def eval_task_epoch(self, loggers, cur_epoch, is_best, **kwargs):
        for split in ['eval', 'test']:
            preds = torch.cat(loggers[split]._pred).detach().cpu().numpy()
            trues = torch.cat(loggers[split]._true).detach().cpu().numpy()
            
            load_preds = preds[:, 0, 0] * 40000
            load_trues = trues[:, 0, 0] * 40000

            mae = np.mean(np.abs(load_trues - load_preds))
            mape = np.mean(np.abs((load_trues - load_preds) / (load_trues + 400)))   
            rmse = np.sqrt(mean_squared_error(load_trues, load_preds))
            
            args = {
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape),
            }
            loggers[split].update_custom_stats(**args)
        self.log_task_figures(loggers, cur_epoch, is_best=is_best)


class BearingLoadPredictionTask(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'bearing_load'
        self.task_level = 'graph'
    
    def log_task_figures(self, loggers, cur_epoch, is_best=False):
        if is_best:
            preds = torch.cat(loggers['test']._pred).detach().cpu().numpy()
            trues = torch.cat(loggers['test']._true).detach().cpu().numpy()
            rot =  torch.cat(loggers['test']._custom_stats['exo_var']).detach().cpu().numpy()
            f = plot_bearing_load(preds, trues, rot=rot)
            if cfg.tensorboard_each_run:
                loggers['test'].tb_writer.add_figure('figures/load', f, global_step=cur_epoch)
            if cfg.neptune_each_run:
                fname = f'best/load'
                loggers['test'].neptune_writer[fname].upload(f)
            # save the figure locally for the best model
            f.savefig(f'{cfg.run_dir}/best_load.png')
        else:
            for split in ['eval', 'test']:
                preds = torch.cat(loggers[split]._pred).detach().cpu().numpy()
                trues = torch.cat(loggers[split]._true).detach().cpu().numpy()
                rot =  torch.cat(loggers[split]._custom_stats['exo_var']).detach().cpu().numpy()
                f = plot_bearing_load(preds, trues, rot=rot)
                if cfg.tensorboard_each_run:
                    loggers[split].tb_writer.add_figure('figures/load', f, global_step=cur_epoch)
                if cfg.neptune_each_run:
                    fname = f'{loggers[split].name}/load'
                    loggers[split].neptune_writer[fname].append(f, step=cur_epoch)
        
    def eval_task_epoch(self, loggers, cur_epoch, is_best, **kwargs):
        for split in ['eval', 'test']:
            preds = torch.cat(loggers[split]._pred).detach().cpu().numpy()
            trues = torch.cat(loggers[split]._true).detach().cpu().numpy()
            preds_x, preds_y = 8000*preds[:, 0], 1000*preds[:, 1]
            trues_x, trues_y = 8000*trues[:, 0], 1000*trues[:, 1]
            mae_x = np.mean(np.abs(trues_x - preds_x))
            mape_x = np.mean(np.abs((trues_x - preds_x) / (trues_x + 100)))
            rmse_x = np.sqrt(mean_squared_error(trues_x, preds_x))
            mae_y = np.mean(np.abs(trues_y - preds_y))
            mape_y = np.mean(np.abs((trues_y - preds_y) / (trues_y + 10)))
            rmse_y = np.sqrt(mean_squared_error(trues_y, preds_y))
            loggers[split].update_custom_stats(**{
                'mae_fx': float(mae_x),
                'mae_fy': float(mae_y),
                'rmse_fx': float(rmse_x),
                'rmse_fy': float(rmse_y),
                'mape_fx': float(mape_x),
                'mape_fy': float(mape_y),
            })
        self.log_task_figures(loggers, cur_epoch, is_best=is_best)
