import logging
from typing import Optional


from .task import BaseTask
from .train import MappingTrainer
from ..config import cfg

TRAINERS = {
    'mapping': MappingTrainer,
}


def create_trainer(
    model, 
    criterion, 
    task=Optional[BaseTask],
    optimizer=None, 
    scheduler=None, 
    scaler=None, 
    neptune_writer=None,
    task_train_type=None,
):
    task_train_type = cfg.task.train_type
    if task_train_type not in TRAINERS:
        raise ValueError(f"Invalid task train type: '{task_train_type}'.")

    args = dict(
        model=model,
        task=task,
        neptune_writer=neptune_writer,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        scaler=scaler,
        task_train_type=task_train_type,
        n_epochs=cfg.optim.max_epochs,
        clip_grad=cfg.train.clip_grad,
        max_grad_norm=cfg.train.max_grad_norm
    )
    
    trainer = TRAINERS[task_train_type](**args)
    logging.info(f"Create {trainer.__class__.__name__}.")
    return trainer
