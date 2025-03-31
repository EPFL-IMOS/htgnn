"""
Experiment tracking with Neptune
"""
import neptune
import logging

from ..config import cfg
from ..utils.credential import (
    NEPTUNE_PROJECT_NAME_BEARING,
    NEPTUNE_PROJECT_NAME_BRIDGE,
    NEPTUNE_API_TOKEN
)


CASE_MAPPING = {
    'bearing': 'BEA',
    'bridge': 'BRI',
}


def get_project_id():
    if cfg.task.type == 'bearing':
        project_id = NEPTUNE_PROJECT_NAME_BEARING
    elif cfg.task.type == 'bridge':
        project_id = NEPTUNE_PROJECT_NAME_BRIDGE
    else:
        raise NotImplementedError(f'Project id for task type {cfg.task.type} not implemented')
    return project_id


def get_run_id(seed):
    if '/' in cfg.case:
        case_name = cfg.case.split('/')[0]
    else:
        case_name = cfg.case
    if case_name not in ['bearing', 'bridge']:
        raise NotImplementedError(f'Run id for case {cfg.case} not implemented. Case should be "bearing" or "bridge".')
    
    case_name = CASE_MAPPING[case_name]
    fname = cfg.fname.replace('=', '')
    run_id = f'{case_name}-{fname}-{seed}'
    # when the customer id has more than 35 chars, it will not be valid
    if len(run_id) > 35:
        run_id = run_id.replace('-', '')
    return run_id


def create_neptune_run_writer(seed):
    id = get_run_id(seed)
    try:
        neptune_writer = get_neptune_run(id)
        logging.info(f'Found Neptune run with id {id}')
    except Exception as e:
        logging.info(f'Creating new Neptune run with id {id}')
        logging.info(f'Project id: {get_project_id()}')
        neptune_writer = neptune.init_run(
            project=get_project_id(),
            api_token=NEPTUNE_API_TOKEN,
            source_files=['src/'],
            custom_run_id=id
        )  # defined in the .env file

    return neptune_writer


def get_neptune_run(run_id):
    run = neptune.init_run(
        project=get_project_id(),
        api_token=NEPTUNE_API_TOKEN,
        with_id=run_id
    )  # defined in the .env file
    return run


def write_cfg_to_neptune(cfg, neptune_writer):
    for key_dict, value in cfg.items():
        try:
            if isinstance(key_dict, dict):
                for k, v in key_dict.items():
                    if isinstance(v, list):
                        neptune_writer[k].append(v)
                    else:
                        neptune_writer[k] = v
            else:
                neptune_writer[key_dict] = value
        except ValueError as e:
            print(f'Error writing {key_dict} to Neptune: {e}')


def write_agg_results_to_neptune(results, neptune_writer):
    neptune_writer['epoch'] = results['train']['epoch']
    neptune_writer['params'] = results['train']['params']
    for split, split_result in results.items():
        split_result.pop('params', None)
        split_result.pop('params_std', None)
        split_result.pop('epoch', None)
        split_result.pop('epoch_std', None)
        split_result.pop('lr', None)
        split_result.pop('lr_std', None)
        split_result.pop('time_iter', None)
        split_result.pop('time_iter_std', None)
        for key, value in split_result.items():
            try:
                if isinstance(value, float) or isinstance(value, int):
                    neptune_writer[f'{split}/{key}'] = value
                elif isinstance(value, list):
                    neptune_writer[f'{split}/{key}'].append(value)
            except Exception as e:
                print(f'Error writing {key} to Neptune: {e}')


def write_test_results_to_neptune(results, neptune_writer):
    for key, value in results.items():
        try:
            if isinstance(value, float) or isinstance(value, int):
                neptune_writer[f'test/{key}'] = value
            elif isinstance(value, list):
                neptune_writer[f'test/{key}'].append(value)
        except Exception as e:
            print(f'Error writing {key} to Neptune: {e}')