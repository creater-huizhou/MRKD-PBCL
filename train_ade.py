import argparse
import os
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

import models.segformer as module_arch
import utils.metric as module_metric
import utils.lr_scheduler as module_lr_scheduler
import data_loader.data_loaders as module_data
from trainer.trainer_ade import Trainer_Segformer_base, Trainer_Segformer_incremental
from utils.parse_config import ConfigParser
from logger.logger import Logger
from utils.memory import (random_sample, class_balanced_sampling, class_balanced_exemplar_selection,
                          codebook_rehearsal_sampling, class_information_maximum_sampling, herding_sampling,
                          sampling_based_on_confidence_score)

torch.backends.cudnn.benchmark = True


def main(config):
    ngpus_per_node = torch.cuda.device_count()
    # print(ngpus_per_node)
    # print(config['multiprocessing_distributed'])
    if config['multiprocessing_distributed']:
        # Single node, mutliple GPUs
        config.config['world_size'] = ngpus_per_node * config['world_size']
        # print('1')
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
    else:
        # print('2')
        # Rather using distributed, use DataParallel
        main_worker(None, ngpus_per_node, config)


def main_worker(gpu, ngpus_per_node, config):
    if config['multiprocessing_distributed']:
        config.config['rank'] = config['rank'] * ngpus_per_node + gpu
    """
    初始化分布式环境：各个进程会在这一步，与master节点进行握手，建立连接,要用的这几张卡就组成一个group。
    backend：pytorch分布式训练通信的后端
    init_method：进行分布式通信的方法，有三种方法：1.共享文件系统，指令为“file://”。 2.IP组播，指令为“tcp://”。
        3.环境变量，指令为“env://” (默认是这个)
    world_size： 整个分布式任务中，并行进程的数量
    rank：在整个分布式任务中进程的序号。rank=0的进程就是master进程。
    """
    """
    dist.init_process_group(
        backend=config['dist_backend'], init_method=config['dist_url'],
        world_size=config['world_size'], rank=config['rank']
    )
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '45637'

    dist.init_process_group(
        backend=config['dist_backend'], init_method=config['environment_value'],
        world_size=config['world_size'], rank=config['rank']
    )

    # Set looging
    rank = dist.get_rank()
    logger = Logger(config.log_dir, rank=rank)
    logger.set_logger(f'train(rank{rank})', verbosity=2)

    # fix random seeds for reproduce
    SEED = config['seed']
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Task information
    task_step = config['data_loader']['args']['task']['step']
    task_name = config['data_loader']['args']['task']['name']
    task_setting = config['data_loader']['args']['task']['setting']

    # Create Dataloader
    dataset = config.init_obj('data_loader', module_data)

    # Create old Model
    if task_step > 0:
        model_old = config.init_obj('arch', module_arch, **{"classes": dataset.get_per_task_classes(task_step - 1)})
    else:
        model_old = None

    # Memory pre-processing
    if task_step > 0 and config['data_loader']['args']['memory']['mem_size'] > 0:
        if task_step > 1:
            old_path = config.save_dir.parent / f"step_{task_step - 1}" / f"checkpoint-epoch{config['trainer']['epochs']}.pth"
        else:
            old_path = config.save_dir.parent / f"step_{task_step - 1}" / "model_best.pth"

        # Load old model
        class_num = config['data_loader']['args']['class_num']
        decoder_dim = config['arch']['args']['decoder_dim']
        prototypes = torch.zeros([class_num + 1, decoder_dim])
        if model_old is not None:
            model_old._load_pretrained_model(f'{old_path}')
            checkpoint = torch.load(old_path, map_location='cpu')
            prototypes = checkpoint['prototype']

        if config['data_loader']['args']['memory']['sample_type'] == 'cbs':
            class_balanced_sampling(
                config,
                model_old,
                dataset.get_old_train_loader(),
                ('voc', task_setting, task_name, task_step),
                logger,
                gpu
            )
        elif config['data_loader']['args']['memory']['sample_type'] == 'cbes':
            class_balanced_exemplar_selection(
                config,
                model_old,
                dataset.get_old_train_loader(),
                ('voc', task_setting, task_name, task_step),
                logger,
                gpu
            )
        elif config['data_loader']['args']['memory']['sample_type'] == 'rs':
            random_sample(
                config,
                model_old,
                dataset.get_old_train_loader(),
                ('voc', task_setting, task_name, task_step),
                logger,
                gpu
            )
        elif config['data_loader']['args']['memory']['sample_type'] == 'ha':
            herding_sampling(
                config,
                model_old,
                prototypes,
                dataset.get_old_train_loader(),
                ('voc', task_setting, task_name, task_step),
                logger,
                gpu
            )
        elif config['data_loader']['args']['memory']['sample_type'] == 'cr':
            codebook_rehearsal_sampling(
                config,
                model_old,
                prototypes,
                dataset.get_old_train_loader(),
                ('voc', task_setting, task_name, task_step),
                logger,
                gpu
            )
        elif config['data_loader']['args']['memory']['sample_type'] == 'cims':
            class_information_maximum_sampling(
                config,
                model_old,
                prototypes,
                dataset.get_old_train_loader(),
                ('voc', task_setting, task_name, task_step),
                logger,
                gpu
            )
        elif config['data_loader']['args']['memory']['sample_type'] == 'sbocs':
            sampling_based_on_confidence_score(
                config,
                model_old,
                prototypes,
                dataset.get_old_train_loader(),
                ('voc', task_setting, task_name, task_step),
                logger,
                gpu
            )

        dataset.get_memory(config, concat=True)
    logger.info(f"{str(dataset)}")
    logger.info(f"{dataset.dataset_info()}")

    if config['multiprocessing_distributed']:
        train_sampler = DistributedSampler(dataset.train_set)
    else:
        train_sampler = None

    train_loader = dataset.get_train_loader(train_sampler)
    val_loader = dataset.get_val_loader()
    test_loader = dataset.get_test_loader()

    new_classes, old_classes = dataset.get_task_labels()
    logger.info(f"Old Classes: {old_classes}")
    logger.info(f"New Classes: {new_classes}")

    # Create Model
    model = config.init_obj('arch', module_arch, **{"classes": dataset.get_per_task_classes()})
    #logger.info(model)

    # Load previous step weights
    if task_step > 0:
        # print(config.save_dir, config.save_dir.parent)
        # old_path = config.save_dir.parent / f"step_{task_step - 1}" / f"checkpoint-epoch{config['trainer']['last_step_best_epoch']}.pth"

        if task_step > 1:
            old_path = config.save_dir.parent / f"step_{task_step - 1}" / f"checkpoint-epoch{config['trainer']['epochs']}.pth"
        else:
            old_path = config.save_dir.parent / f"step_{task_step - 1}" / "model_best.pth"
        model._load_pretrained_model(f'{old_path}')
        logger.info(f"Load weights from a previous step:{old_path}")

        # Load old model to use KD
        if model_old is not None:
            model_old._load_pretrained_model(f'{old_path}')

        if config['hyperparameter']['ac'] > 0:
            logger.info('** Proposed Initialization Technique using an Auxiliary Classifier**')
            model.init_novel_classifier()
        else:
            logger.info('** Random Initialization **')
    else:
        logger.info('Train from scratch')

    # Build optimizer
    if task_step > 0:
        optimizer = config.init_obj(
            'optimizer',
            torch.optim,
            [{"params": model.get_backbone_params(), "weight_decay": 0},
            {"params": model.get_old_classifer_params(), "lr": config["optimizer"]["args"]["lr"] * 10, "weight_decay": 0},
            {"params": model.get_new_classifer_params(), "lr": config["optimizer"]["args"]["lr"] * 10}]
        )
    else:
        optimizer = config.init_obj(
            'optimizer',
            torch.optim,
            [{"params": model.get_backbone_params()},
            {"params": model.get_classifer_params(), "lr": config["optimizer"]["args"]["lr"] * 10}]
        )

    if config['lr_scheduler']['type'] == 'PolyLR':
        lr_scheduler = config.init_obj(
            'lr_scheduler',
            module_lr_scheduler,
            **{"optimizer": optimizer, "max_iters": config["trainer"]['epochs'] * len(train_loader)}
        )
    elif config['lr_scheduler']['type'] == 'WarmupPolyLR':
        lr_scheduler = config.init_obj(
            'lr_scheduler',
            module_lr_scheduler,
            **{"optimizer": optimizer, "max_iters": config["trainer"]['epochs'] * len(train_loader)}
        )
    elif config['lr_scheduler']['type'] == 'ExpLR':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=0.9
        )
    elif config['lr_scheduler']['type'] == 'MultiStepLR':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            # milestones=[40], # epoch 40
            milestones=[30, 40, 50], # joint training/continue epoch 60
            # milestones=[30, 50, 70], # joint training epoch 100
            gamma=0.5
        )
    elif config['lr_scheduler']['type'] == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=10,
            gamma=0.9
        )
    else:
        raise ValueError("No lr_scheduler!!!")

    evaluator_val = config.init_obj(
        'evaluator',
        module_metric,
        *[dataset.n_classes + 1, [0], new_classes]
    )

    old_classes, _ = dataset.get_task_labels(step=0)
    new_classes = []
    for i in range(1, task_step + 1):
        c, _ = dataset.get_task_labels(step=i)
        new_classes += c

    evaluator_test = config.init_obj(
        'evaluator',
        module_metric,
        *[dataset.n_classes + 1, list(set(old_classes + [0])), new_classes]
    )

    if task_step > 0:
        trainer = Trainer_Segformer_incremental(
            model=model.cuda(), model_old=model_old.cuda(),
            optimizer=optimizer, evaluator=(evaluator_val, evaluator_test),
            config=config, task_info=dataset.task_info(),
            data_loader=(train_loader, val_loader, test_loader),
            lr_scheduler=lr_scheduler, logger=logger, gpu=gpu
        )
    else:
        trainer = Trainer_Segformer_base(
            model=model.cuda(), optimizer=optimizer,
            evaluator=(evaluator_val, evaluator_test),
            config=config, task_info=dataset.task_info(),
            data_loader=(train_loader, val_loader, test_loader),
            lr_scheduler=lr_scheduler, logger=logger, gpu=gpu
        )

    torch.distributed.barrier()

    trainer.train()
    trainer.test()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Class incremental Semantic Segmentation')
    args.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type action target', defaults=(None, float, None, None))
    options = [
        CustomArgs(['--multiprocessing_distributed'], action='store_true', target='multiprocessing_distributed'),
        CustomArgs(['--dist_url'], type=str, target='dist_url'),
        CustomArgs(['--name'], type=str, target='name'),
        CustomArgs(['--save_dir'], type=str, target='trainer;save_dir'),
        CustomArgs(['--mem_size'], type=int, target='data_loader;args;memory;mem_size'),
        CustomArgs(['--sample_type'], type=str, target='data_loader;args;memory;sample_type'),
        CustomArgs(['--seed'], type=int, target='seed'),
        CustomArgs(['--ep', '--epochs'], type=int, target='trainer;epochs'),
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;train;batch_size'),
        CustomArgs(['--task_name'], type=str, target='data_loader;args;task;name'),
        CustomArgs(['--task_step'], type=int, target='data_loader;args;task;step'),
        CustomArgs(['--task_setting'], type=str, target='data_loader;args;task;setting'),
        CustomArgs(['--pos_weight'], type=float, target='hyperparameter;pos_weight'),
        CustomArgs(['--mbce'], type=float, target='hyperparameter;mbce'),
        CustomArgs(['--kd'], type=float, target='hyperparameter;kd'),
        CustomArgs(['--dkd_pos'], type=float, target='hyperparameter;dkd_pos'),
        CustomArgs(['--dkd_neg'], type=float, target='hyperparameter;dkd_neg'),
        CustomArgs(['--ac'], type=float, target='hyperparameter;ac'),
        CustomArgs(['--flag_MRKD'], action='store_true', target='trainer;flag_MRKD'),
        CustomArgs(['--flag_PBCL'], action='store_true', target='trainer;flag_PBCL'),
        CustomArgs(['--flag_SDR_CL'], action='store_true', target='trainer;flag_SDR_CL'),
        CustomArgs(['--flag_IDEC_CL'], action='store_true', target='trainer;flag_IDEC_CL'),
        CustomArgs(['--flag_COINSEG_CL'], action='store_true', target='trainer;flag_COINSEG_CL'),
        CustomArgs(['--test'], action='store_true', target='test'),
        CustomArgs(['--monitor'], type=str, target='trainer;monitor'),
        CustomArgs(['--lr_scheduler'], type=str, target='lr_scheduler;type'),
        CustomArgs(['--optimizer'], type=str, target='optimizer;type'),
        CustomArgs(['--freeze'], action='store_true', target='arch;args;freeze'),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)

