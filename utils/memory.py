"""
We modified the code from SSUL

SSUL
Copyright (c) 2021-present NAVER Corp.
MIT License
"""

import math
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import models.model as module_arch
import utils.metric as module_metric
import utils.lr_scheduler as module_lr_scheduler
import data_loader.data_loaders as module_data
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP

from data_loader.task import get_task_labels, get_per_task_classes
from trainer.trainer_voc import Trainer_Segformer_base


def _prepare_device(n_gpu_use, logger):
    """
    setup GPU device if available, move model into configured device
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        logger.warning("Warning: There\'s no GPU available on this machine,"
                       "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                       "on this machine.".format(n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


# 随机采样
def random_sample(config, model, train_loader, task_info, logger, gpu):
    if gpu is None:
        # setup GPU device if available, move model into configured device
        device, device_ids = _prepare_device(config['n_gpu'], logger)
    else:
        device = gpu
        device_ids = None

    if not torch.cuda.is_available():
        logger.info("using CPU, this will be slow")
    elif config['multiprocessing_distributed']:
        if gpu is not None:
            torch.cuda.set_device(device)
            model.to(device)
            # When using a single GPU per process and per
            # DDP, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            model = DDP(model, device_ids=[gpu])  # Detectron: broadcast_buffers=False
        else:
            model.to(device)
            # DDP will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = DDP(model)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = nn.DataParallel(model, device_ids=device_ids)

    task_dataset, task_setting, task_name, task_step = task_info
    new_classes, old_classes = get_task_labels(task_dataset, task_name, task_step)
    prev_num_classes = len(old_classes)  # 15

    # memory_json = f'./data/{task_dataset}/{task_setting}_{task_name}_memory.json'
    memory_json = config.save_dir.parent / f'step_{task_step}' / 'memory.json'
    memory_size = config['data_loader']['args']['memory']['mem_size']

    if task_step > 1:
        old_memory_json = config.save_dir.parent / f'step_{task_step - 1}' / 'memory.json'
        with open(old_memory_json, "r") as json_file:
            memory_list = json.load(json_file)
        memory_candidates = memory_list[f"step_{task_step - 1}"]["memory_candidates"]
    else:
        memory_list = {}
        memory_candidates = []

    logger.info("...start memory candidates collection")
    torch.distributed.barrier()

    model.eval()
    for batch_idx, data in enumerate(train_loader):
        if task_step > 1:
            with torch.no_grad():
                images, targets, img_names = data['image'].to(device), data['label'].to(device), data['image_name']

                outputs, _, _, _ = model(images, ret_intermediate=False)
                logit = torch.sigmoid(outputs).detach()
                pred_scores, pred_labels = torch.max(logit[:, 1:], dim=1)
                pred_labels += 1

                """ pseudo labeling """
                targets = torch.where((targets == 0) & (pred_scores >= 0.9), pred_labels.long(), targets.long())
        else:
            images, targets, img_names = data['image'].to(device), data['label'].to(device), data['image_name']

        for b in range(images.size(0)):
            img_name = img_names[b]
            target = targets[b]
            labels = torch.unique(target).detach().cpu().numpy().tolist()
            if 0 in labels:
                labels.remove(0)

            memory_candidates.append([img_name, labels])

        if batch_idx % 100 == 0:
            logger.info(f"{batch_idx}/{len(train_loader)}")

    logger.info(f"...end memory candidates collection : {len(memory_candidates)}")

    model.to('cpu')
    torch.cuda.empty_cache()
    ####################################################################################################
    logger.info("...start memory list generation")
    curr_memory_list = {f"class_{cls}": [] for cls in range(1, prev_num_classes + 1)}  # 1~15
    sorted_memory_candidates = memory_candidates.copy()
    np.random.shuffle(sorted_memory_candidates)

    random_class_order = old_classes.copy()
    np.random.shuffle(random_class_order)
    num_sampled = 0

    while memory_size > num_sampled:
        for idx, mem in enumerate(sorted_memory_candidates):
            img_name, labels = mem
            if len(labels) == 0:
                continue
            if labels[0] == 255:
                continue
            curr_memory_list[f"class_{labels[0]}"].append(mem)
            num_sampled += 1

            if memory_size <= num_sampled:
                break

    ######################################
    """ save memory info """
    memory_str = ''
    for i in range(1, prev_num_classes + 1):
        memory_str += f"\nclass_{i}: "
        for j in curr_memory_list[f"class_{i}"]:
            if task_dataset == 'ade':
                memory_str += j[0].split()[0][10:]
            elif task_dataset == 'voc':
                memory_str += j[0].split()[0][12:]
                print(j[0], j[0].split()[0], j[0].split()[0][12:])
            else:
                raise NotImplementedError
            memory_str += ' '
    logger.info(memory_str)

    sampled_memory_list = [mem for mem_cls in curr_memory_list.values() for mem in mem_cls]  # gather all memory

    memory_list[f"step_{task_step}"] = {
        "memory_candidates": sampled_memory_list,
        "memory_list": sorted([mem[0] for mem in sampled_memory_list])
    }

    if torch.distributed.get_rank() == 0:
        with open(memory_json, "w") as json_file:
            json.dump(memory_list, json_file)

    torch.distributed.barrier()


def class_balanced_sampling(config, model, train_loader, task_info, logger, gpu):
    if gpu is None:
        # setup GPU device if available, move model into configured device
        device, device_ids = _prepare_device(config['n_gpu'], logger)
    else:
        device = gpu
        device_ids = None

    if not torch.cuda.is_available():
        logger.info("using CPU, this will be slow")
    elif config['multiprocessing_distributed']:
        if gpu is not None:
            torch.cuda.set_device(device)
            model.cuda()
            # When using a single GPU per process and per
            # DDP, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            model = DDP(model, device_ids=[gpu])  # Detectron: broadcast_buffers=False
        else:
            model.cuda()
            # DDP will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = DDP(model)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = nn.DataParallel(model, device_ids=device_ids)

    task_dataset, task_setting, task_name, task_step = task_info
    new_classes, old_classes = get_task_labels(task_dataset, task_name, task_step)
    prev_num_classes = len(old_classes)  # 15

    # memory_json = f'./data/{task_dataset}/{task_setting}_{task_name}_memory.json'
    memory_json = config.save_dir.parent / f'step_{task_step}' / 'memory.json'
    memory_size = config['data_loader']['args']['memory']['mem_size']

    if task_step > 1:
        old_memory_json = config.save_dir.parent / f'step_{task_step - 1}' / 'memory.json'
        with open(old_memory_json, "r") as json_file:
            memory_list = json.load(json_file)
        memory_candidates = memory_list[f"step_{task_step - 1}"]["memory_candidates"]
    else:
        memory_list = {}
        memory_candidates = []

    logger.info("...start memory candidates collection")
    torch.distributed.barrier()
    
    model.eval()
    for batch_idx, data in enumerate(train_loader):
        if task_step > 1:
            with torch.no_grad():
                images, targets, img_names = data['image'].to(device), data['label'].to(device), data['image_name']

                outputs, _, _, _ = model(images, ret_intermediate=False)
                logit = torch.sigmoid(outputs).detach()
                pred_scores, pred_labels = torch.max(logit[:, 1:], dim=1)
                pred_labels += 1
                
                """ pseudo labeling """
                targets = torch.where((targets == 0) & (pred_scores >= 0.9), pred_labels.long(), targets.long())
        else:
            images, targets, img_names = data['image'].to(device), data['label'].to(device), data['image_name']

        for b in range(images.size(0)):
            img_name = img_names[b]
            target = targets[b]
            labels = torch.unique(target).detach().cpu().numpy().tolist()
            if 0 in labels:
                labels.remove(0)
            
            memory_candidates.append([img_name, labels])

        if batch_idx % 100 == 0:
            logger.info(f"{batch_idx}/{len(train_loader)}")

    logger.info(f"...end memory candidates collection : {len(memory_candidates)}")

    model.to('cpu')
    torch.cuda.empty_cache()
    ####################################################################################################
    logger.info("...start memory list generation")
    curr_memory_list = {f"class_{cls}": [] for cls in range(1, prev_num_classes + 1)}  # 1~15
    sorted_memory_candidates = memory_candidates.copy()
    np.random.shuffle(sorted_memory_candidates)
    
    random_class_order = old_classes.copy()
    np.random.shuffle(random_class_order)
    num_sampled = 0
    
    while memory_size > num_sampled:
        for cls in random_class_order:
            for idx, mem in enumerate(sorted_memory_candidates):
                img_name, labels = mem

                if cls in labels:
                    curr_memory_list[f"class_{cls}"].append(mem)
                    num_sampled += 1
                    del sorted_memory_candidates[idx]
                    break
                    
            if memory_size <= num_sampled:
                break
        
    ######################################
    """ save memory info """
    memory_str = ''
    for i in range(1, prev_num_classes + 1):
        memory_str += f"\nclass_{i}: "
        for j in curr_memory_list[f"class_{i}"]:
            if task_dataset == 'ade':
                memory_str += j[0].split()[0][10:]
            elif task_dataset == 'voc':
                memory_str += j[0].split()[0][12:]
                print(j[0], j[0].split()[0], j[0].split()[0][12:])
            else:
                raise NotImplementedError
            memory_str += ' '
    logger.info(memory_str)

    sampled_memory_list = [mem for mem_cls in curr_memory_list.values() for mem in mem_cls]  # gather all memory
    
    memory_list[f"step_{task_step}"] = {
        "memory_candidates": sampled_memory_list,
        "memory_list": sorted([mem[0] for mem in sampled_memory_list])
    }
    
    if torch.distributed.get_rank() == 0:
        with open(memory_json, "w") as json_file:
            json.dump(memory_list, json_file)

    torch.distributed.barrier()


# 保证每个类别的数量相等
def class_balanced_exemplar_selection(config, model, train_loader, task_info, logger, gpu):
    if gpu is None:
        # setup GPU device if available, move model into configured device
        device, device_ids = _prepare_device(config['n_gpu'], logger)
    else:
        device = gpu
        device_ids = None

    if not torch.cuda.is_available():
        logger.info("using CPU, this will be slow")
    elif config['multiprocessing_distributed']:
        if gpu is not None:
            torch.cuda.set_device(device)
            model.cuda()
            # When using a single GPU per process and per
            # DDP, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            model = DDP(model, device_ids=[gpu])  # Detectron: broadcast_buffers=False
        else:
            model.cuda()
            # DDP will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = DDP(model)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = nn.DataParallel(model, device_ids=device_ids)

    task_dataset, task_setting, task_name, task_step = task_info
    new_classes, old_classes = get_task_labels(task_dataset, task_name, task_step)
    prev_num_classes = len(old_classes)  # 15

    # memory_json = f'./data/{task_dataset}/{task_setting}_{task_name}_memory.json'
    memory_json = config.save_dir.parent / f'step_{task_step}' / 'memory.json'
    memory_size = config['data_loader']['args']['memory']['mem_size']

    if task_step > 1:
        old_memory_json = config.save_dir.parent / f'step_{task_step - 1}' / 'memory.json'
        with open(old_memory_json, "r") as json_file:
            memory_list = json.load(json_file)
        memory_candidates = memory_list[f"step_{task_step - 1}"]["memory_candidates"]
    else:
        memory_list = {}
        memory_candidates = []

    logger.info("...start memory candidates collection")
    torch.distributed.barrier()

    model.eval()
    for batch_idx, data in enumerate(train_loader):
        if task_step > 1:
            with torch.no_grad():
                images, targets, img_names = data['image'].to(device), data['label'].to(device), data['image_name']

                outputs, _, _, _ = model(images, ret_intermediate=False)
                logit = torch.sigmoid(outputs).detach()
                pred_scores, pred_labels = torch.max(logit[:, 1:], dim=1)
                pred_labels += 1

                """ pseudo labeling """
                targets = torch.where((targets == 0) & (pred_scores >= 0.9), pred_labels.long(), targets.long())
        else:
            images, targets, img_names = data['image'].to(device), data['label'].to(device), data['image_name']

        for b in range(images.size(0)):
            img_name = img_names[b]
            target = targets[b]
            labels = torch.unique(target).detach().cpu().numpy().tolist()
            if 0 in labels:
                labels.remove(0)

            memory_candidates.append([img_name, labels])

        if batch_idx % 100 == 0:
            logger.info(f"{batch_idx}/{len(train_loader)}")

    logger.info(f"...end memory candidates collection : {len(memory_candidates)}")

    model.to('cpu')
    torch.cuda.empty_cache()
    ####################################################################################################
    logger.info("...start memory list generation")
    curr_memory_list = {f"class_{cls}": [] for cls in range(1, prev_num_classes + 1)}  # 1~15
    sorted_memory_candidates = memory_candidates.copy()
    np.random.shuffle(sorted_memory_candidates)

    random_class_order = old_classes.copy()
    np.random.shuffle(random_class_order)
    class_max_num_sampled = memory_size // len(old_classes)

    for cls in random_class_order:
        num_sampled = 0
        for idx, mem in enumerate(sorted_memory_candidates):
            img_name, labels = mem
            if cls in labels:
                curr_memory_list[f"class_{cls}"].append(mem)
                num_sampled += 1
                del sorted_memory_candidates[idx]

            if num_sampled >= class_max_num_sampled:
                break
    ######################################
    """ save memory info """
    memory_str = ''
    for i in range(1, prev_num_classes + 1):
        memory_str += f"\nclass_{i}: "
        for j in curr_memory_list[f"class_{i}"]:
            if task_dataset == 'ade':
                memory_str += j[0].split()[0][10:]
            elif task_dataset == 'voc':
                memory_str += j[0].split()[0][12:]
                print(j[0], j[0].split()[0], j[0].split()[0][12:])
            else:
                raise NotImplementedError
            memory_str += ' '
    logger.info(memory_str)

    sampled_memory_list = [mem for mem_cls in curr_memory_list.values() for mem in mem_cls]  # gather all memory

    memory_list[f"step_{task_step}"] = {
        "memory_candidates": sampled_memory_list,
        "memory_list": sorted([mem[0] for mem in sampled_memory_list])
    }

    if torch.distributed.get_rank() == 0:
        with open(memory_json, "w") as json_file:
            json.dump(memory_list, json_file)

    torch.distributed.barrier()


# 保存距离类别中心最近的K个样本
def herding_sampling(config, model, prototypes, train_loader, task_info, logger, gpu):
    if gpu is None:
        # setup GPU device if available, move model into configured device
        device, device_ids = _prepare_device(config['n_gpu'], logger)
    else:
        device = gpu
        device_ids = None

    if not torch.cuda.is_available():
        logger.info("using CPU, this will be slow")
    elif config['multiprocessing_distributed']:
        if gpu is not None:
            torch.cuda.set_device(device)
            model.cuda()
            # When using a single GPU per process and per
            # DDP, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            model = DDP(model, device_ids=[gpu])  # Detectron: broadcast_buffers=False
        else:
            model.cuda()
            # DDP will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = DDP(model)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = nn.DataParallel(model, device_ids=device_ids)

    task_dataset, task_setting, task_name, task_step = task_info
    new_classes, old_classes = get_task_labels(task_dataset, task_name, task_step)
    prev_num_classes = len(old_classes)  # 15
    now_num_classes = len(new_classes)

    # memory_json = f'./data/{task_dataset}/{task_setting}_{task_name}_memory.json'
    memory_json = config.save_dir.parent / f'step_{task_step}' / 'memory.json'
    memory_size = config['data_loader']['args']['memory']['mem_size']

    if task_step > 1:
        old_memory_json = config.save_dir.parent / f'step_{task_step - 1}' / 'memory.json'
        with open(old_memory_json, "r") as json_file:
            memory_list = json.load(json_file)
        memory_candidates = memory_list[f"step_{task_step - 1}"]["memory_candidates"]
        img_similarity_dict = memory_list[f"step_{task_step - 1}"]["img_similarity"]
        img_similarity_dict_new = {f"class_{cls}": {} for cls in range(prev_num_classes, prev_num_classes + now_num_classes)}  # 16
    else:
        memory_list = {}
        memory_candidates = []
        img_similarity_dict = {f"class_{cls}": {} for cls in range(1, prev_num_classes + 1)}  # 1~15

    logger.info("...start memory candidates collection")
    torch.distributed.barrier()

    model.eval()
    for batch_idx, data in enumerate(train_loader):
        if task_step > 1:
            with torch.no_grad():
                prototypes = prototypes.to(device)
                images, targets, img_names = data['image'].to(device), data['label'].to(device), data['image_name']
                outputs, features, _, _ = model(images, ret_intermediate=False)
                logit = torch.sigmoid(outputs).detach()
                pred_scores, pred_labels = torch.max(logit[:, 1:], dim=1)
                pred_labels += 1

                """ pseudo labeling """
                targets = torch.where((targets == 0) & (pred_scores >= 0.9), pred_labels.long(), targets.long())

                B, C, H, W = features.shape[0], features.shape[1], features.shape[2], features.shape[3]
                labels = data['label'].unsqueeze(dim=1).to(device)
                labels_down = (F.interpolate(input=labels.double(), size=(H, W), mode='nearest')).long()

                for bs in range(B):
                    cl_present = torch.unique(input=labels_down[bs]).long()
                    if cl_present[0] == 0:
                        cl_present = cl_present[1:]
                    if len(cl_present) == 0:
                        continue
                    if cl_present[-1] == 255:
                        cl_present = cl_present[:-1]
                    if len(cl_present) == 0:
                        continue

                    for cl in cl_present:
                        # [dim, k]
                        features_cl = features[bs][(labels_down[bs] == cl).expand(C, -1, -1)].view(C, -1).detach()
                        # [dim]
                        features_cl = F.normalize(torch.mean(features_cl, dim=-1).float(), p=2, dim=0)
                        cl_prototypes = F.normalize(prototypes[int(cl)], p=2, dim=0)
                        similarity = torch.dot(features_cl, cl_prototypes)
                        key_name = 'class_' + str(cl.item())
                        img_name = data['image_name'][bs]
                        img_similarity_dict_new[key_name].update({img_name: similarity.item()})

            img_similarity_dict.update(img_similarity_dict_new)
        else:
            with torch.no_grad():
                prototypes = prototypes.to(device)
                images, targets, img_names = data['image'].to(device), data['label'].to(device), data['image_name']
                outputs, features, _, _ = model(images, ret_intermediate=False)

                B, C, H, W = features.shape[0], features.shape[1], features.shape[2], features.shape[3]
                labels = data['label'].unsqueeze(dim=1).to(device)
                labels_down = (F.interpolate(input=labels.double(), size=(H, W), mode='nearest')).long()

                for bs in range(B):
                    cl_present = torch.unique(input=labels_down[bs]).long()
                    if cl_present[0] == 0:
                        cl_present = cl_present[1:]
                    if len(cl_present) == 0:
                        continue
                    if cl_present[-1] == 255:
                        cl_present = cl_present[:-1]
                    if len(cl_present) == 0:
                        continue

                    for cl in cl_present:
                        # [dim, k]
                        features_cl = features[bs][(labels_down[bs] == cl).expand(C, -1, -1)].view(C, -1).detach()
                        # [dim]
                        features_cl = F.normalize(torch.mean(features_cl, dim=-1).float(), p=2, dim=0)
                        cl_prototypes = F.normalize(prototypes[int(cl)], p=2, dim=0)
                        similarity = torch.dot(features_cl, cl_prototypes)
                        key_name = 'class_' + str(cl.item())
                        img_name = data['image_name'][bs]
                        img_similarity_dict[key_name].update({img_name: similarity.item()})

            images, targets, img_names = data['image'].to(device), data['label'].to(device), data['image_name']

        for b in range(images.size(0)):
            img_name = img_names[b]
            target = targets[b]
            labels = torch.unique(target).detach().cpu().numpy().tolist()
            if 0 in labels:
                labels.remove(0)

            memory_candidates.append([img_name, labels])

        if batch_idx % 100 == 0:
            logger.info(f"{batch_idx}/{len(train_loader)}")

    logger.info(f"...end memory candidates collection : {len(memory_candidates)}")

    model.to('cpu')
    torch.cuda.empty_cache()
    ####################################################################################################
    logger.info("...start memory list generation")
    curr_memory_list = {f"class_{cls}": [] for cls in range(1, prev_num_classes + 1)}  # 1~15
    sorted_memory_candidates = memory_candidates.copy()
    np.random.shuffle(sorted_memory_candidates)

    class_max_num_sampled = memory_size // len(old_classes)

    for cls in old_classes:
        class_name = 'class_' + str(cls)
        dict = img_similarity_dict[class_name]
        sorted_list = sorted(dict.items(), key=lambda x: x[1], reverse=True)
        for i in range(class_max_num_sampled):
            curr_memory_list[class_name].append([sorted_list[i][0], cls])

    ######################################
    """ save memory info """
    memory_str = ''
    for i in range(1, prev_num_classes + 1):
        memory_str += f"\nclass_{i}: "
        for j in curr_memory_list[f"class_{i}"]:
            if task_dataset == 'ade':
                memory_str += j[0].split()[0][10:]
            elif task_dataset == 'voc':
                memory_str += j[0].split()[0][12:]
                print(j[0], j[0].split()[0], j[0].split()[0][12:])
            else:
                raise NotImplementedError
            memory_str += ' '
    logger.info(memory_str)

    sampled_memory_list = [mem for mem_cls in curr_memory_list.values() for mem in mem_cls]  # gather all memory

    memory_list[f"step_{task_step}"] = {
        "memory_candidates": sampled_memory_list,
        "memory_list": sorted([mem[0] for mem in sampled_memory_list]),
        "img_similarity": img_similarity_dict
    }

    if torch.distributed.get_rank() == 0:
        with open(memory_json, "w") as json_file:
            json.dump(memory_list, json_file)

    torch.distributed.barrier()


# 保存距离类别中心最远的K个样本
def codebook_rehearsal_sampling(config, model, prototypes, train_loader, task_info, logger, gpu):
    if gpu is None:
        # setup GPU device if available, move model into configured device
        device, device_ids = _prepare_device(config['n_gpu'], logger)
    else:
        device = gpu
        device_ids = None

    if not torch.cuda.is_available():
        logger.info("using CPU, this will be slow")
    elif config['multiprocessing_distributed']:
        if gpu is not None:
            torch.cuda.set_device(device)
            model.cuda()
            # When using a single GPU per process and per
            # DDP, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            model = DDP(model, device_ids=[gpu])  # Detectron: broadcast_buffers=False
        else:
            model.cuda()
            # DDP will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = DDP(model)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = nn.DataParallel(model, device_ids=device_ids)

    task_dataset, task_setting, task_name, task_step = task_info
    new_classes, old_classes = get_task_labels(task_dataset, task_name, task_step)
    prev_num_classes = len(old_classes)  # 15
    now_num_classes = len(new_classes)

    # memory_json = f'./data/{task_dataset}/{task_setting}_{task_name}_memory.json'
    memory_json = config.save_dir.parent / f'step_{task_step}' / 'memory.json'
    memory_size = config['data_loader']['args']['memory']['mem_size']

    if task_step > 1:
        old_memory_json = config.save_dir.parent / f'step_{task_step - 1}' / 'memory.json'
        with open(old_memory_json, "r") as json_file:
            memory_list = json.load(json_file)
        memory_candidates = memory_list[f"step_{task_step - 1}"]["memory_candidates"]
        img_similarity_dict = memory_list[f"step_{task_step - 1}"]["img_similarity"]
        img_similarity_dict_new = {f"class_{cls}": {} for cls in range(prev_num_classes, prev_num_classes + now_num_classes)}  # 16
    else:
        memory_list = {}
        memory_candidates = []
        img_similarity_dict = {f"class_{cls}": {} for cls in range(1, prev_num_classes + 1)}  # 1~15

    logger.info("...start memory candidates collection")
    torch.distributed.barrier()

    model.eval()
    for batch_idx, data in enumerate(train_loader):
        if task_step > 1:
            with torch.no_grad():
                prototypes = prototypes.to(device)
                images, targets, img_names = data['image'].to(device), data['label'].to(device), data['image_name']
                outputs, features, _, _ = model(images, ret_intermediate=False)
                logit = torch.sigmoid(outputs).detach()
                pred_scores, pred_labels = torch.max(logit[:, 1:], dim=1)
                pred_labels += 1

                """ pseudo labeling """
                targets = torch.where((targets == 0) & (pred_scores >= 0.9), pred_labels.long(), targets.long())

                B, C, H, W = features.shape[0], features.shape[1], features.shape[2], features.shape[3]
                labels = data['label'].unsqueeze(dim=1).to(device)
                labels_down = (F.interpolate(input=labels.double(), size=(H, W), mode='nearest')).long()

                for bs in range(B):
                    cl_present = torch.unique(input=labels_down[bs]).long()
                    if cl_present[0] == 0:
                        cl_present = cl_present[1:]
                    if len(cl_present) == 0:
                        continue
                    if cl_present[-1] == 255:
                        cl_present = cl_present[:-1]
                    if len(cl_present) == 0:
                        continue

                    for cl in cl_present:
                        # [dim, k]
                        features_cl = features[bs][(labels_down[bs] == cl).expand(C, -1, -1)].view(C, -1).detach()
                        # [dim]
                        features_cl = F.normalize(torch.mean(features_cl, dim=-1).float(), p=2, dim=0)
                        cl_prototypes = F.normalize(prototypes[int(cl)], p=2, dim=0)
                        similarity = torch.dot(features_cl, cl_prototypes)
                        key_name = 'class_' + str(cl.item())
                        img_name = data['image_name'][bs]
                        img_similarity_dict_new[key_name].update({img_name: similarity.item()})

            img_similarity_dict.update(img_similarity_dict_new)
        else:
            with torch.no_grad():
                prototypes = prototypes.to(device)
                images, targets, img_names = data['image'].to(device), data['label'].to(device), data['image_name']
                outputs, features, _, _ = model(images, ret_intermediate=False)

                B, C, H, W = features.shape[0], features.shape[1], features.shape[2], features.shape[3]
                labels = data['label'].unsqueeze(dim=1).to(device)
                labels_down = (F.interpolate(input=labels.double(), size=(H, W), mode='nearest')).long()

                for bs in range(B):
                    cl_present = torch.unique(input=labels_down[bs]).long()
                    if cl_present[0] == 0:
                        cl_present = cl_present[1:]
                    if len(cl_present) == 0:
                        continue
                    if cl_present[-1] == 255:
                        cl_present = cl_present[:-1]
                    if len(cl_present) == 0:
                        continue

                    for cl in cl_present:
                        # [dim, k]
                        features_cl = features[bs][(labels_down[bs] == cl).expand(C, -1, -1)].view(C, -1).detach()
                        # [dim]
                        features_cl = F.normalize(torch.mean(features_cl, dim=-1).float(), p=2, dim=0)
                        cl_prototypes = F.normalize(prototypes[int(cl)], p=2, dim=0)
                        similarity = torch.dot(features_cl, cl_prototypes)
                        key_name = 'class_' + str(cl.item())
                        img_name = data['image_name'][bs]
                        img_similarity_dict[key_name].update({img_name: similarity.item()})

            images, targets, img_names = data['image'].to(device), data['label'].to(device), data['image_name']

        for b in range(images.size(0)):
            img_name = img_names[b]
            target = targets[b]
            labels = torch.unique(target).detach().cpu().numpy().tolist()
            if 0 in labels:
                labels.remove(0)

            memory_candidates.append([img_name, labels])

        if batch_idx % 100 == 0:
            logger.info(f"{batch_idx}/{len(train_loader)}")

    logger.info(f"...end memory candidates collection : {len(memory_candidates)}")

    model.to('cpu')
    torch.cuda.empty_cache()
    ####################################################################################################
    logger.info("...start memory list generation")
    curr_memory_list = {f"class_{cls}": [] for cls in range(1, prev_num_classes + 1)}  # 1~15
    sorted_memory_candidates = memory_candidates.copy()
    np.random.shuffle(sorted_memory_candidates)

    class_max_num_sampled = memory_size // len(old_classes)

    for cls in old_classes:
        class_name = 'class_' + str(cls)
        dict = img_similarity_dict[class_name]
        sorted_list = sorted(dict.items(), key=lambda x: x[1])
        for i in range(class_max_num_sampled):
            curr_memory_list[class_name].append([sorted_list[i][0], cls])
    ######################################
    """ save memory info """
    memory_str = ''
    for i in range(1, prev_num_classes + 1):
        memory_str += f"\nclass_{i}: "
        for j in curr_memory_list[f"class_{i}"]:
            if task_dataset == 'ade':
                memory_str += j[0].split()[0][10:]
            elif task_dataset == 'voc':
                memory_str += j[0].split()[0][12:]
                print(j[0], j[0].split()[0], j[0].split()[0][12:])
            else:
                raise NotImplementedError
            memory_str += ' '
    logger.info(memory_str)

    sampled_memory_list = [mem for mem_cls in curr_memory_list.values() for mem in mem_cls]  # gather all memory

    memory_list[f"step_{task_step}"] = {
        "memory_candidates": sampled_memory_list,
        "memory_list": sorted([mem[0] for mem in sampled_memory_list]),
        "img_similarity": img_similarity_dict
    }

    if torch.distributed.get_rank() == 0:
        with open(memory_json, "w") as json_file:
            json.dump(memory_list, json_file)

    torch.distributed.barrier()


# 保存距离类别中心最近的K//2个样本和距离类别中心最远的K-K//2个样本
def class_information_maximum_sampling_1(config, model, prototypes, train_loader, task_info, logger, gpu):
    if gpu is None:
        # setup GPU device if available, move model into configured device
        device, device_ids = _prepare_device(config['n_gpu'], logger)
    else:
        device = gpu
        device_ids = None

    if not torch.cuda.is_available():
        logger.info("using CPU, this will be slow")
    elif config['multiprocessing_distributed']:
        if gpu is not None:
            torch.cuda.set_device(device)
            model.cuda()
            # When using a single GPU per process and per
            # DDP, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            model = DDP(model, device_ids=[gpu])  # Detectron: broadcast_buffers=False
        else:
            model.cuda()
            # DDP will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = DDP(model)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = nn.DataParallel(model, device_ids=device_ids)

    task_dataset, task_setting, task_name, task_step = task_info
    new_classes, old_classes = get_task_labels(task_dataset, task_name, task_step)
    prev_num_classes = len(old_classes)  # 15
    now_num_classes = len(new_classes)

    # memory_json = f'./data/{task_dataset}/{task_setting}_{task_name}_memory.json'
    memory_json = config.save_dir.parent / f'step_{task_step}' / 'memory.json'
    memory_size = config['data_loader']['args']['memory']['mem_size']

    if task_step > 1:
        old_memory_json = config.save_dir.parent / f'step_{task_step - 1}' / 'memory.json'
        with open(old_memory_json, "r") as json_file:
            memory_list = json.load(json_file)
        memory_candidates = memory_list[f"step_{task_step - 1}"]["memory_candidates"]
        img_similarity_dict = memory_list[f"step_{task_step - 1}"]["img_similarity"]
        img_similarity_dict_new = {f"class_{cls}": {} for cls in range(prev_num_classes, prev_num_classes + now_num_classes)}  # 16
    else:
        memory_list = {}
        memory_candidates = []
        img_similarity_dict = {f"class_{cls}": {} for cls in range(1, prev_num_classes + 1)}  # 1~15
    logger.info("...start memory candidates collection")
    torch.distributed.barrier()

    model.eval()
    for batch_idx, data in enumerate(train_loader):
        if task_step > 1:
            with torch.no_grad():
                prototypes = prototypes.to(device)
                images, targets, img_names = data['image'].to(device), data['label'].to(device), data['image_name']
                outputs, features, _, _ = model(images, ret_intermediate=False)
                logit = torch.sigmoid(outputs).detach()
                pred_scores, pred_labels = torch.max(logit[:, 1:], dim=1)
                pred_labels += 1

                """ pseudo labeling """
                targets = torch.where((targets == 0) & (pred_scores >= 0.9), pred_labels.long(), targets.long())

                B, C, H, W = features.shape[0], features.shape[1], features.shape[2], features.shape[3]
                labels = data['label'].unsqueeze(dim=1).to(device)
                labels_down = (F.interpolate(input=labels.double(), size=(H, W), mode='nearest')).long()

                for bs in range(B):
                    cl_present = torch.unique(input=labels_down[bs]).long()
                    if cl_present[0] == 0:
                        cl_present = cl_present[1:]
                    if len(cl_present) == 0:
                        continue
                    if cl_present[-1] == 255:
                        cl_present = cl_present[:-1]
                    if len(cl_present) == 0:
                        continue

                    for cl in cl_present:
                        # [dim, k]
                        features_cl = features[bs][(labels_down[bs] == cl).expand(C, -1, -1)].view(C, -1).detach()
                        # [dim]
                        features_cl = F.normalize(torch.mean(features_cl, dim=-1).float(), p=2, dim=0)
                        cl_prototypes = F.normalize(prototypes[int(cl)], p=2, dim=0)
                        similarity = torch.dot(features_cl, cl_prototypes)
                        key_name = 'class_' + str(cl.item())
                        img_name = data['image_name'][bs]
                        img_similarity_dict_new[key_name].update({img_name: similarity.item()})

            img_similarity_dict.update(img_similarity_dict_new)
        else:
            with torch.no_grad():
                prototypes = prototypes.to(device)
                images, targets, img_names = data['image'].to(device), data['label'].to(device), data['image_name']
                outputs, features, _, _ = model(images, ret_intermediate=False)

                B, C, H, W = features.shape[0], features.shape[1], features.shape[2], features.shape[3]
                labels = data['label'].unsqueeze(dim=1).to(device)
                labels_down = (F.interpolate(input=labels.double(), size=(H, W), mode='nearest')).long()

                for bs in range(B):
                    cl_present = torch.unique(input=labels_down[bs]).long()
                    if cl_present[0] == 0:
                        cl_present = cl_present[1:]
                    if len(cl_present) == 0:
                        continue
                    if cl_present[-1] == 255:
                        cl_present = cl_present[:-1]
                    if len(cl_present) == 0:
                        continue

                    for cl in cl_present:
                        # [dim, k]
                        features_cl = features[bs][(labels_down[bs] == cl).expand(C, -1, -1)].view(C, -1).detach()
                        # [dim]
                        features_cl = F.normalize(torch.mean(features_cl, dim=-1).float(), p=2, dim=0)
                        cl_prototypes = F.normalize(prototypes[int(cl)], p=2, dim=0)
                        similarity = torch.dot(features_cl, cl_prototypes)
                        key_name = 'class_' + str(cl.item())
                        img_name = data['image_name'][bs]
                        img_similarity_dict[key_name].update({img_name: similarity.item()})

            images, targets, img_names = data['image'].to(device), data['label'].to(device), data['image_name']

        for b in range(images.size(0)):
            img_name = img_names[b]
            target = targets[b]
            labels = torch.unique(target).detach().cpu().numpy().tolist()
            if 0 in labels:
                labels.remove(0)

            memory_candidates.append([img_name, labels])

        if batch_idx % 100 == 0:
            logger.info(f"{batch_idx}/{len(train_loader)}")

    logger.info(f"...end memory candidates collection : {len(memory_candidates)}")

    model.to('cpu')
    torch.cuda.empty_cache()
    ####################################################################################################
    logger.info("...start memory list generation")
    curr_memory_list = {f"class_{cls}": [] for cls in range(1, prev_num_classes + 1)}  # 1~15
    sorted_memory_candidates = memory_candidates.copy()
    np.random.shuffle(sorted_memory_candidates)

    class_max_num_sampled = memory_size // len(old_classes)
    near = class_max_num_sampled // 2

    for cls in old_classes:
        class_name = 'class_' + str(cls)
        dict = img_similarity_dict[class_name]
        sorted_list = sorted(dict.items(), key=lambda x: x[1])
        for i in range(class_max_num_sampled - near):
            curr_memory_list[class_name].append([sorted_list[i][0], cls])
        sorted_list = sorted(dict.items(), key=lambda x: x[1], reverse=True)
        for j in range(near):
            curr_memory_list[class_name].append([sorted_list[j][0], cls])
    ######################################
    """ save memory info """
    memory_str = ''
    for i in range(1, prev_num_classes + 1):
        memory_str += f"\nclass_{i}: "
        for j in curr_memory_list[f"class_{i}"]:
            if task_dataset == 'ade':
                memory_str += j[0].split()[0][10:]
            elif task_dataset == 'voc':
                memory_str += j[0].split()[0][12:]
                print(j[0], j[0].split()[0], j[0].split()[0][12:])
            else:
                raise NotImplementedError
            memory_str += ' '
    logger.info(memory_str)

    sampled_memory_list = [mem for mem_cls in curr_memory_list.values() for mem in mem_cls]  # gather all memory

    memory_list[f"step_{task_step}"] = {
        "memory_candidates": sampled_memory_list,
        "memory_list": sorted([mem[0] for mem in sampled_memory_list]),
        "img_similarity": img_similarity_dict
    }

    if torch.distributed.get_rank() == 0:
        with open(memory_json, "w") as json_file:
            json.dump(memory_list, json_file)

    torch.distributed.barrier()


# 保存距离类别中心最近的K//2个样本和距离类别中心最远的K-K//2个样本，若还有多余的空间，则随机均匀采样
def class_information_maximum_sampling_2(config, model, prototypes, train_loader, task_info, logger, gpu):
    if gpu is None:
        # setup GPU device if available, move model into configured device
        device, device_ids = _prepare_device(config['n_gpu'], logger)
    else:
        device = gpu
        device_ids = None

    if not torch.cuda.is_available():
        logger.info("using CPU, this will be slow")
    elif config['multiprocessing_distributed']:
        if gpu is not None:
            torch.cuda.set_device(device)
            model.cuda()
            # When using a single GPU per process and per
            # DDP, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            model = DDP(model, device_ids=[gpu])  # Detectron: broadcast_buffers=False
        else:
            model.cuda()
            # DDP will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = DDP(model)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = nn.DataParallel(model, device_ids=device_ids)

    task_dataset, task_setting, task_name, task_step = task_info
    new_classes, old_classes = get_task_labels(task_dataset, task_name, task_step)
    prev_num_classes = len(old_classes)  # 15
    now_num_classes = len(new_classes)

    # memory_json = f'./data/{task_dataset}/{task_setting}_{task_name}_memory.json'
    memory_json = config.save_dir.parent / f'step_{task_step}' / 'memory.json'
    memory_size = config['data_loader']['args']['memory']['mem_size']

    if task_step > 1:
        old_memory_json = config.save_dir.parent / f'step_{task_step - 1}' / 'memory.json'
        with open(old_memory_json, "r") as json_file:
            memory_list = json.load(json_file)
        memory_candidates = memory_list[f"step_{task_step - 1}"]["memory_candidates"]
        img_similarity_dict = memory_list[f"step_{task_step - 1}"]["img_similarity"]
        img_similarity_dict_new = {f"class_{cls}": {} for cls in range(prev_num_classes, prev_num_classes + now_num_classes)}  # 16
    else:
        memory_list = {}
        memory_candidates = []
        img_similarity_dict = {f"class_{cls}": {} for cls in range(1, prev_num_classes + 1)}  # 1~15
    logger.info("...start memory candidates collection")
    torch.distributed.barrier()

    model.eval()
    for batch_idx, data in enumerate(train_loader):
        if task_step > 1:
            with torch.no_grad():
                prototypes = prototypes.to(device)
                images, targets, img_names = data['image'].to(device), data['label'].to(device), data['image_name']
                outputs, features, _, _ = model(images, ret_intermediate=False)
                logit = torch.sigmoid(outputs).detach()
                pred_scores, pred_labels = torch.max(logit[:, 1:], dim=1)
                pred_labels += 1

                """ pseudo labeling """
                targets = torch.where((targets == 0) & (pred_scores >= 0.9), pred_labels.long(), targets.long())

                B, C, H, W = features.shape[0], features.shape[1], features.shape[2], features.shape[3]
                labels = data['label'].unsqueeze(dim=1).to(device)
                labels_down = (F.interpolate(input=labels.double(), size=(H, W), mode='nearest')).long()

                for bs in range(B):
                    cl_present = torch.unique(input=labels_down[bs]).long()
                    if cl_present[0] == 0:
                        cl_present = cl_present[1:]
                    if len(cl_present) == 0:
                        continue
                    if cl_present[-1] == 255:
                        cl_present = cl_present[:-1]
                    if len(cl_present) == 0:
                        continue

                    for cl in cl_present:
                        # [dim, k]
                        features_cl = features[bs][(labels_down[bs] == cl).expand(C, -1, -1)].view(C, -1).detach()
                        # [dim]
                        features_cl = F.normalize(torch.mean(features_cl, dim=-1).float(), p=2, dim=0)
                        cl_prototypes = F.normalize(prototypes[int(cl)], p=2, dim=0)
                        similarity = torch.dot(features_cl, cl_prototypes)
                        key_name = 'class_' + str(cl.item())
                        img_name = data['image_name'][bs]
                        img_similarity_dict_new[key_name].update({img_name: similarity.item()})

            img_similarity_dict.update(img_similarity_dict_new)
        else:
            with torch.no_grad():
                prototypes = prototypes.to(device)
                images, targets, img_names = data['image'].to(device), data['label'].to(device), data['image_name']
                outputs, features, _, _ = model(images, ret_intermediate=False)

                B, C, H, W = features.shape[0], features.shape[1], features.shape[2], features.shape[3]
                labels = data['label'].unsqueeze(dim=1).to(device)
                labels_down = (F.interpolate(input=labels.double(), size=(H, W), mode='nearest')).long()

                for bs in range(B):
                    cl_present = torch.unique(input=labels_down[bs]).long()
                    if cl_present[0] == 0:
                        cl_present = cl_present[1:]
                    if len(cl_present) == 0:
                        continue
                    if cl_present[-1] == 255:
                        cl_present = cl_present[:-1]
                    if len(cl_present) == 0:
                        continue

                    for cl in cl_present:
                        # [dim, k]
                        features_cl = features[bs][(labels_down[bs] == cl).expand(C, -1, -1)].view(C, -1).detach()
                        # [dim]
                        features_cl = F.normalize(torch.mean(features_cl, dim=-1).float(), p=2, dim=0)
                        cl_prototypes = F.normalize(prototypes[int(cl)], p=2, dim=0)
                        similarity = torch.dot(features_cl, cl_prototypes)
                        key_name = 'class_' + str(cl.item())
                        img_name = data['image_name'][bs]
                        img_similarity_dict[key_name].update({img_name: similarity.item()})

            images, targets, img_names = data['image'].to(device), data['label'].to(device), data['image_name']

        for b in range(images.size(0)):
            img_name = img_names[b]
            target = targets[b]
            labels = torch.unique(target).detach().cpu().numpy().tolist()
            if 0 in labels:
                labels.remove(0)

            memory_candidates.append([img_name, labels])

        if batch_idx % 100 == 0:
            logger.info(f"{batch_idx}/{len(train_loader)}")

    logger.info(f"...end memory candidates collection : {len(memory_candidates)}")

    model.to('cpu')
    torch.cuda.empty_cache()
    ####################################################################################################
    logger.info("...start memory list generation")
    curr_memory_list = {f"class_{cls}": [] for cls in range(1, prev_num_classes + 1)}  # 1~15
    sorted_memory_candidates = memory_candidates.copy()
    # np.random.shuffle(sorted_memory_candidates)

    class_max_num_sampled = memory_size // len(old_classes)
    near = class_max_num_sampled // 2

    for cls in old_classes:
        class_name = 'class_' + str(cls)
        dict = img_similarity_dict[class_name]
        sorted_list = sorted(dict.items(), key=lambda x: x[1])
        for i in range(class_max_num_sampled - near):
            curr_memory_list[class_name].append([sorted_list[i][0], cls])
        sorted_list = sorted(dict.items(), key=lambda x: x[1], reverse=True)
        for j in range(near):
            curr_memory_list[class_name].append([sorted_list[j][0], cls])

    random_class_order = old_classes.copy()
    np.random.shuffle(random_class_order)
    num_sampled = class_max_num_sampled * len(old_classes)

    while memory_size > num_sampled:
        for cls in random_class_order:
            for idx, mem in enumerate(sorted_memory_candidates):
                img_name, labels = mem
                if isinstance(labels, int):
                    labels = [labels]
                if cls in labels:
                    flag = 0
                    for name in curr_memory_list[f"class_{cls}"]:
                        if img_name == name[0]:
                            flag = 1
                    if flag == 1:
                        continue
                    curr_memory_list[f"class_{cls}"].append(mem)
                    num_sampled += 1
                    del sorted_memory_candidates[idx]
                    break

            if memory_size <= num_sampled:
                break

    ######################################
    """ save memory info """
    memory_str = ''
    for i in range(1, prev_num_classes + 1):
        memory_str += f"\nclass_{i}: "
        for j in curr_memory_list[f"class_{i}"]:
            if task_dataset == 'ade':
                memory_str += j[0].split()[0][10:]
            elif task_dataset == 'voc':
                memory_str += j[0].split()[0][12:]
                print(j[0], j[0].split()[0], j[0].split()[0][12:])
            else:
                raise NotImplementedError
            memory_str += ' '
    logger.info(memory_str)

    sampled_memory_list = [mem for mem_cls in curr_memory_list.values() for mem in mem_cls]  # gather all memory

    memory_list[f"step_{task_step}"] = {
        "memory_candidates": sampled_memory_list,
        "memory_list": sorted([mem[0] for mem in sampled_memory_list]),
        "img_similarity": img_similarity_dict
    }

    if torch.distributed.get_rank() == 0:
        with open(memory_json, "w") as json_file:
            json.dump(memory_list, json_file)

    torch.distributed.barrier()


# 先按照易混淆程度，为每个类别分配空间（K >= 2），然后再每个类别中，保存距离类别中心最近的K//2个样本和距离类别中心最远的K-K//2个样本，若还有多余的空间，则随机均匀采样
def class_information_maximum_sampling(config, model, prototypes, train_loader, task_info, logger, gpu):
    if gpu is None:
        # setup GPU device if available, move model into configured device
        device, device_ids = _prepare_device(config['n_gpu'], logger)
    else:
        device = gpu
        device_ids = None

    if not torch.cuda.is_available():
        logger.info("using CPU, this will be slow")
    elif config['multiprocessing_distributed']:
        if gpu is not None:
            torch.cuda.set_device(device)
            model.cuda()
            # When using a single GPU per process and per
            # DDP, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            model = DDP(model, device_ids=[gpu])  # Detectron: broadcast_buffers=False
        else:
            model.cuda()
            # DDP will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = DDP(model)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = nn.DataParallel(model, device_ids=device_ids)

    task_dataset, task_setting, task_name, task_step = task_info
    new_classes, old_classes = get_task_labels(task_dataset, task_name, task_step)
    prev_num_classes = len(old_classes)  # 15
    now_num_classes = len(new_classes)

    # memory_json = f'./data/{task_dataset}/{task_setting}_{task_name}_memory.json'
    memory_json = config.save_dir.parent / f'step_{task_step}' / 'memory.json'
    memory_size = config['data_loader']['args']['memory']['mem_size']

    if task_step > 1:
        old_memory_json = config.save_dir.parent / f'step_{task_step - 1}' / 'memory.json'
        with open(old_memory_json, "r") as json_file:
            memory_list = json.load(json_file)
        memory_candidates = memory_list[f"step_{task_step - 1}"]["memory_candidates"]
        img_similarity_dict = memory_list[f"step_{task_step - 1}"]["img_similarity"]
        img_similarity_dict_new = {f"class_{cls}": {} for cls in range(prev_num_classes, prev_num_classes + now_num_classes)}  # 16
    else:
        memory_list = {}
        memory_candidates = []
        img_similarity_dict = {f"class_{cls}": {} for cls in range(1, prev_num_classes + 1)}  # 1~15
    logger.info("...start memory candidates collection")
    torch.distributed.barrier()

    model.eval()
    for batch_idx, data in enumerate(train_loader):
        if task_step > 1:
            with torch.no_grad():
                prototypes = prototypes.to(device)
                images, targets, img_names = data['image'].to(device), data['label'].to(device), data['image_name']
                outputs, features, _, _ = model(images, ret_intermediate=False)
                logit = torch.sigmoid(outputs).detach()
                pred_scores, pred_labels = torch.max(logit[:, 1:], dim=1)
                pred_labels += 1

                """ pseudo labeling """
                targets = torch.where((targets == 0) & (pred_scores >= 0.9), pred_labels.long(), targets.long())

                B, C, H, W = features.shape[0], features.shape[1], features.shape[2], features.shape[3]
                labels = data['label'].unsqueeze(dim=1).to(device)
                labels_down = (F.interpolate(input=labels.double(), size=(H, W), mode='nearest')).long()

                for bs in range(B):
                    cl_present = torch.unique(input=labels_down[bs]).long()
                    if cl_present[0] == 0:
                        cl_present = cl_present[1:]
                    if len(cl_present) == 0:
                        continue
                    if cl_present[-1] == 255:
                        cl_present = cl_present[:-1]
                    if len(cl_present) == 0:
                        continue

                    for cl in cl_present:
                        # [dim, k]
                        features_cl = features[bs][(labels_down[bs] == cl).expand(C, -1, -1)].view(C, -1)
                        # [dim]
                        features_cl = F.normalize(torch.mean(features_cl, dim=-1).float(), p=2, dim=0)
                        cl_prototypes = F.normalize(prototypes[int(cl)], p=2, dim=0)
                        similarity = torch.dot(features_cl, cl_prototypes)
                        key_name = 'class_' + str(cl.item())
                        img_name = data['image_name'][bs]
                        img_similarity_dict_new[key_name].update({img_name: similarity.item()})

            img_similarity_dict.update(img_similarity_dict_new)
        else:
            with torch.no_grad():
                prototypes = prototypes.to(device)
                images, targets, img_names = data['image'].to(device), data['label'].to(device), data['image_name']
                outputs, features, _, _ = model(images, ret_intermediate=False)

                B, C, H, W = features.shape[0], features.shape[1], features.shape[2], features.shape[3]
                labels = data['label'].unsqueeze(dim=1).to(device)
                labels_down = (F.interpolate(input=labels.double(), size=(H, W), mode='nearest')).long()

                for bs in range(B):
                    cl_present = torch.unique(input=labels_down[bs]).long()
                    if cl_present[0] == 0:
                        cl_present = cl_present[1:]
                    if len(cl_present) == 0:
                        continue
                    if cl_present[-1] == 255:
                        cl_present = cl_present[:-1]
                    if len(cl_present) == 0:
                        continue

                    for cl in cl_present:
                        # [dim, k]
                        features_cl = features[bs][(labels_down[bs] == cl).expand(C, -1, -1)].view(C, -1).detach()
                        # [dim]
                        features_cl = F.normalize(torch.mean(features_cl, dim=-1).float(), p=2, dim=0)
                        cl_prototypes = F.normalize(prototypes[int(cl)], p=2, dim=0)
                        similarity = torch.dot(features_cl, cl_prototypes)
                        key_name = 'class_' + str(cl.item())
                        img_name = data['image_name'][bs]
                        img_similarity_dict[key_name].update({img_name: similarity.item()})

            images, targets, img_names = data['image'].to(device), data['label'].to(device), data['image_name']

        for b in range(images.size(0)):
            img_name = img_names[b]
            target = targets[b]
            labels = torch.unique(target).detach().cpu().numpy().tolist()
            if 0 in labels:
                labels.remove(0)

            memory_candidates.append([img_name, labels])

        if batch_idx % 100 == 0:
            logger.info(f"{batch_idx}/{len(train_loader)}")

    logger.info(f"...end memory candidates collection : {len(memory_candidates)}")

    model.to('cpu')
    torch.cuda.empty_cache()
    ####################################################################################################
    logger.info("...start memory list generation")
    curr_memory_list = {f"class_{cls}": [] for cls in range(1, prev_num_classes + 1)}  # 1~15
    sorted_memory_candidates = memory_candidates.copy()
    # np.random.shuffle(sorted_memory_candidates)

    # 初始化一个形状为 (prev_num_classes, prev_num_classes) 的零张量，用于存储余弦相似度计算结果
    cos_sim = torch.zeros(prev_num_classes + 1, prev_num_classes + 1)
    sum_cos_sim = torch.tensor(0.)
    # 计算余弦相似度
    for i in range(1, prev_num_classes + 1):
        for j in range(1, prev_num_classes + 1):
            prototypes_i = F.normalize(prototypes[i], p=2, dim=0)
            prototypes_j = F.normalize(prototypes[j], p=2, dim=0)
            cos_sim[i, j] = F.cosine_similarity(prototypes_i, prototypes_j, dim=0)
            sum_cos_sim += cos_sim[i, j]

    class_sampled_mat = torch.zeros(prev_num_classes + 1, prev_num_classes + 1)
    for i in range(1, prev_num_classes + 1):
        for j in range(1, prev_num_classes + 1):
            class_sampled_mat[i, j] = cos_sim[i, j] / sum_cos_sim * memory_size * config['data_loader']['args']['memory']['mem_ratio']

    per_class_sampled_num = torch.floor(torch.sum(class_sampled_mat, dim=0))

    num_sampled = 0

    for cls in old_classes:
        class_max_num_sampled = int(per_class_sampled_num[cls] + memory_size * (1 - config['data_loader']['args']['memory']['mem_ratio']) / len(old_classes))
        near = class_max_num_sampled // 2
        # print(class_max_num_sampled, near)
        class_name = 'class_' + str(cls)
        dict = img_similarity_dict[class_name]
        sorted_list = sorted(dict.items(), key=lambda x: x[1])
        for i in range(class_max_num_sampled - near):
            curr_memory_list[class_name].append([sorted_list[i][0], cls])
        sorted_list = sorted(dict.items(), key=lambda x: x[1], reverse=True)
        for j in range(near):
            curr_memory_list[class_name].append([sorted_list[j][0], cls])
        num_sampled += class_max_num_sampled

    random_class_order = old_classes.copy()
    np.random.shuffle(random_class_order)

    while memory_size > num_sampled:
        for cls in random_class_order:
            for idx, mem in enumerate(sorted_memory_candidates):
                img_name, labels = mem
                if isinstance(labels, int):
                    labels = [labels]
                if cls in labels:
                    flag = 0
                    for name in curr_memory_list[f"class_{cls}"]:
                        if img_name == name[0]:
                            flag = 1
                    if flag == 1:
                        continue
                    curr_memory_list[f"class_{cls}"].append(mem)
                    num_sampled += 1
                    del sorted_memory_candidates[idx]
                    break

            if memory_size <= num_sampled:
                break

    ######################################
    """ save memory info """
    memory_str = ''
    for i in range(1, prev_num_classes + 1):
        memory_str += f"\nclass_{i}: "
        for j in curr_memory_list[f"class_{i}"]:
            if task_dataset == 'ade':
                memory_str += j[0].split()[0][10:]
            elif task_dataset == 'voc':
                memory_str += j[0].split()[0][12:]
                print(j[0], j[0].split()[0], j[0].split()[0][12:])
            else:
                raise NotImplementedError
            memory_str += ' '
    logger.info(memory_str)

    sampled_memory_list = [mem for mem_cls in curr_memory_list.values() for mem in mem_cls]  # gather all memory

    memory_list[f"step_{task_step}"] = {
        "memory_candidates": sampled_memory_list,
        "memory_list": sorted([mem[0] for mem in sampled_memory_list]),
        "img_similarity": img_similarity_dict
    }

    if torch.distributed.get_rank() == 0:
        with open(memory_json, "w") as json_file:
            json.dump(memory_list, json_file)

    torch.distributed.barrier()


def sampling_based_on_confidence_score(config, model, prototypes, train_loader, task_info, logger, gpu):
    if gpu is None:
        # setup GPU device if available, move model into configured device
        device, device_ids = _prepare_device(config['n_gpu'], logger)
    else:
        device = gpu
        device_ids = None

    if not torch.cuda.is_available():
        logger.info("using CPU, this will be slow")
    elif config['multiprocessing_distributed']:
        if gpu is not None:
            torch.cuda.set_device(device)
            model.cuda()
            # When using a single GPU per process and per
            # DDP, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            model = DDP(model, device_ids=[gpu])  # Detectron: broadcast_buffers=False
        else:
            model.cuda()
            # DDP will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = DDP(model)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = nn.DataParallel(model, device_ids=device_ids)

    task_dataset, task_setting, task_name, task_step = task_info
    new_classes, old_classes = get_task_labels(task_dataset, task_name, task_step)
    prev_num_classes = len(old_classes)  # 15
    now_num_classes = len(new_classes)

    # memory_json = f'./data/{task_dataset}/{task_setting}_{task_name}_memory.json'
    memory_json = config.save_dir.parent / f'step_{task_step}' / 'memory.json'
    memory_size = config['data_loader']['args']['memory']['mem_size']

    if task_step > 1:
        old_memory_json = config.save_dir.parent / f'step_{task_step - 1}' / 'memory.json'
        with open(old_memory_json, "r") as json_file:
            memory_list = json.load(json_file)
        memory_candidates = memory_list[f"step_{task_step - 1}"]["memory_candidates"]
        img_confidence_score_dict = memory_list[f"step_{task_step - 1}"]["img_confidence_score"]
        img_confidence_score_dict_new = {f"class_{cls}": {} for cls in range(prev_num_classes, prev_num_classes + now_num_classes)}  # 16
    else:
        memory_list = {}
        memory_candidates = []
        img_confidence_score_dict = {f"class_{cls}": {} for cls in range(1, prev_num_classes + 1)}  # 1~15
    logger.info("...start memory candidates collection")
    torch.distributed.barrier()

    model.eval()
    for batch_idx, data in enumerate(train_loader):
        if task_step > 1:
            with torch.no_grad():
                prototypes = prototypes.to(device)
                images, targets, img_names = data['image'].to(device), data['label'].to(device), data['image_name']
                outputs, _, _, _ = model(images, ret_intermediate=False)
                logit = torch.sigmoid(outputs).detach()
                pred_scores, pred_labels = torch.max(logit[:, 1:], dim=1)
                pred_labels += 1

                """ pseudo labeling """
                targets = torch.where((targets == 0) & (pred_scores >= 0.9), pred_labels.long(), targets.long())

                labels = data['label'].long().to(device)

                for bs in range(outputs.shape[0]):
                    cl_present = torch.unique(input=labels[bs]).long()
                    if cl_present[0] == 0:
                        cl_present = cl_present[1:]
                    if len(cl_present) == 0:
                        continue
                    if cl_present[-1] == 255:
                        cl_present = cl_present[:-1]
                    if len(cl_present) == 0:
                        continue

                    for cl in cl_present:
                        score_cl = pred_scores[bs][labels[bs] == cl].view(-1)
                        score_mean = torch.mean(score_cl)
                        max_values, _ = torch.max(score_cl, dim=0)
                        min_values, _ = torch.min(score_cl, dim=0)
                        score_range = max_values.item() - min_values.item()
                        confidence_score = score_mean / score_range
                        key_name = 'class_' + str(cl.item())
                        img_name = data['image_name'][bs]
                        img_confidence_score_dict_new[key_name].update({img_name: confidence_score.item()})

            img_confidence_score_dict.update(img_confidence_score_dict_new)
        else:
            with torch.no_grad():
                images, targets, img_names = data['image'].to(device), data['label'].to(device), data['image_name']
                outputs, _, _, _ = model(images, ret_intermediate=False)
                logit = torch.sigmoid(outputs).detach()
                pred_scores, pred_labels = torch.max(logit[:, 1:], dim=1)
                pred_labels += 1

                labels = data['label'].to(device)

                for bs in range(outputs.shape[0]):
                    cl_present = torch.unique(input=labels[bs]).long()
                    if cl_present[0] == 0:
                        cl_present = cl_present[1:]
                    if len(cl_present) == 0:
                        continue
                    if cl_present[-1] == 255:
                        cl_present = cl_present[:-1]
                    if len(cl_present) == 0:
                        continue

                    for cl in cl_present:
                        score_cl = pred_scores[bs][labels[bs] == cl].view(-1)
                        score_mean = torch.mean(score_cl)
                        max_values, _ = torch.max(score_cl, dim=-1)
                        min_values, _ = torch.min(score_cl, dim=-1)
                        score_range = max_values.item() - min_values.item()
                        confidence_score = score_mean / score_range
                        key_name = 'class_' + str(cl.item())
                        img_name = data['image_name'][bs]
                        img_confidence_score_dict[key_name].update({img_name: confidence_score.item()})

            images, targets, img_names = data['image'].to(device), data['label'].to(device), data['image_name']

        for b in range(images.size(0)):
            img_name = img_names[b]
            target = targets[b]
            labels = torch.unique(target).detach().cpu().numpy().tolist()
            if 0 in labels:
                labels.remove(0)

            memory_candidates.append([img_name, labels])

        if batch_idx % 100 == 0:
            logger.info(f"{batch_idx}/{len(train_loader)}")

    logger.info(f"...end memory candidates collection : {len(memory_candidates)}")

    model.to('cpu')
    torch.cuda.empty_cache()
    ####################################################################################################
    logger.info("...start memory list generation")
    curr_memory_list = {f"class_{cls}": [] for cls in range(1, prev_num_classes + 1)}  # 1~15
    sorted_memory_candidates = memory_candidates.copy()
    # np.random.shuffle(sorted_memory_candidates)

    # 初始化一个形状为 (prev_num_classes, prev_num_classes) 的零张量，用于存储余弦相似度计算结果
    cos_sim = torch.zeros(prev_num_classes + 1, prev_num_classes + 1)
    sum_cos_sim = torch.tensor(0.)
    # 计算余弦相似度
    for i in range(1, prev_num_classes + 1):
        for j in range(1, prev_num_classes + 1):
            prototypes_i = F.normalize(prototypes[i], p=2, dim=0)
            prototypes_j = F.normalize(prototypes[j], p=2, dim=0)
            cos_sim[i, j] = F.cosine_similarity(prototypes_i, prototypes_j, dim=0)
            sum_cos_sim += cos_sim[i, j]

    class_sampled_mat = torch.zeros(prev_num_classes + 1, prev_num_classes + 1)
    for i in range(1, prev_num_classes + 1):
        for j in range(1, prev_num_classes + 1):
            class_sampled_mat[i, j] = cos_sim[i, j] / sum_cos_sim * memory_size * config['data_loader']['args']['memory']['mem_ratio']

    per_class_sampled_num = torch.floor(torch.sum(class_sampled_mat, dim=0))

    num_sampled = 0

    for cls in old_classes:
        class_max_num_sampled = int(per_class_sampled_num[cls] + memory_size * (1 - config['data_loader']['args']['memory']['mem_ratio']) / len(old_classes))
        near = class_max_num_sampled // 2
        # print(class_max_num_sampled, near)
        class_name = 'class_' + str(cls)
        dict = img_confidence_score_dict[class_name]
        sorted_list = sorted(dict.items(), key=lambda x: x[1])
        for i in range(class_max_num_sampled - near):
            curr_memory_list[class_name].append([sorted_list[i][0], cls])
        sorted_list = sorted(dict.items(), key=lambda x: x[1], reverse=True)
        for j in range(near):
            curr_memory_list[class_name].append([sorted_list[j][0], cls])
        num_sampled += class_max_num_sampled

    random_class_order = old_classes.copy()
    np.random.shuffle(random_class_order)

    while memory_size > num_sampled:
        for cls in random_class_order:
            for idx, mem in enumerate(sorted_memory_candidates):
                img_name, labels = mem
                if isinstance(labels, int):
                    labels = [labels]
                if cls in labels:
                    flag = 0
                    for name in curr_memory_list[f"class_{cls}"]:
                        if img_name == name[0]:
                            flag = 1
                    if flag == 1:
                        continue
                    curr_memory_list[f"class_{cls}"].append(mem)
                    num_sampled += 1
                    del sorted_memory_candidates[idx]
                    break

            if memory_size <= num_sampled:
                break

    ######################################
    """ save memory info """
    memory_str = ''
    for i in range(1, prev_num_classes + 1):
        memory_str += f"\nclass_{i}: "
        for j in curr_memory_list[f"class_{i}"]:
            if task_dataset == 'ade':
                memory_str += j[0].split()[0][10:]
            elif task_dataset == 'voc':
                memory_str += j[0].split()[0][12:]
                print(j[0], j[0].split()[0], j[0].split()[0][12:])
            else:
                raise NotImplementedError
            memory_str += ' '
    logger.info(memory_str)

    sampled_memory_list = [mem for mem_cls in curr_memory_list.values() for mem in mem_cls]  # gather all memory

    memory_list[f"step_{task_step}"] = {
        "memory_candidates": sampled_memory_list,
        "memory_list": sorted([mem[0] for mem in sampled_memory_list]),
        "img_confidence_score": img_confidence_score_dict
    }

    if torch.distributed.get_rank() == 0:
        with open(memory_json, "w") as json_file:
            json.dump(memory_list, json_file)

    torch.distributed.barrier()


def sampling_based_on_confidence_score_2(config, model, prototypes, train_loader, task_info, logger, gpu):
    if gpu is None:
        # setup GPU device if available, move model into configured device
        device, device_ids = _prepare_device(config['n_gpu'], logger)
    else:
        device = gpu
        device_ids = None

    if not torch.cuda.is_available():
        logger.info("using CPU, this will be slow")
    elif config['multiprocessing_distributed']:
        if gpu is not None:
            torch.cuda.set_device(device)
            model.cuda()
            # When using a single GPU per process and per
            # DDP, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            model = DDP(model, device_ids=[gpu])  # Detectron: broadcast_buffers=False
        else:
            model.cuda()
            # DDP will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = DDP(model)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = nn.DataParallel(model, device_ids=device_ids)

    task_dataset, task_setting, task_name, task_step = task_info
    new_classes, old_classes = get_task_labels(task_dataset, task_name, task_step)
    prev_num_classes = len(old_classes)  # 15
    now_num_classes = len(new_classes)

    # memory_json = f'./data/{task_dataset}/{task_setting}_{task_name}_memory.json'
    memory_json = config.save_dir.parent / f'step_{task_step}' / 'memory.json'
    memory_size = config['data_loader']['args']['memory']['mem_size']

    if task_step > 1:
        old_memory_json = config.save_dir.parent / f'step_{task_step - 1}' / 'memory.json'
        with open(old_memory_json, "r") as json_file:
            memory_list = json.load(json_file)
        memory_candidates = memory_list[f"step_{task_step - 1}"]["memory_candidates"]
        img_confidence_score_dict = memory_list[f"step_{task_step - 1}"]["img_confidence_score"]
        img_confidence_score_dict_new = {f"class_{cls}": {} for cls in range(prev_num_classes, prev_num_classes + now_num_classes)}  # 16
    else:
        memory_list = {}
        memory_candidates = []
        img_confidence_score_dict = {f"class_{cls}": {} for cls in range(1, prev_num_classes + 1)}  # 1~15
    logger.info("...start memory candidates collection")
    torch.distributed.barrier()

    model.eval()
    for batch_idx, data in enumerate(train_loader):
        if task_step > 1:
            with torch.no_grad():
                prototypes = prototypes.to(device)
                images, targets, img_names = data['image'].to(device), data['label'].to(device), data['image_name']
                outputs, _, _, _ = model(images, ret_intermediate=False)
                logit = torch.sigmoid(outputs).detach()
                pred_scores, pred_labels = torch.max(logit[:, 1:], dim=1)
                pred_labels += 1

                """ pseudo labeling """
                targets = torch.where((targets == 0) & (pred_scores >= 0.9), pred_labels.long(), targets.long())

                labels = data['label'].long().to(device)

                for bs in range(outputs.shape[0]):
                    cl_present = torch.unique(input=labels[bs]).long()
                    if cl_present[0] == 0:
                        cl_present = cl_present[1:]
                    if len(cl_present) == 0:
                        continue
                    if cl_present[-1] == 255:
                        cl_present = cl_present[:-1]
                    if len(cl_present) == 0:
                        continue

                    for cl in cl_present:
                        score_cl = pred_scores[bs][labels[bs] == cl].view(-1)
                        score_mean = torch.mean(score_cl)
                        max_values, _ = torch.max(score_cl, dim=0)
                        min_values, _ = torch.min(score_cl, dim=0)
                        score_range = max_values.item() - min_values.item()
                        confidence_score = score_mean / score_range
                        key_name = 'class_' + str(cl.item())
                        img_name = data['image_name'][bs]
                        img_confidence_score_dict_new[key_name].update({img_name: confidence_score.item()})

            img_confidence_score_dict.update(img_confidence_score_dict_new)
        else:
            with torch.no_grad():
                images, targets, img_names = data['image'].to(device), data['label'].to(device), data['image_name']
                outputs, _, _, _ = model(images, ret_intermediate=False)
                logit = torch.sigmoid(outputs).detach()
                pred_scores, pred_labels = torch.max(logit[:, 1:], dim=1)
                pred_labels += 1

                labels = data['label'].to(device)

                for bs in range(outputs.shape[0]):
                    cl_present = torch.unique(input=labels[bs]).long()
                    if cl_present[0] == 0:
                        cl_present = cl_present[1:]
                    if len(cl_present) == 0:
                        continue
                    if cl_present[-1] == 255:
                        cl_present = cl_present[:-1]
                    if len(cl_present) == 0:
                        continue

                    for cl in cl_present:
                        score_cl = pred_scores[bs][labels[bs] == cl].view(-1)
                        score_mean = torch.mean(score_cl)
                        max_values, _ = torch.max(score_cl, dim=-1)
                        min_values, _ = torch.min(score_cl, dim=-1)
                        score_range = max_values.item() - min_values.item()
                        confidence_score = score_mean / score_range
                        key_name = 'class_' + str(cl.item())
                        img_name = data['image_name'][bs]
                        img_confidence_score_dict[key_name].update({img_name: confidence_score.item()})

            images, targets, img_names = data['image'].to(device), data['label'].to(device), data['image_name']

        for b in range(images.size(0)):
            img_name = img_names[b]
            target = targets[b]
            labels = torch.unique(target).detach().cpu().numpy().tolist()
            if 0 in labels:
                labels.remove(0)

            memory_candidates.append([img_name, labels])

        if batch_idx % 100 == 0:
            logger.info(f"{batch_idx}/{len(train_loader)}")

    logger.info(f"...end memory candidates collection : {len(memory_candidates)}")

    model.to('cpu')
    torch.cuda.empty_cache()
    ####################################################################################################
    logger.info("...start memory list generation")
    curr_memory_list = {f"class_{cls}": [] for cls in range(1, prev_num_classes + 1)}  # 1~15
    sorted_memory_candidates = memory_candidates.copy()
    # np.random.shuffle(sorted_memory_candidates)

    # 初始化一个形状为 (prev_num_classes, prev_num_classes) 的零张量，用于存储余弦相似度计算结果
    cos_sim = torch.zeros(prev_num_classes + 1, prev_num_classes + 1)
    sum_cos_sim = torch.tensor(0.)
    # 计算余弦相似度
    for i in range(1, prev_num_classes + 1):
        for j in range(1, prev_num_classes + 1):
            prototypes_i = F.normalize(prototypes[i], p=2, dim=0)
            prototypes_j = F.normalize(prototypes[j], p=2, dim=0)
            cos_sim[i, j] = F.cosine_similarity(prototypes_i, prototypes_j, dim=0)
            sum_cos_sim += cos_sim[i, j]

    class_sampled_mat = torch.zeros(prev_num_classes + 1, prev_num_classes + 1)
    for i in range(1, prev_num_classes + 1):
        for j in range(1, prev_num_classes + 1):
            class_sampled_mat[i, j] = cos_sim[i, j] / sum_cos_sim * memory_size * config['data_loader']['args']['memory']['mem_ratio']

    per_class_sampled_num = torch.floor(torch.sum(class_sampled_mat, dim=0))

    num_sampled = 0

    for cls in old_classes:
        class_max_num_sampled = int(per_class_sampled_num[cls] + memory_size * (1 - config['data_loader']['args']['memory']['mem_ratio']) / len(old_classes))
        # print(class_max_num_sampled, near)
        class_name = 'class_' + str(cls)
        dict = img_confidence_score_dict[class_name]
        sorted_list = sorted(dict.items(), key=lambda x: x[1])
        for i in range(class_max_num_sampled):
            curr_memory_list[class_name].append([sorted_list[i][0], cls])
        num_sampled += class_max_num_sampled

    random_class_order = old_classes.copy()
    np.random.shuffle(random_class_order)

    while memory_size > num_sampled:
        for cls in random_class_order:
            for idx, mem in enumerate(sorted_memory_candidates):
                img_name, labels = mem
                if isinstance(labels, int):
                    labels = [labels]
                if cls in labels:
                    flag = 0
                    for name in curr_memory_list[f"class_{cls}"]:
                        if img_name == name[0]:
                            flag = 1
                    if flag == 1:
                        continue
                    curr_memory_list[f"class_{cls}"].append(mem)
                    num_sampled += 1
                    del sorted_memory_candidates[idx]
                    break

            if memory_size <= num_sampled:
                break

    ######################################
    """ save memory info """
    memory_str = ''
    for i in range(1, prev_num_classes + 1):
        memory_str += f"\nclass_{i}: "
        for j in curr_memory_list[f"class_{i}"]:
            if task_dataset == 'ade':
                memory_str += j[0].split()[0][10:]
            elif task_dataset == 'voc':
                memory_str += j[0].split()[0][12:]
                print(j[0], j[0].split()[0], j[0].split()[0][12:])
            else:
                raise NotImplementedError
            memory_str += ' '
    logger.info(memory_str)

    sampled_memory_list = [mem for mem_cls in curr_memory_list.values() for mem in mem_cls]  # gather all memory

    memory_list[f"step_{task_step}"] = {
        "memory_candidates": sampled_memory_list,
        "memory_list": sorted([mem[0] for mem in sampled_memory_list]),
        "img_confidence_score": img_confidence_score_dict
    }

    if torch.distributed.get_rank() == 0:
        with open(memory_json, "w") as json_file:
            json.dump(memory_list, json_file)

    torch.distributed.barrier()


def sampling_based_on_confidence_score_3(config, model, prototypes, train_loader, task_info, logger, gpu):
    if gpu is None:
        # setup GPU device if available, move model into configured device
        device, device_ids = _prepare_device(config['n_gpu'], logger)
    else:
        device = gpu
        device_ids = None

    if not torch.cuda.is_available():
        logger.info("using CPU, this will be slow")
    elif config['multiprocessing_distributed']:
        if gpu is not None:
            torch.cuda.set_device(device)
            model.cuda()
            # When using a single GPU per process and per
            # DDP, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            model = DDP(model, device_ids=[gpu])  # Detectron: broadcast_buffers=False
        else:
            model.cuda()
            # DDP will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = DDP(model)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = nn.DataParallel(model, device_ids=device_ids)

    task_dataset, task_setting, task_name, task_step = task_info
    new_classes, old_classes = get_task_labels(task_dataset, task_name, task_step)
    prev_num_classes = len(old_classes)  # 15
    now_num_classes = len(new_classes)

    # memory_json = f'./data/{task_dataset}/{task_setting}_{task_name}_memory.json'
    memory_json = config.save_dir.parent / f'step_{task_step}' / 'memory.json'
    memory_size = config['data_loader']['args']['memory']['mem_size']

    if task_step > 1:
        old_memory_json = config.save_dir.parent / f'step_{task_step - 1}' / 'memory.json'
        with open(old_memory_json, "r") as json_file:
            memory_list = json.load(json_file)
        memory_candidates = memory_list[f"step_{task_step - 1}"]["memory_candidates"]
        img_confidence_score_dict = memory_list[f"step_{task_step - 1}"]["img_confidence_score"]
        img_confidence_score_dict_new = {f"class_{cls}": {} for cls in range(prev_num_classes, prev_num_classes + now_num_classes)}  # 16
    else:
        memory_list = {}
        memory_candidates = []
        img_confidence_score_dict = {f"class_{cls}": {} for cls in range(1, prev_num_classes + 1)}  # 1~15
    logger.info("...start memory candidates collection")
    torch.distributed.barrier()

    model.eval()
    for batch_idx, data in enumerate(train_loader):
        if task_step > 1:
            with torch.no_grad():
                prototypes = prototypes.to(device)
                images, targets, img_names = data['image'].to(device), data['label'].to(device), data['image_name']
                outputs, _, _, _ = model(images, ret_intermediate=False)
                logit = torch.sigmoid(outputs).detach()
                pred_scores, pred_labels = torch.max(logit[:, 1:], dim=1)
                pred_labels += 1

                """ pseudo labeling """
                targets = torch.where((targets == 0) & (pred_scores >= 0.9), pred_labels.long(), targets.long())

                labels = data['label'].long().to(device)

                for bs in range(outputs.shape[0]):
                    cl_present = torch.unique(input=labels[bs]).long()
                    if cl_present[0] == 0:
                        cl_present = cl_present[1:]
                    if len(cl_present) == 0:
                        continue
                    if cl_present[-1] == 255:
                        cl_present = cl_present[:-1]
                    if len(cl_present) == 0:
                        continue

                    for cl in cl_present:
                        score_cl = pred_scores[bs][labels[bs] == cl].view(-1)
                        score_mean = torch.mean(score_cl)
                        max_values, _ = torch.max(score_cl, dim=0)
                        min_values, _ = torch.min(score_cl, dim=0)
                        score_range = max_values.item() - min_values.item()
                        confidence_score = score_mean / score_range
                        key_name = 'class_' + str(cl.item())
                        img_name = data['image_name'][bs]
                        img_confidence_score_dict_new[key_name].update({img_name: confidence_score.item()})

            img_confidence_score_dict.update(img_confidence_score_dict_new)
        else:
            with torch.no_grad():
                images, targets, img_names = data['image'].to(device), data['label'].to(device), data['image_name']
                outputs, _, _, _ = model(images, ret_intermediate=False)
                logit = torch.sigmoid(outputs).detach()
                pred_scores, pred_labels = torch.max(logit[:, 1:], dim=1)
                pred_labels += 1

                labels = data['label'].to(device)

                for bs in range(outputs.shape[0]):
                    cl_present = torch.unique(input=labels[bs]).long()
                    if cl_present[0] == 0:
                        cl_present = cl_present[1:]
                    if len(cl_present) == 0:
                        continue
                    if cl_present[-1] == 255:
                        cl_present = cl_present[:-1]
                    if len(cl_present) == 0:
                        continue

                    for cl in cl_present:
                        score_cl = pred_scores[bs][labels[bs] == cl].view(-1)
                        score_mean = torch.mean(score_cl)
                        max_values, _ = torch.max(score_cl, dim=-1)
                        min_values, _ = torch.min(score_cl, dim=-1)
                        score_range = max_values.item() - min_values.item()
                        confidence_score = score_mean / score_range
                        key_name = 'class_' + str(cl.item())
                        img_name = data['image_name'][bs]
                        img_confidence_score_dict[key_name].update({img_name: confidence_score.item()})

            images, targets, img_names = data['image'].to(device), data['label'].to(device), data['image_name']

        for b in range(images.size(0)):
            img_name = img_names[b]
            target = targets[b]
            labels = torch.unique(target).detach().cpu().numpy().tolist()
            if 0 in labels:
                labels.remove(0)

            memory_candidates.append([img_name, labels])

        if batch_idx % 100 == 0:
            logger.info(f"{batch_idx}/{len(train_loader)}")

    logger.info(f"...end memory candidates collection : {len(memory_candidates)}")

    model.to('cpu')
    torch.cuda.empty_cache()
    ####################################################################################################
    logger.info("...start memory list generation")
    curr_memory_list = {f"class_{cls}": [] for cls in range(1, prev_num_classes + 1)}  # 1~15
    sorted_memory_candidates = memory_candidates.copy()
    # np.random.shuffle(sorted_memory_candidates)

    # 初始化一个形状为 (prev_num_classes, prev_num_classes) 的零张量，用于存储余弦相似度计算结果
    cos_sim = torch.zeros(prev_num_classes + 1, prev_num_classes + 1)
    sum_cos_sim = torch.tensor(0.)
    # 计算余弦相似度
    for i in range(1, prev_num_classes + 1):
        for j in range(1, prev_num_classes + 1):
            prototypes_i = F.normalize(prototypes[i], p=2, dim=0)
            prototypes_j = F.normalize(prototypes[j], p=2, dim=0)
            cos_sim[i, j] = F.cosine_similarity(prototypes_i, prototypes_j, dim=0)
            sum_cos_sim += cos_sim[i, j]

    class_sampled_mat = torch.zeros(prev_num_classes + 1, prev_num_classes + 1)
    for i in range(1, prev_num_classes + 1):
        for j in range(1, prev_num_classes + 1):
            class_sampled_mat[i, j] = cos_sim[i, j] / sum_cos_sim * memory_size * config['data_loader']['args']['memory']['mem_ratio']

    per_class_sampled_num = torch.floor(torch.sum(class_sampled_mat, dim=0))

    num_sampled = 0

    for cls in old_classes:
        class_max_num_sampled = int(per_class_sampled_num[cls] + memory_size * (1 - config['data_loader']['args']['memory']['mem_ratio']) / len(old_classes))
        # print(class_max_num_sampled, near)
        class_name = 'class_' + str(cls)
        dict = img_confidence_score_dict[class_name]
        sorted_list = sorted(dict.items(), key=lambda x: x[1], reverse=True)
        for j in range(class_max_num_sampled):
            curr_memory_list[class_name].append([sorted_list[j][0], cls])
        num_sampled += class_max_num_sampled

    random_class_order = old_classes.copy()
    np.random.shuffle(random_class_order)

    while memory_size > num_sampled:
        for cls in random_class_order:
            for idx, mem in enumerate(sorted_memory_candidates):
                img_name, labels = mem
                if isinstance(labels, int):
                    labels = [labels]
                if cls in labels:
                    flag = 0
                    for name in curr_memory_list[f"class_{cls}"]:
                        if img_name == name[0]:
                            flag = 1
                    if flag == 1:
                        continue
                    curr_memory_list[f"class_{cls}"].append(mem)
                    num_sampled += 1
                    del sorted_memory_candidates[idx]
                    break

            if memory_size <= num_sampled:
                break

    ######################################
    """ save memory info """
    memory_str = ''
    for i in range(1, prev_num_classes + 1):
        memory_str += f"\nclass_{i}: "
        for j in curr_memory_list[f"class_{i}"]:
            if task_dataset == 'ade':
                memory_str += j[0].split()[0][10:]
            elif task_dataset == 'voc':
                memory_str += j[0].split()[0][12:]
                print(j[0], j[0].split()[0], j[0].split()[0][12:])
            else:
                raise NotImplementedError
            memory_str += ' '
    logger.info(memory_str)

    sampled_memory_list = [mem for mem_cls in curr_memory_list.values() for mem in mem_cls]  # gather all memory

    memory_list[f"step_{task_step}"] = {
        "memory_candidates": sampled_memory_list,
        "memory_list": sorted([mem[0] for mem in sampled_memory_list]),
        "img_confidence_score": img_confidence_score_dict
    }

    if torch.distributed.get_rank() == 0:
        with open(memory_json, "w") as json_file:
            json.dump(memory_list, json_file)

    torch.distributed.barrier()