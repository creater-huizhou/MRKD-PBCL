import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP
from base import BaseTrainer
from utils import MetricTracker, MetricTracker_scalars
from models.loss import *
from data_loader import ADE
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(rc={'figure.figsize': (11.7, 8.27)})
from tsnecuda import TSNE
from MulticoreTSNE import MulticoreTSNE
import os


# Trainer class for a base step
class Trainer_Segformer_base_joint(BaseTrainer):
    def __init__(self, model, optimizer, evaluator, config, task_info, data_loader, lr_scheduler=None,
                 logger=None, gpu=None):
        super().__init__(config, logger, gpu)
        if not torch.cuda.is_available():
            logger.info("using CPU, this will be slow")
        elif config['multiprocessing_distributed']:
            if gpu is not None:
                torch.cuda.set_device(self.device)
                model.to(self.device)
                # When using a single GPU per process and per
                # DDP, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                self.model = DDP(model, device_ids=[gpu])  # Detectron: broadcast_buffers=False
            else:
                model.to(self.device)
                # DDP will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                self.model = DDP(model)
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            self.device_ids = config["device_ids"]
            self.model = nn.DataParallel(model, device_ids=self.device_ids)

        self.optimizer = optimizer
        self.evaluator_val = evaluator[0]
        self.evaluator_test = evaluator[1]

        self.task_info = task_info
        self.old_classes = self.task_info['old_class']  # 0
        self.new_classes = self.task_info['new_class']  # 19-1: 19 | 15-5: 15 | 15-1: 15...
        self.n_old_classes = len(self.task_info['old_class'])  # 0
        self.n_new_classes = len(self.task_info['new_class'])  # 19-1: 19 | 15-5: 15 | 15-1: 15...
        self.total_classes = self.n_old_classes + self.n_new_classes

        self.train_loader = data_loader[0]
        if self.train_loader is not None:
            self.len_epoch = len(self.train_loader)

        self.val_loader = data_loader[1]
        if self.val_loader is not None:
            self.do_validation = self.val_loader is not None

        self.test_loader = data_loader[2]
        if self.test_loader is not None:
            self.do_test = self.test_loader is not None

        self.lr_scheduler = lr_scheduler

        # For automatic mixed precision(AMP)
        self.scaler = torch.cuda.amp.GradScaler(enabled=config['use_amp'])

        if self.evaluator_val is not None:
            self.metric_ftns_val = [getattr(self.evaluator_val, met) for met in config['metrics']]
        if self.evaluator_test is not None:
            self.metric_ftns_test = [getattr(self.evaluator_test, met) for met in config['metrics']]

        self.train_metrics = MetricTracker(
            'loss', 'loss_mbce',
            writer=self.writer,
            colums=['total', 'counts', 'average'],
        )

        self.valid_metrics = MetricTracker_scalars(writer=self.writer)
        self.test_metrics = MetricTracker_scalars(writer=self.writer)

        if config.resume is not None:
            self._resume_checkpoint(config.resume, config['test'])

        pos_weight = (torch.ones([len(self.task_info['new_class'])], device=self.device) *
                      self.config['hyperparameter']['pos_weight'])
        self.BCELoss = WBCELoss(pos_weight=pos_weight, n_old_classes=self.n_old_classes + 1,
                                n_new_classes=self.n_new_classes)

        self._print_train_info()

    def _print_train_info(self):
        self.logger.info(f"Total loss = {self.config['hyperparameter']['mbce']} * L_mbce")

    def _update_running_stats(self, labels, features, prototypes, count_features):
        C, H, W = features.shape[1], features.shape[2], features.shape[3]
        labels = labels.unsqueeze(dim=1)
        labels_down = F.interpolate(labels.double(), size=(H, W), mode="nearest").long()
        # The category in the current mask
        cl_present = torch.unique(input=labels_down)

        # if overlapped: exclude background as we could not have a reliable statistics
        # if disjoint (not overlapped) and step is > 0: exclude bgr as could contain old classes
        if cl_present[0] == 0:
            cl_present = cl_present[1:]
        if cl_present[-1] == 255:
            cl_present = cl_present[:-1]

        class_num = self.config['data_loader']['args']['class_num']
        decoder_dim = self.config['arch']['args']['decoder_dim']
        features_local_mean = torch.zeros([class_num + 1, decoder_dim]).to(self.device)

        for cl in cl_present:
            features_cl = features[(labels_down == cl).expand(-1, C, -1, -1)].view(C, -1).detach()
            features_local_mean[cl] = torch.mean(features_cl.detach(), dim=-1)
            features_cl_sum = torch.sum(features_cl.detach(), dim=-1)
            # cumulative moving average for each feature vector
            # S_{n+f} = ( sum(x_{n+1} + ... + x_{n+f}) + n * S_n) / (n + f)
            features_running_mean_tot_cl = (features_cl_sum + count_features.detach()[cl] * prototypes.detach()[cl]) \
                                           / (count_features.detach()[cl] + features_cl.shape[-1])
            count_features[cl] += features_cl.shape[-1]
            prototypes[cl] = features_running_mean_tot_cl

        return prototypes, count_features

    def _train_epoch(self, epoch, prototypes, count_features):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        torch.distributed.barrier()

        self.model.train()
        self.train_metrics.reset()
        self.logger.info(f'Epoch - {epoch}')

        # Random shuffling
        if not isinstance(self.train_loader.sampler, torch.utils.data.RandomSampler):
            self.train_loader.sampler.set_epoch(epoch)

        for batch_idx, data in enumerate(self.train_loader):
            data['image'], data['label'] = data['image'].to(self.device), data['label'].to(self.device)
            # print(data['image'].device, data['label'].device)
            with (torch.cuda.amp.autocast(enabled=self.config['use_amp'])):
                # logit: [N, |Ct|+1, H, W]
                # layer_features is a list, every element is [N, D, H, W], the number of
                logit, features, layer_features, pos_neg_features = self.model(data['image'], ret_intermediate=False)

                loss_mbce = self.BCELoss(
                    logit[:, -self.n_new_classes:],  # [N, |Ct|, H, W]
                    data['label'],  # [N, H, W]
                ).mean(dim=[0, 2, 3])  # [|Ct|]

                prototypes, count_features = self._update_running_stats(data['label'],
                                                                        features,
                                                                        prototypes,
                                                                        count_features)


                loss = self.config['hyperparameter']['mbce'] * loss_mbce.sum()

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            self.train_metrics.update('loss_mbce', loss_mbce.sum().item())

            # Get First lr
            if batch_idx == 0:
                self.writer.add_scalars('lr', {'lr': self.optimizer.param_groups[0]['lr']}, epoch - 1)
                self.logger.info(
                    f"lr[0]: {self.optimizer.param_groups[0]['lr']:.6f} / lr[1]: {self.optimizer.param_groups[1]['lr']:.6f}")

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.progress(self.logger, batch_idx, len(self.train_loader))

            if batch_idx == self.len_epoch:
                break

        # average train loss per epoch
        log = self.train_metrics.result()

        val_flag = False
        if self.do_validation and (epoch % self.validation_period) == 0:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
            if self.rank == 0:
                val_flag = True

        return log, val_flag, prototypes, count_features

    def _valid_epoch(self, epoch):
        torch.distributed.barrier()

        log = {}
        self.evaluator_val.reset()
        self.logger.info(f"Number of val loader: {len(self.val_loader)}")

        self.model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.val_loader):
                data['image'], data['label'] = data['image'].to(self.device), data['label'].to(self.device)
                target = data['label'].cpu().numpy()

                logit, _, _, _ = self.model(data['image'])

                logit = torch.sigmoid(logit)
                pred = logit[:, 1:].argmax(dim=1) + 1  # pred: [N. H, W]
                idx = (logit[:, 1:] > 0.5).float()  # logit: [N, C, H, W]
                idx = idx.sum(dim=1)  # logit: [N, H, W]
                pred[idx == 0] = 0  # set background (non-target class)

                pred = pred.cpu().numpy()
                self.evaluator_val.add_batch(target, pred)

            if self.rank == 0:
                self.writer.set_step((epoch), 'valid')

            for met in self.metric_ftns_val:
                if len(met().keys()) > 2:
                    self.valid_metrics.update(met.__name__, [met()['old'], met()['new'], met()['harmonic']], 'old',
                                              'new', 'harmonic', n=1)
                else:
                    self.valid_metrics.update(met.__name__, [met()['overall']], 'overall', n=1)

                if 'old' in met().keys():
                    log.update({met.__name__ + '_old': f"{met()['old']:.2f}"})
                if 'new' in met().keys():
                    log.update({met.__name__ + '_new': f"{met()['new']:.2f}"})
                if 'harmonic' in met().keys():
                    log.update({met.__name__ + '_harmonic': f"{met()['harmonic']:.2f}"})
                if 'overall' in met().keys():
                    log.update({met.__name__ + '_overall': f"{met()['overall']:.2f}"})

        return log

    def _test(self, epoch=None):
        torch.distributed.barrier()

        log = {}
        self.evaluator_test.reset()
        self.logger.info(f"Number of test loader: {len(self.test_loader)}")

        if self.config['data_loader']['args']['task']['step'] > 0:
            pth_name = 'checkpoint-epoch' + str(self.config['trainer']['epochs']) + '.pth'
        else:
            pth_name = 'model_best.pth'

        task_name = self.config['data_loader']['args']['task']['setting'] + '_' + \
                    self.config['data_loader']['args']['task']['name'] + '_' + \
                    self.config['name']

        step_name = 'step_' + str(self.config['data_loader']['args']['task']['step'])
        path = self.config['trainer']['save_dir'] + '/models/' + task_name + '/' + step_name + '/' + pth_name
        checkpoint = torch.load(path, map_location='cpu')
        if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            self.model.module.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

        with torch.no_grad():
            for batch_idx, data in enumerate(self.test_loader):
                data['image'], data['label'] = data['image'].to(self.device), data['label'].to(self.device)
                target = data['label'].cpu().numpy()

                logit, _, _, _ = self.model(data['image'])
                logit = torch.sigmoid(logit)
                pred = logit[:, 1:].argmax(dim=1) + 1  # pred: [N. H, W]

                idx = (logit[:, 1:] > 0.5).float()  # logit: [N, C, H, W]
                idx = idx.sum(dim=1)  # logit: [N, H, W]

                pred[idx == 0] = 0  # set background (non-target class)

                pred = pred.cpu().numpy()
                self.evaluator_test.add_batch(target, pred)

            if epoch is not None:
                if self.rank == 0:
                    self.writer.set_step((epoch), 'test')

            for met in self.metric_ftns_test:
                if epoch is not None:
                    if len(met().keys()) > 2:
                        self.test_metrics.update(met.__name__, [met()['old'], met()['new'], met()['harmonic']], 'old',
                                                 'new', 'harmonic', n=1)
                    else:
                        self.test_metrics.update(met.__name__, [met()['overall']], 'overall', n=1)

                if 'old' in met().keys():
                    log.update({met.__name__ + '_old': f"{met()['old']:.2f}"})
                if 'new' in met().keys():
                    log.update({met.__name__ + '_new': f"{met()['new']:.2f}"})
                if 'harmonic' in met().keys():
                    log.update({met.__name__ + '_harmonic': f"{met()['harmonic']:.2f}"})
                if 'overall' in met().keys():
                    log.update({met.__name__ + '_overall': f"{met()['overall']:.2f}"})
                if 'by_class' in met().keys():
                    by_class_str = '\n'
                    for i in range(len(met()['by_class'])):
                        if i in self.evaluator_test.new_classes_idx:
                            by_class_str = by_class_str + f"{i:2d} *{ADE[i]} {met()['by_class'][i]:.2f}\n"
                        else:
                            by_class_str = by_class_str + f"{i:2d}  {ADE[i]} {met()['by_class'][i]:.2f}\n"
                    log.update({met.__name__ + '_by_class': by_class_str})

        return log


class Trainer_Segformer_base(BaseTrainer):
    def __init__(self, model, optimizer, evaluator, config, task_info, data_loader, lr_scheduler=None,
                 logger=None, gpu=None):
        super().__init__(config, logger, gpu)
        if not torch.cuda.is_available():
            logger.info("using CPU, this will be slow")
        elif config['multiprocessing_distributed']:
            if gpu is not None:
                torch.cuda.set_device(self.device)
                model.to(self.device)
                # When using a single GPU per process and per
                # DDP, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                self.model = DDP(model, device_ids=[gpu])  # Detectron: broadcast_buffers=False
            else:
                model.to(self.device)
                # DDP will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                self.model = DDP(model)
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            self.device_ids = config["device_ids"]
            self.model = nn.DataParallel(model, device_ids=self.device_ids)

        self.optimizer = optimizer
        self.evaluator_val = evaluator[0]
        self.evaluator_test = evaluator[1]

        self.task_info = task_info
        self.old_classes = self.task_info['old_class']  # 0
        self.new_classes = self.task_info['new_class']  # 19-1: 19 | 15-5: 15 | 15-1: 15...
        self.n_old_classes = len(self.task_info['old_class'])  # 0
        self.n_new_classes = len(self.task_info['new_class'])  # 19-1: 19 | 15-5: 15 | 15-1: 15...
        self.total_classes = self.n_old_classes + self.n_new_classes

        self.train_loader = data_loader[0]
        if self.train_loader is not None:
            self.len_epoch = len(self.train_loader)

        self.val_loader = data_loader[1]
        if self.val_loader is not None:
            self.do_validation = self.val_loader is not None

        self.test_loader = data_loader[2]
        if self.test_loader is not None:
            self.do_test = self.test_loader is not None

        self.lr_scheduler = lr_scheduler

        # For automatic mixed precision(AMP)
        self.scaler = torch.cuda.amp.GradScaler(enabled=config['use_amp'])

        if self.evaluator_val is not None:
            self.metric_ftns_val = [getattr(self.evaluator_val, met) for met in config['metrics']]
        if self.evaluator_test is not None:
            self.metric_ftns_test = [getattr(self.evaluator_test, met) for met in config['metrics']]

        self.flag_total = ''

        self.train_metrics = MetricTracker(
            'loss', 'loss_mbce', 'loss_ac',
            writer=self.writer,
            colums=['total', 'counts', 'average'],
        )

        self.valid_metrics = MetricTracker_scalars(writer=self.writer)
        self.test_metrics = MetricTracker_scalars(writer=self.writer)

        if config.resume is not None:
            self._resume_checkpoint(config.resume, config['test'])

        pos_weight = (torch.ones([len(self.task_info['new_class'])], device=self.device) *
                      self.config['hyperparameter']['pos_weight'])
        self.BCELoss = WBCELoss(pos_weight=pos_weight, n_old_classes=self.n_old_classes + 1,
                                n_new_classes=self.n_new_classes)
        self.ACLoss = ACLoss()

        # Multi Scale Region Distillation loss
        self.multi_scale_region_distillation_loss = Multi_Scale_Region_Distillation_Loss(config=self.config,
                                                                                         device=self.device)
        # Prototype Contrastive Learning loss
        self.prototype_balanced_contrastive_loss = Prototype_Balanced_Contrastive_Loss(config=self.config,
                                                                                       device=self.device)
        # Contrastive Learning loss defined in 2021-CVPR Continual Semantic Segmentation via Repulsion-Attraction of Sparse and Disentangled Latent Representations
        self.sdr_contrastive_loss = SDR_Contrastive_Loss(config=self.config,
                                                         device=self.device)
        # Contrastive Learning loss defined in 2023-TPAMI Inherit With Distillation and Evolve With Contrast: Exploring Class Incremental Semantic Segmentation Without Exemplar Memory
        self.idec_contrastive_loss = IDEC_Contrastive_Loss(config=self.config,
                                                           device=self.device)
        # Contrastive Learning loss defined in 2023-ICCV CoinSeg: Contrast Inter- and Intra- Class Representations for Incremental Segmentation
        self.coinseg_contrastive_loss = COINSEG_Contrastive_Loss(config=self.config,
                                                                 device=self.device)
        # Knowledge Distillation loss
        self.knowledge_distillation_loss = KDLoss(pos_weight=None, reduction='none')

        self._print_train_info()

    def _print_train_info(self):
        self.logger.info(
            f"Total loss = {self.config['hyperparameter']['mbce']} * L_mbce"
            f"+ {self.config['hyperparameter']['ac']} * L_ac"
        )

    def _update_running_stats(self, labels, features, prototypes, count_features):
        B, C, H, W = features.shape[0], features.shape[1], features.shape[2], features.shape[3]
        # labels = labels.unsqueeze(dim=1)
        labels_down = F.interpolate(labels.double(), size=(H, W), mode="nearest").long()

        # The category in the current mask
        cl_present = torch.unique(input=labels_down)

        # if overlapped: exclude background as we could not have a reliable statistics
        # if disjoint (not overlapped) and step is > 0: exclude bgr as could contain old classes
        if cl_present[0] == 0:
            cl_present = cl_present[1:]
        if cl_present[-1] == 255:
            cl_present = cl_present[:-1]

        _pl = labels_down.view(-1)  # b*h*w -> bhw
        feat = einops.rearrange(features, 'b c h w -> ( b h w ) c ')

        for cl in cl_present:
            pl = (_pl == cl)
            features_cl = feat[pl].detach()
            features_cl_sum = torch.sum(features_cl.detach(), dim=0)
            # cumulative moving average for each feature vector
            # S_{n+f} = ( sum(x_{n+1} + ... + x_{n+f}) + n * S_n) / (n + f)
            features_running_mean_tot_cl = (features_cl_sum + count_features.detach()[cl] * prototypes.detach()[cl]) / (
                        count_features.detach()[cl] + features_cl.shape[0])
            count_features[cl] += features_cl.shape[0]
            prototypes[cl] = features_running_mean_tot_cl

        return prototypes, count_features

    def _train_epoch(self, epoch, prototypes, count_features):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        torch.distributed.barrier()

        self.model.train()
        self.train_metrics.reset()
        self.logger.info(f'Epoch - {epoch}')

        # Random shuffling
        if not isinstance(self.train_loader.sampler, torch.utils.data.RandomSampler):
            self.train_loader.sampler.set_epoch(epoch)

        for batch_idx, data in enumerate(self.train_loader):
            data['image'], data['label'] = data['image'].to(self.device), data['label'].to(self.device)
            # print(data['image'].device, data['label'].device)
            with (torch.cuda.amp.autocast(enabled=self.config['use_amp'])):
                # logit: [N, |Ct|+1, H, W]
                # layer_features is a list, every element is [N, D, H, W], the number of
                logit, features, layer_features, pos_neg_features = self.model(data['image'], ret_intermediate=False)

                loss_mbce = self.BCELoss(
                    logit[:, -self.n_new_classes:],  # [N, |Ct|, H, W]
                    data['label'],  # [N, H, W]
                ).mean(dim=[0, 2, 3])  # [|Ct|]

                loss_ac = self.ACLoss(logit[:, 0: 1]).mean(dim=[0, 2, 3])  # [1]

                prototypes, count_features = self._update_running_stats(data['label'].unsqueeze(1),
                                                                        features,
                                                                        prototypes,
                                                                        count_features)


                loss = self.config['hyperparameter']['mbce'] * loss_mbce.sum() \
                     + self.config['hyperparameter']['ac'] * loss_ac.sum()

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            self.train_metrics.update('loss_mbce', loss_mbce.sum().item())
            self.train_metrics.update('loss_ac', loss_ac.sum().item())

            # Get First lr
            if batch_idx == 0:
                self.writer.add_scalars('lr', {'lr': self.optimizer.param_groups[0]['lr']}, epoch - 1)
                self.logger.info(
                    f"lr[0]: {self.optimizer.param_groups[0]['lr']:.6f} / lr[1]: {self.optimizer.param_groups[1]['lr']:.6f}")

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.progress(self.logger, batch_idx, len(self.train_loader))

            if batch_idx == self.len_epoch:
                break

        # average train loss per epoch
        log = self.train_metrics.result()

        val_flag = False
        if self.do_validation and (epoch % self.validation_period) == 0:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
            if self.rank == 0:
                val_flag = True

        return log, val_flag, prototypes, count_features

    def _valid_epoch(self, epoch):
        torch.distributed.barrier()

        log = {}
        self.evaluator_val.reset()
        self.logger.info(f"Number of val loader: {len(self.val_loader)}")

        self.model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.val_loader):
                data['image'], data['label'] = data['image'].to(self.device), data['label'].to(self.device)
                target = data['label'].cpu().numpy()

                logit, _, _, _ = self.model(data['image'])

                logit = torch.sigmoid(logit)
                pred = logit[:, 1:].argmax(dim=1) + 1  # pred: [N. H, W]
                idx = (logit[:, 1:] > 0.5).float()  # logit: [N, C, H, W]
                idx = idx.sum(dim=1)  # logit: [N, H, W]
                pred[idx == 0] = 0  # set background (non-target class)

                pred = pred.cpu().numpy()
                self.evaluator_val.add_batch(target, pred)

            if self.rank == 0:
                self.writer.set_step((epoch), 'valid')

            for met in self.metric_ftns_val:
                if len(met().keys()) > 2:
                    self.valid_metrics.update(met.__name__, [met()['old'], met()['new'], met()['harmonic']], 'old',
                                              'new', 'harmonic', n=1)
                else:
                    self.valid_metrics.update(met.__name__, [met()['overall']], 'overall', n=1)

                if 'old' in met().keys():
                    log.update({met.__name__ + '_old': f"{met()['old']:.2f}"})
                if 'new' in met().keys():
                    log.update({met.__name__ + '_new': f"{met()['new']:.2f}"})
                if 'harmonic' in met().keys():
                    log.update({met.__name__ + '_harmonic': f"{met()['harmonic']:.2f}"})
                if 'overall' in met().keys():
                    log.update({met.__name__ + '_overall': f"{met()['overall']:.2f}"})

        return log

    def _test(self, epoch=None):
        torch.distributed.barrier()

        log = {}
        self.evaluator_test.reset()
        self.logger.info(f"Number of test loader: {len(self.test_loader)}")

        if self.config['data_loader']['args']['task']['step'] > 0:
            pth_name = 'checkpoint-epoch' + str(self.config['trainer']['epochs']) + '.pth'
        else:
            pth_name = 'model_best.pth'

        task_name = self.config['data_loader']['args']['task']['setting'] + '_' + \
                    self.config['data_loader']['args']['task']['name'] + '_' + \
                    self.config['name']

        step_name = 'step_' + str(self.config['data_loader']['args']['task']['step'])
        path = self.config['trainer']['save_dir'] + '/models/' + task_name + '/' + step_name + '/' + pth_name
        checkpoint = torch.load(path, map_location='cpu')
        if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            self.model.module.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

        with torch.no_grad():
            for batch_idx, data in enumerate(self.test_loader):
                data['image'], data['label'] = data['image'].to(self.device), data['label'].to(self.device)
                target = data['label'].cpu().numpy()

                logit, _, _, _ = self.model(data['image'])
                logit = torch.sigmoid(logit)
                pred = logit[:, 1:].argmax(dim=1) + 1  # pred: [N. H, W]

                idx = (logit[:, 1:] > 0.5).float()  # logit: [N, C, H, W]
                idx = idx.sum(dim=1)  # logit: [N, H, W]

                pred[idx == 0] = 0  # set background (non-target class)

                pred = pred.cpu().numpy()
                self.evaluator_test.add_batch(target, pred)

            if epoch is not None:
                if self.rank == 0:
                    self.writer.set_step((epoch), 'test')

            for met in self.metric_ftns_test:
                if epoch is not None:
                    if len(met().keys()) > 2:
                        self.test_metrics.update(met.__name__, [met()['old'], met()['new'], met()['harmonic']], 'old',
                                                 'new', 'harmonic', n=1)
                    else:
                        self.test_metrics.update(met.__name__, [met()['overall']], 'overall', n=1)

                if 'old' in met().keys():
                    log.update({met.__name__ + '_old': f"{met()['old']:.2f}"})
                if 'new' in met().keys():
                    log.update({met.__name__ + '_new': f"{met()['new']:.2f}"})
                if 'harmonic' in met().keys():
                    log.update({met.__name__ + '_harmonic': f"{met()['harmonic']:.2f}"})
                if 'overall' in met().keys():
                    log.update({met.__name__ + '_overall': f"{met()['overall']:.2f}"})
                if 'by_class' in met().keys():
                    by_class_str = '\n'
                    for i in range(len(met()['by_class'])):
                        if i in self.evaluator_test.new_classes_idx:
                            by_class_str = by_class_str + f"{i:2d} *{ADE[i]} {met()['by_class'][i]:.2f}\n"
                        else:
                            by_class_str = by_class_str + f"{i:2d}  {ADE[i]} {met()['by_class'][i]:.2f}\n"
                    log.update({met.__name__ + '_by_class': by_class_str})

        return log



# Trainer class for incremental steps
class Trainer_Segformer_incremental(Trainer_Segformer_base):
    def __init__(self, model, model_old, optimizer, evaluator, config, task_info, data_loader,
                 lr_scheduler=None, logger=None, gpu=None):
        super().__init__(model=model, optimizer=optimizer, evaluator=evaluator, config=config, task_info=task_info,
                         data_loader=data_loader, lr_scheduler=lr_scheduler, logger=logger, gpu=gpu)

        if config['multiprocessing_distributed']:
            if gpu is not None:
                if model_old is not None:
                    model_old.to(self.device)
                    self.model_old = DDP(model_old, device_ids=[gpu])
            else:
                if model_old is not None:
                    model_old.to(self.device)
                    self.model_old = DDP(model_old)
        else:
            if model_old is not None:
                self.device_ids = config["device_ids"]
                self.model_old = nn.DataParallel(model_old, device_ids=self.device_ids)

        if config['trainer']['flag_MRKD']:
            flag_mrkd = '1'
        else:
            flag_mrkd = '0'
        if config['trainer']['flag_PBCL']:
            flag_pbcl = '1'
        else:
            flag_pbcl = '0'
        if config['trainer']['flag_SDR_CL']:
            flag_sdr_cl = '1'
        else:
            flag_sdr_cl = '0'
        if config['trainer']['flag_IDEC_CL']:
            flag_idec_cl = '1'
        else:
            flag_idec_cl = '0'
        if config['trainer']['flag_COINSEG_CL']:
            flag_coinseg_cl = '1'
        else:
            flag_coinseg_cl = '0'

        self.flag_total = flag_mrkd + flag_pbcl + flag_sdr_cl + flag_idec_cl + flag_coinseg_cl
        # print(self.flag_total)
        if self.flag_total == '00000':
            self.train_metrics = MetricTracker(
                'loss', 'loss_mbce', 'loss_kd', 'loss_dkd_pos', 'loss_dkd_neg', 'loss_ac',
                writer=self.writer,
                colums=['total', 'counts', 'average'],
            )
        elif self.flag_total == '00001':
            self.train_metrics = MetricTracker(
                'loss', 'loss_mbce', 'loss_kd', 'loss_dkd_pos', 'loss_dkd_neg', 'loss_ac', 'loss_coinseg_cl',
                writer=self.writer,
                colums=['total', 'counts', 'average'],
            )
        elif self.flag_total == '00010':
            self.train_metrics = MetricTracker(
                'loss', 'loss_mbce', 'loss_kd', 'loss_dkd_pos', 'loss_dkd_neg', 'loss_ac', 'loss_idec_cl',
                writer=self.writer,
                colums=['total', 'counts', 'average'],
            )
        elif self.flag_total == '00100':
            self.train_metrics = MetricTracker(
                'loss', 'loss_mbce', 'loss_kd', 'loss_dkd_pos', 'loss_dkd_neg', 'loss_ac', 'loss_sdr_cl',
                writer=self.writer,
                colums=['total', 'counts', 'average'],
            )
        elif self.flag_total == '01000':
            self.train_metrics = MetricTracker(
                'loss', 'loss_mbce', 'loss_kd', 'loss_dkd_pos', 'loss_dkd_neg', 'loss_ac', 'loss_pbcl',
                writer=self.writer,
                colums=['total', 'counts', 'average'],
            )
        elif self.flag_total == '10000':
            self.train_metrics = MetricTracker(
                'loss', 'loss_mbce', 'loss_kd', 'loss_dkd_pos', 'loss_dkd_neg', 'loss_ac', 'loss_mrkd',
                writer=self.writer,
                colums=['total', 'counts', 'average'],
            )
        elif self.flag_total == '11000':
            self.train_metrics = MetricTracker(
                'loss', 'loss_mbce', 'loss_kd', 'loss_dkd_pos', 'loss_dkd_neg', 'loss_ac', 'loss_mrkd', 'loss_pbcl',
                writer=self.writer,
                colums=['total', 'counts', 'average'],
            )

        if config.resume is not None:
            self._resume_checkpoint(config.resume, config['test'])

        if self.config['data_loader']['args']['task']['step'] > 1:
            pth_name = 'checkpoint-epoch' + str(config['trainer']['epochs']) + '.pth'
        else:
            pth_name = 'model_best.pth'
        task_name = config['data_loader']['args']['task']['setting'] + '_' + \
                    config['data_loader']['args']['task']['name'] + '_' + \
                    config['name']
        step_name = 'step_' + str(config['data_loader']['args']['task']['step'] - 1)
        path = config['trainer']['save_dir'] + '/models/' + task_name + '/' + step_name + '/' + pth_name
        self._resume_prototype(path)

        self._print_train_info()

    def _print_train_info(self):
        if self.flag_total == '00000':
            self.logger.info(
                f"Total loss = {self.config['hyperparameter']['mbce']} * L_mbce "
                f"+ {self.config['hyperparameter']['kd']} * L_kd "
                f"+ {self.config['hyperparameter']['dkd_pos']} * L_dkd_pos "
                f"+ {self.config['hyperparameter']['dkd_neg']} * L_dkd_neg "
                f"+ {self.config['hyperparameter']['ac']} * L_ac "
            )
        elif self.flag_total == '00001':
            self.logger.info(
                f"Total loss = {self.config['hyperparameter']['mbce']} * L_mbce "
                f"+ {self.config['hyperparameter']['kd']} * L_kd "
                f"+ {self.config['hyperparameter']['dkd_pos']} * L_dkd_pos "
                f"+ {self.config['hyperparameter']['dkd_neg']} * L_dkd_neg "
                f"+ {self.config['hyperparameter']['ac']} * L_ac "
                f"+ {self.config['hyperparameter']['coinseg_cl']} * L_coinseg_cl "
            )
        elif self.flag_total == '00010':
            self.logger.info(
                f"Total loss = {self.config['hyperparameter']['mbce']} * L_mbce "
                f"+ {self.config['hyperparameter']['kd']} * L_kd "
                f"+ {self.config['hyperparameter']['dkd_pos']} * L_dkd_pos "
                f"+ {self.config['hyperparameter']['dkd_neg']} * L_dkd_neg "
                f"+ {self.config['hyperparameter']['ac']} * L_ac "
                f"+ {self.config['hyperparameter']['idec_cl']} * L_idec_cl "
            )
        elif self.flag_total == '00100':
            self.logger.info(
                f"Total loss = {self.config['hyperparameter']['mbce']} * L_mbce "
                f"+ {self.config['hyperparameter']['kd']} * L_kd "
                f"+ {self.config['hyperparameter']['dkd_pos']} * L_dkd_pos "
                f"+ {self.config['hyperparameter']['dkd_neg']} * L_dkd_neg "
                f"+ {self.config['hyperparameter']['ac']} * L_ac "
                f"+ {self.config['hyperparameter']['sdr_cl']} * L_sdr_cl "
            )
        elif self.flag_total == '01000':
            self.logger.info(
                f"Total loss = {self.config['hyperparameter']['mbce']} * L_mbce "
                f"+ {self.config['hyperparameter']['kd']} * L_kd "
                f"+ {self.config['hyperparameter']['dkd_pos']} * L_dkd_pos "
                f"+ {self.config['hyperparameter']['dkd_neg']} * L_dkd_neg "
                f"+ {self.config['hyperparameter']['ac']} * L_ac "
                f"+ {self.config['hyperparameter']['pbcl']} * L_pbcl "
            )
        elif self.flag_total == '10000':
            self.logger.info(
                f"Total loss = {self.config['hyperparameter']['mbce']} * L_mbce "
                f"+ {self.config['hyperparameter']['kd']} * L_kd "
                f"+ {self.config['hyperparameter']['dkd_pos']} * L_dkd_pos "
                f"+ {self.config['hyperparameter']['dkd_neg']} * L_dkd_neg "
                f"+ {self.config['hyperparameter']['ac']} * L_ac "
                f"+ {self.config['hyperparameter']['mrkd']} * L_mrkd "
            )
        elif self.flag_total == '11000':
            self.logger.info(
                f"Total loss = {self.config['hyperparameter']['mbce']} * L_mbce "
                f"+ {self.config['hyperparameter']['kd']} * L_kd "
                f"+ {self.config['hyperparameter']['dkd_pos']} * L_dkd_pos "
                f"+ {self.config['hyperparameter']['dkd_neg']} * L_dkd_neg "
                f"+ {self.config['hyperparameter']['ac']} * L_ac "
                f"+ {self.config['hyperparameter']['mrkd']} * L_mrkd "
                f"+ {self.config['hyperparameter']['pbcl']} * L_pbcl "
            )

    """
    Training logic for an epoch
    :param epoch: Integer, current training epoch.
    :return: A log that contains average loss and metric in this epoch.
    """

    def _train_epoch(self, epoch, prototypes, count_features):
        torch.distributed.barrier()

        self.model.train()
        self.model_old.eval()

        self.train_metrics.reset()
        self.logger.info(f'Epoch - {epoch}')

        # Random shuffling
        if not isinstance(self.train_loader.sampler, torch.utils.data.RandomSampler):
            self.train_loader.sampler.set_epoch(epoch)

        for batch_idx, data in enumerate(self.train_loader):
            self.optimizer.zero_grad(set_to_none=True)
            data['image'], data['label'] = data['image'].to(self.device), data['label'].to(self.device)
            with torch.cuda.amp.autocast(enabled=self.config['use_amp']):
                logit, features, layer_features, pos_neg_features = self.model(data['image'], ret_intermediate=True)

                if self.model_old is not None:
                    with torch.no_grad():
                        logit_old, features_old, layer_features_old, pos_neg_features_old = self.model_old(
                            data['image'], ret_intermediate=True)

                # generate pseudo label using old model's output
                B, C, H, W = features.shape[0], features.shape[1], features.shape[2], features.shape[3]
                labels = data['label'].unsqueeze(dim=1)
                labels_down = (F.interpolate(input=labels.double(), size=(H, W), mode='nearest')).long()
                labels_down_bgr_mask = (labels_down == 0).long()

                threshold = self.config['hyperparameter']['threshold']
                outputs_old = logit_old.sigmoid()
                outputs_old[outputs_old[:, :, :, :] < threshold] = 0
                outputs_old = torch.argmax(outputs_old, dim=1, keepdim=True)
                outputs_old_down = (F.interpolate(input=outputs_old.double(), size=(H, W), mode='nearest')).long()
                pseudo_label_old_down = (outputs_old_down * labels_down_bgr_mask).long()
                pseudo_label = pseudo_label_old_down + labels_down

                prototypes, count_features = self._update_running_stats(pseudo_label,
                                                                        features,
                                                                        prototypes,
                                                                        count_features)

                # [|Ct|]
                loss_mbce = self.BCELoss(
                    logit[:, -self.n_new_classes:],  # [N, |Ct|, H, W]
                    data['label'],  # [N, H, W]
                ).mean(dim=[0, 2, 3])

                # [|C0:t-1|]
                loss_kd = self.knowledge_distillation_loss(
                    logit[:, 1:self.n_old_classes + 1],  # [N, |C0:t|, H, W]
                    logit_old[:, 1:].sigmoid()  # [N, |C0:t|, H, W]
                ).mean(dim=[0, 2, 3])

                # [1]
                loss_ac = self.ACLoss(logit[:, 0:1]).mean(dim=[0, 2, 3])

                # [|C0:t-1|]
                loss_dkd_pos = self.knowledge_distillation_loss(
                    pos_neg_features['pos_reg'][:, :self.n_old_classes],
                    pos_neg_features_old['pos_reg'].sigmoid()
                ).mean(dim=[0, 2, 3])

                # [|C0:t-1|]
                loss_dkd_neg = self.knowledge_distillation_loss(
                    pos_neg_features['neg_reg'][:, :self.n_old_classes],
                    pos_neg_features_old['neg_reg'].sigmoid()
                ).mean(dim=[0, 2, 3])

                if self.flag_total == '00000':
                    loss = self.config['hyperparameter']['mbce'] * loss_mbce.sum() + \
                           self.config['hyperparameter']['kd'] * loss_kd.sum() + \
                           self.config['hyperparameter']['dkd_pos'] * loss_dkd_pos.sum() + \
                           self.config['hyperparameter']['dkd_neg'] * loss_dkd_neg.sum() + \
                           self.config['hyperparameter']['ac'] * loss_ac.sum()
                elif self.flag_total == '00001':
                    loss_coinseg_cl = self.coinseg_contrastive_loss(data['label'],
                                                                    features_old,
                                                                    features,
                                                                    logit_old,
                                                                    logit,
                                                                    prototypes,
                                                                    self.total_classes,
                                                                    self.n_old_classes,
                                                                    self.n_new_classes,
                                                                    epoch,
                                                                    batch_idx,
                                                                    self.len_epoch)
                    loss = self.config['hyperparameter']['mbce'] * loss_mbce.sum() + \
                           self.config['hyperparameter']['kd'] * loss_kd.sum() + \
                           self.config['hyperparameter']['dkd_pos'] * loss_dkd_pos.sum() + \
                           self.config['hyperparameter']['dkd_neg'] * loss_dkd_neg.sum() + \
                           self.config['hyperparameter']['ac'] * loss_ac.sum() + \
                           self.config['hyperparameter']['coinseg_cl'] * loss_coinseg_cl.sum()
                elif self.flag_total == '00010':
                    loss_idec_cl = self.idec_contrastive_loss(data['label'],
                                                              features_old,
                                                              features,
                                                              logit_old,
                                                              logit,
                                                              prototypes,
                                                              self.total_classes,
                                                              self.n_old_classes,
                                                              self.n_new_classes,
                                                              epoch,
                                                              batch_idx,
                                                              self.len_epoch)
                    loss = self.config['hyperparameter']['mbce'] * loss_mbce.sum() + \
                           self.config['hyperparameter']['kd'] * loss_kd.sum() + \
                           self.config['hyperparameter']['dkd_pos'] * loss_dkd_pos.sum() + \
                           self.config['hyperparameter']['dkd_neg'] * loss_dkd_neg.sum() + \
                           self.config['hyperparameter']['ac'] * loss_ac.sum() + \
                           self.config['hyperparameter']['idec_cl'] * loss_idec_cl.sum()
                elif self.flag_total == '00100':
                    loss_sdr_cl = self.sdr_contrastive_loss(data['label'],
                                                            features_old,
                                                            features,
                                                            logit_old,
                                                            logit,
                                                            prototypes,
                                                            self.total_classes,
                                                            self.n_old_classes,
                                                            self.n_new_classes,
                                                            epoch,
                                                            batch_idx,
                                                            self.len_epoch)
                    loss = self.config['hyperparameter']['mbce'] * loss_mbce.sum() + \
                           self.config['hyperparameter']['kd'] * loss_kd.sum() + \
                           self.config['hyperparameter']['dkd_pos'] * loss_dkd_pos.sum() + \
                           self.config['hyperparameter']['dkd_neg'] * loss_dkd_neg.sum() + \
                           self.config['hyperparameter']['ac'] * loss_ac.sum() + \
                           self.config['hyperparameter']['sdr_cl'] * loss_sdr_cl.sum()
                elif self.flag_total == '01000':
                    loss_pbcl = self.prototype_balanced_contrastive_loss(pseudo_label_old_down,
                                                                         pseudo_label,
                                                                         features_old,
                                                                         features,
                                                                         prototypes,
                                                                         self.total_classes,
                                                                         self.n_old_classes)
                    loss = self.config['hyperparameter']['mbce'] * loss_mbce.sum() + \
                           self.config['hyperparameter']['kd'] * loss_kd.sum() + \
                           self.config['hyperparameter']['dkd_pos'] * loss_dkd_pos.sum() + \
                           self.config['hyperparameter']['dkd_neg'] * loss_dkd_neg.sum() + \
                           self.config['hyperparameter']['ac'] * loss_ac.sum() + \
                           self.config['hyperparameter']['pbcl'] * loss_pbcl.sum()
                elif self.flag_total == '10000':
                    loss_mrkd = self.multi_scale_region_distillation_loss(pseudo_label,
                                                                          layer_features_old,
                                                                          layer_features,
                                                                          self.total_classes,
                                                                          self.n_old_classes)
                    loss = self.config['hyperparameter']['mbce'] * loss_mbce.sum() + \
                           self.config['hyperparameter']['kd'] * loss_kd.sum() + \
                           self.config['hyperparameter']['dkd_pos'] * loss_dkd_pos.sum() + \
                           self.config['hyperparameter']['dkd_neg'] * loss_dkd_neg.sum() + \
                           self.config['hyperparameter']['ac'] * loss_ac.sum() + \
                           self.config['hyperparameter']['mrkd'] * loss_mrkd.sum()
                elif self.flag_total == '11000':
                    loss_mrkd = self.multi_scale_region_distillation_loss(pseudo_label,
                                                                          layer_features_old,
                                                                          layer_features,
                                                                          self.total_classes,
                                                                          self.n_old_classes)
                    loss_pbcl = self.prototype_balanced_contrastive_loss(pseudo_label_old_down,
                                                                         pseudo_label,
                                                                         features_old,
                                                                         features,
                                                                         prototypes,
                                                                         self.total_classes,
                                                                         self.n_old_classes)
                    loss = self.config['hyperparameter']['mbce'] * loss_mbce.sum() + \
                           self.config['hyperparameter']['kd'] * loss_kd.sum() + \
                           self.config['hyperparameter']['dkd_pos'] * loss_dkd_pos.sum() + \
                           self.config['hyperparameter']['dkd_neg'] * loss_dkd_neg.sum() + \
                           self.config['hyperparameter']['ac'] * loss_ac.sum() + \
                           self.config['hyperparameter']['mrkd'] * loss_mrkd.sum() + \
                           self.config['hyperparameter']['pbcl'] * loss_pbcl.sum()

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.optimizer.zero_grad(set_to_none=True)

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            self.train_metrics.update('loss_mbce', loss_mbce.sum().item())
            self.train_metrics.update('loss_kd', loss_kd.sum().item())
            self.train_metrics.update('loss_ac', loss_ac.sum().item())
            self.train_metrics.update('loss_dkd_pos', loss_dkd_pos.sum().item())
            self.train_metrics.update('loss_dkd_neg', loss_dkd_neg.sum().item())
            if self.config['trainer']['flag_MRKD']:
                self.train_metrics.update('loss_mrkd', loss_mrkd.sum().item())
            if self.config['trainer']['flag_PBCL']:
                self.train_metrics.update('loss_pbcl', loss_pbcl.sum().item())
            if self.config['trainer']['flag_SDR_CL']:
                self.train_metrics.update('loss_sdr_cl', loss_sdr_cl.sum().item())
            if self.config['trainer']['flag_IDEC_CL']:
                self.train_metrics.update('loss_idec_cl', loss_idec_cl.sum().item())
            if self.config['trainer']['flag_COINSEG_CL']:
                self.train_metrics.update('loss_coinseg_cl', loss_coinseg_cl.sum().item())

            # Get First lr
            if batch_idx == 0:
                self.writer.add_scalars('lr', {'lr': self.optimizer.param_groups[0]['lr']}, epoch - 1)
                self.logger.info(
                    f"lr[0]: {self.optimizer.param_groups[0]['lr']:.6f} / lr[1]: {self.optimizer.param_groups[1]['lr']:.6f} / lr[2]: {self.optimizer.param_groups[2]['lr']:.6f}")

            #if self.lr_scheduler is not None:
            #    self.lr_scheduler.step()

            self.progress(self.logger, batch_idx, len(self.train_loader))

            if batch_idx == self.len_epoch:
                break

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # average train loss per epoch
        log = self.train_metrics.result()

        val_flag = False
        if self.do_validation and (epoch % self.validation_period) == 0:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
            if self.rank == 0:
                val_flag = True

        return log, val_flag, prototypes, count_features