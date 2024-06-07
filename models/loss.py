import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
from tsnecuda import TSNE
from MulticoreTSNE import MulticoreTSNE
import os
import random
import einops


class BCELoss(nn.Module):
    def __init__(self, ignore_index=255, ignore_bg=True, pos_weight=None, reduction='mean'):
        super().__init__()
        self.ignore_index = ignore_index
        self.pos_weight = pos_weight
        self.reduction = reduction

        if ignore_bg is True:
            self.ignore_indexes = [0, self.ignore_index]
        else:
            self.ignore_indexes = [self.ignore_index]

    def forward(self, logit, label, logit_old=None):
        # logit:     [N, C_tot, H, W]
        # logit_old: [N, C_old, H, W]
        # label:     [N, H, W] or [N, C, H, W]
        C = logit.shape[1]
        if logit_old is None:
            if len(label.shape) == 3:
                # target: [N, C, H, W]
                target = torch.zeros_like(logit).float().to(logit.device)
                for cls_idx in label.unique():
                    if cls_idx in self.ignore_indexes:
                        continue
                    target[:, int(cls_idx)] = (label == int(cls_idx)).float()
            elif len(label.shape) == 4:
                target = label
            else:
                raise NotImplementedError
            
            logit = logit.permute(0, 2, 3, 1).reshape(-1, C)
            target = target.permute(0, 2, 3, 1).reshape(-1, C)

            return nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction=self.reduction)(logit, target)
        else:
            if len(label.shape) == 3:
                # target: [N, C, H, W]
                target = torch.zeros_like(logit).float().to(logit.device)
                target[:, 1:logit_old.shape[1]] = logit_old.sigmoid()[:, 1:]
                for cls_idx in label.unique():
                    if cls_idx in self.ignore_indexes:
                        continue
                    target[:, int(cls_idx)] = (label == int(cls_idx)).float()
            else:
                raise NotImplementedError
            
            loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction=self.reduction)(logit, target)
            del target

            return loss


class WBCELoss(nn.Module):
    def __init__(self, ignore_index=255, pos_weight=None, reduction='none', n_old_classes=0, n_new_classes=0):
        super().__init__()
        self.ignore_index = ignore_index
        self.n_old_classes = n_old_classes  # |C0:t-1| + 1(bg), 19-1: 20 | 15-5: 16 | 15-1: 16...
        self.n_new_classes = n_new_classes  # |Ct|, 19-1: 1 | 15-5: 5 | 15-1: 1
        
        self.reduction = reduction
        # pos_weight为每个类别赋予的权重值
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=self.reduction)
        
    def forward(self, logit, label):
        # logit:     [N, |Ct|, H, W]
        # label:     [N, H, W]

        N, C, H, W = logit.shape
        target = torch.zeros_like(logit, device=logit.device).float()
        for cls_idx in label.unique():
            if cls_idx in [0, self.ignore_index]:
                continue
            target[:, int(cls_idx) - self.n_old_classes] = (label == int(cls_idx)).float()
        
        loss = self.criterion(
            logit.permute(0, 2, 3, 1).reshape(-1, C),
            target.permute(0, 2, 3, 1).reshape(-1, C)
        )

        if self.reduction == 'none':
            return loss.reshape(N, H, W, C).permute(0, 3, 1, 2)  # [N, C, H, W]
        elif self.reduction == 'mean':
            return loss
        else:
            raise NotImplementedError


class KDLoss(nn.Module):
    def __init__(self, pos_weight=None, reduction='mean'):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)

    def forward(self, logit, logit_old=None):
        # logit:     [N, |Ct|, H, W]
        # logit_old: [N, |Ct|, H, W]
        
        N, C, H, W = logit.shape
        loss = self.criterion(
            logit.permute(0, 2, 3, 1).reshape(-1, C),
            logit_old.permute(0, 2, 3, 1).reshape(-1, C)
        ).reshape(N, H, W, C).permute(0, 3, 1, 2)
        return loss


class ACLoss(nn.Module):
    def __init__(self, reduction='none'):
        super().__init__()
        self.reduction = reduction
        self.criterion = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, logit):
        # logit: [N, 1, H, W]
        
        return self.criterion(logit, torch.zeros_like(logit))
        # loss = -torch.log(1 - logit.sigmoid())


# mse loss for kd
class Multi_Scale_Region_Distillation_Loss_1(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.loss_fn = nn.MSELoss()

    def forward(self, labels, outputs_old, layer_features_old, layer_features, num_class, num_old_class):
        loss = torch.tensor(0., device=self.device)
        labels = labels.unsqueeze(dim=1)
        # generate pseudo label using old model's output
        threshold = self.config['hyperparameter']['threshold']
        outputs_old[outputs_old[:, :, :, :] < threshold] = 0
        outputs_old = torch.argmax(outputs_old, dim=1, keepdim=True)
        weight = torch.tensor([1, 2, 3, 4]).to(self.device)

        for idx in range(len(layer_features)):
            B, C, H, W = layer_features[idx].shape[0], layer_features[idx].shape[1], layer_features[idx].shape[2], layer_features[idx].shape[3]
            labels_down = (F.interpolate(input=labels.double(), size=(H, W), mode='nearest')).long()
            labels_down_bgr_mask = (labels_down == 0).long()

            outputs_old_down = (F.interpolate(input=outputs_old.double(), size=(H, W), mode='nearest')).long()
            pseudo_label_old_down = (outputs_old_down * labels_down_bgr_mask).long()
            pseudo_label = pseudo_label_old_down + labels_down

            cl_present = torch.unique(input=pseudo_label).long()
            if cl_present[-1] == 255:
                cl_present = cl_present[:-1]

            for cl in cl_present:
                feat = layer_features[idx][(pseudo_label== cl).expand(-1, C, -1, -1)]
                feat_old = layer_features_old[idx][(pseudo_label == cl).expand(-1, C, -1, -1)]
                if cl == 0:
                    dis = num_old_class / num_class * self.loss_fn(feat, feat_old)
                elif cl <= num_old_class:
                    dis = self.loss_fn(feat, feat_old)
                else:
                    dis = torch.tensor(0., device=self.device)

                loss += weight[idx] * dis

        loss /= len(layer_features)

        return loss


# cos distance for kd
class Multi_Scale_Region_Distillation_Loss(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.loss_fn = cos_distance()

    def forward(self, pseudo_labels, layer_features_old, layer_features, num_class, num_old_class):
        loss = torch.tensor(0., device=self.device)
        weight = torch.tensor([1, 2, 3, 4, 5]).to(self.device)

        for idx in range(len(layer_features)):
            B, C, H, W = layer_features[idx].shape[0], layer_features[idx].shape[1], layer_features[idx].shape[2], layer_features[idx].shape[3]
            pseudo_label= (F.interpolate(input=pseudo_labels.double(), size=(H, W), mode='nearest')).long()

            pseudo_label = pseudo_label.view(-1) #b*h*w -> bhw
            cl_present = torch.unique(input=pseudo_label).long()
            if cl_present[-1] == 255:
                cl_present = cl_present[:-1]

            feat = einops.rearrange(layer_features[idx], 'b c h w -> ( b h w ) c ')
            feat_old = einops.rearrange(layer_features_old[idx], 'b c h w -> ( b h w ) c ')

            for cl in cl_present:
                pl = (pseudo_label == cl)
                feat_cl = feat[pl]
                feat_old_cl = feat_old[pl]
                #print("feat_cl.shape: ", feat_cl.shape)
                #print("feat_old_cl.shape: ", feat_old_cl.shape)
                if cl == 0:
                    dis = num_old_class / num_class * self.loss_fn(feat_cl, feat_old_cl)
                elif cl <= num_old_class:
                    dis = self.loss_fn(feat_cl, feat_old_cl)
                else:
                    dis = torch.tensor(0., device=self.device)
                # raise ValueError("1")
                loss += weight[idx] * dis

        return loss


# kl_divergence for kd
class Multi_Scale_Region_Distillation_Loss_3(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.loss_fn = kl_divergence()

    def forward(self, pseudo_labels, layer_features_old, layer_features, num_class, num_old_class):
        loss = torch.tensor(0., device=self.device)
        weight = torch.tensor([1, 2, 3, 4, 5, 1]).to(self.device)

        for idx in range(len(layer_features)):
            B, C, H, W = layer_features[idx].shape[0], layer_features[idx].shape[1], layer_features[idx].shape[2], \
            layer_features[idx].shape[3]
            pseudo_label = (F.interpolate(input=pseudo_labels.double(), size=(H, W), mode='nearest')).long()

            pseudo_label = pseudo_label.view(-1)  # b*h*w -> bhw
            cl_present = torch.unique(input=pseudo_label).long()
            if cl_present[-1] == 255:
                cl_present = cl_present[:-1]

            feat = einops.rearrange(layer_features[idx], 'b c h w -> ( b h w ) c ')
            feat_old = einops.rearrange(layer_features_old[idx], 'b c h w -> ( b h w ) c ')

            for cl in cl_present:
                pl = (pseudo_label == cl)
                feat_cl = feat[pl]
                feat_old_cl = feat_old[pl]
                # print("feat_cl.shape: ", feat_cl.shape)
                # print("feat_old_cl.shape: ", feat_old_cl.shape)
                if cl == 0:
                    dis = num_old_class / num_class * self.loss_fn(feat_cl, feat_old_cl)
                elif cl <= num_old_class:
                    dis = self.loss_fn(feat_cl, feat_old_cl)
                else:
                    dis = torch.tensor(0., device=self.device)
                # raise ValueError("1")
                loss += weight[idx] * dis

        return loss


class cos_distance(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CosineSimilarity(dim=1)

    def forward(self, x, y):
        # print("cos_results.shape: ", self.loss(x, y).shape)
        outputs = torch.tensor(1.).cuda() - torch.mean(self.loss(x, y))

        return outputs


class kl_divergence(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, inputs, targets):
        outputs = torch.softmax(inputs, dim=1)
        outputs_log = torch.log_softmax(inputs, dim=1)
        labels_log = torch.log_softmax(targets, dim=1)

        loss = torch.sum(outputs * (outputs_log - labels_log), dim=1)
        # print("cos_results.shape: ", loss.shape)
        if self.reduction == 'mean':
            outputs = torch.mean(loss)
        else:
            outputs = torch.sum(loss)

        return outputs


# Contrastive Learning loss defined in 2021-CVPR Continual Semantic Segmentation via Repulsion-Attraction of Sparse and Disentangled Latent Representations
class SDR_Contrastive_Loss(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device

    def forward(self, labels, features_old, features, outputs_old, outputs, prototypes, num_class, num_old_class,
                    num_new_class, epoch, train_step, len_epoch):
        loss_features_clustering = torch.tensor(0., device=self.device)
        loss_separationclustering = torch.tensor(0., device=self.device)

        labels = labels.unsqueeze(dim=1)
        # 对mask进行下采样，使其保持与中间层特征相同的大小
        labels_down = (F.interpolate(input=labels.double(), size=(features.shape[2], features.shape[3]), mode='nearest')).long()
        # 挑出labels_down中的独立不重复元素，并按升序排列，例如输入：[0,2,5,6,1,5,6]，会得到[0,1,2,5,6]
        cl_present = torch.unique(input=labels_down)

        if cl_present[0] == 0:
            cl_present = cl_present[1:]

        if cl_present[-1] == 255:
            cl_present = cl_present[:-1]

        decoder_dim = self.config['arch']['args']['decoder_dim']
        features_local_mean = torch.zeros([num_class + 1, decoder_dim], device=self.device)

        for cl in cl_present:
            features_cl = features[(labels_down == cl).expand(-1, features.shape[1], -1, -1)].view(features.shape[1], -1)

            # L2 normalization of the features
            # features_cl = F.normalize(features_cl, p=2, dim=0)
            # prototypes = F.normalize(prototypes, p=2, dim=0)

            features_local_mean[cl] = torch.mean(features_cl, dim=-1)

            loss_to_use = nn.MSELoss()
            loss_features_clustering += loss_to_use(features_cl, prototypes[cl].unsqueeze(1).expand(-1, features_cl.shape[1]))

            loss_features_clustering /= (cl_present.shape[0])

        # remove zero rows
        features_local_mean_reduced = features_local_mean[cl_present, :]

        inv_pairwise_D = 1 / torch.cdist(features_local_mean_reduced.unsqueeze(dim=0), features_local_mean_reduced.unsqueeze(dim=0)).squeeze()
        loss_separationclustering_temp = inv_pairwise_D[~torch.isinf(inv_pairwise_D)].mean()
        if ~torch.isnan(loss_separationclustering_temp): 
            loss_separationclustering = loss_separationclustering_temp

        loss = loss_features_clustering + loss_separationclustering

        return loss


# Contrastive Learning loss defined in 2023-TPAMI Inherit With Distillation and Evolve With Contrast: Exploring Class Incremental Semantic Segmentation Without Exemplar Memory
class IDEC_Contrastive_Loss(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device

    '''
    Args:
        outputs: model outputs in t-step       : [b, curr_numclass, 512, 512]
        feature: embedding in t-step           : [b, 256, 128, 128]
        outputs_old: model outputs in oldious step : [b, old_numclass, 512, 512]
        feature_old: embedding in t-1 step    : [b, 256, 128, 128]
        num_classes: t-step classes number
        use_sigmoid: if True, use torch.sigmoid otherwise torch.softmax  
    return:
        contrastive loss
    '''
    def forward(self, labels, features_old, features, outputs_old, outputs, prototypes, num_class, num_old_class,
                    num_new_class, epoch, train_step, len_epoch):
        # use metric learning loss 
        criterion = nn.TripletMarginLoss(margin=1.0, p=2.0, eps=1e-06, swap=False, size_average=None, reduce=None, reduction='mean')

        B, C, H, W = features.shape[0], features.shape[1], features.shape[2], features.shape[3]
        labels = labels.unsqueeze(dim=1)
        labels_down = (F.interpolate(input=labels.double(), size=(H, W), mode='nearest')).long()
        labels_down_bgr_mask = (labels_down == 0).long()

        # generate pseudo label using old model's output
        threshold = self.config['hyperparameter']['threshold']
        outputs_old[outputs_old[:, :, :, :] < threshold] = 0
        outputs_old = torch.argmax(outputs_old, dim=1, keepdim=True)
        outputs_old_down = (F.interpolate(input=outputs_old.double(), size=(H, W), mode='nearest')).long()
        pseudo_label_old_down = (outputs_old_down * labels_down_bgr_mask).long()
        pseudo_label = labels_down + pseudo_label_old_down

        # 对每一个类别的响应进行正负样本对的构建，以t-1步模型的类别响应为anchor，以t步模型的相同类别响应构建正样本对，不同类别响应构建负样本对
        # 遍历每个类，以t-1-step model输出的结果作为anchor，t-step model输出的结果作为positive和negative
        # initilize loss value
        contrastive_loss = torch.tensor(0.).to(self.device)

        cl_present = torch.unique(input=pseudo_label).long()
        if cl_present[0] == 0:
            cl_present = cl_present[1:]
        if cl_present[-1] == 255:
            cl_present = cl_present[:-1]
        if len(cl_present) == 1:
            return contrastive_loss
        for i in cl_present:
            class_embedding_positive = features_old[(pseudo_label == i).expand(-1, features.shape[1], -1, -1)].view(-1, features.shape[1])
            class_embedding_anchor = features[(pseudo_label == i).expand(-1, features.shape[1], -1, -1)].view(-1, features.shape[1])
            for j in cl_present:
                if j == i:
                    continue
                class_embedding_negative = features[(pseudo_label == j).expand(-1, features.shape[1], -1, -1)].view(-1, features.shape[1])

                class_embedding_anchor = torch.mean(class_embedding_anchor, dim=0).unsqueeze(0)
                class_embedding_positive = torch.mean(class_embedding_positive, dim=0).unsqueeze(0)
                class_embedding_negative = torch.mean(class_embedding_negative, dim=0).unsqueeze(0)
                # print(class_embedding_anchor.shape, class_embedding_positive.shape, class_embedding_negative.shape)

                # length = min(class_embedding_anchor.shape[0], class_embedding_positive.shape[0], class_embedding_negative.shape[0])

                # nn.TripletMarginLoss()输入是anchor, positive, negative三个B*N的张量（表示Batchsize个N为的特征向量），输出triplet loss的值。
                # contrastive_loss += criterion(class_embedding_anchor[:length], class_embedding_positive[:length], class_embedding_negative[:length])
                contrastive_loss += criterion(class_embedding_anchor, class_embedding_positive, class_embedding_negative)
                # print(len(cl_present), criterion(class_embedding_anchor, class_embedding_positive, class_embedding_negative))
        contrastive_loss = contrastive_loss / len(cl_present) / (len(cl_present) - 1)

        return contrastive_loss


# Contrastive Learning loss defined in 2023-ICCV CoinSeg: Contrast Inter- and Intra- Class Representations for Incremental Segmentation
class COINSEG_Contrastive_Loss(nn.Module):
    def __init__(self, config, device, temperature=0.07, use_pseudo_label=True):
        super().__init__()
        self.config = config
        self.device = device
        self.temperature = temperature

        self.use_pseudo_label = use_pseudo_label
        if self.use_pseudo_label:
            self.num = self.config['data_loader']['args']['class_num']

    '''
    Args:
        outputs: model outputs in t-step       : [b, curr_numclass, 512, 512]
        feature: embedding in t-step           : [b, 256, 128, 128]
        outputs_old: model outputs in oldious step : [b, old_numclass, 512, 512]
        feature_old: embedding in t-1 step    : [b, 256, 128, 128]
        num_classes: t-step classes number
        use_sigmoid: if True, use torch.sigmoid otherwise torch.softmax  
    return:
        contrastive loss
    '''
    def forward(self, labels, features_old, features, outputs_old, outputs, prototypes, num_class, num_old_class,
                num_new_class, epoch, train_step, len_epoch):
        B, C, H, W = features.shape
        labels = labels.unsqueeze(dim=1)
        labels_down = (F.interpolate(input=labels.double(), size=(H, W), mode='nearest')).long()
        labels_down_bgr_mask = (labels_down == 0).long()
        # generate pseudo label using old model's output
        threshold = self.config['hyperparameter']['threshold']
        outputs_old[outputs_old[:, :, :, :] < threshold] = 0
        outputs_old = torch.argmax(outputs_old, dim=1, keepdim=True)
        outputs_old_down = (F.interpolate(input=outputs_old.double(), size=(H, W), mode='nearest')).long()
        pseudo_label_old_down = (outputs_old_down * labels_down_bgr_mask).long()
        pseudo_label = pseudo_label_old_down + labels_down
        labels = pseudo_label
        labels = (F.interpolate(input=labels.float(), size=(H, W), mode='nearest')).to(torch.long).squeeze(1)
        labels = labels.view(-1)  # bhw
        # mask without dummy labels
        mask_undummy = (labels >= 0) & (labels < self.num)
        features = einops.rearrange(features, 'b c h w -> ( b h w ) c ')
        features_old = einops.rearrange(features_old, 'b c h w -> ( b h w ) c ')
        feature_anc = F.normalize(features[mask_undummy], dim=1)
        label_anc = labels[mask_undummy]
        conts = F.normalize(features_old[mask_undummy], dim=1)

        # get contrastive probablity mask with old logits
        # prev_prob = torch.softmax(outputs_old, dim=1)
        # feature_prev_anc = prev_prob[mask_undummy]
        # feature_prev_con = torch.cat([feature_prev_anc, feature_prev_anc], dim=0)

        anc_uni = torch.unique(label_anc)
        con_uni = torch.unique(label_anc)
        anc_prototype = []
        for i in anc_uni:
            mask_i = (label_anc == i)
            proto = feature_anc[mask_i].mean(dim=0)
            anc_prototype.append(proto)
        anc_prototype = torch.stack(anc_prototype, dim=0)
        con_prototype_prev = []
        for i in con_uni:
            mask_i = (label_anc == i)
            proto = conts[mask_i].mean(dim=0)
            con_prototype_prev.append(proto)
        con_prototype_prev = torch.stack(con_prototype_prev, dim=0)
        con_prototype = torch.cat([anc_prototype, con_prototype_prev], dim=0).detach()
        con_uni = torch.cat([anc_uni, con_uni], dim=0).detach()

        anchor_features = anc_prototype
        contrast_feature = con_prototype
        anchor_labels = anc_uni
        contrast_labels = con_uni

        anchor_labels = anchor_labels.view(-1, 1)  # n 1
        contrast_labels = contrast_labels.view(-1, 1)  # 2*n 1

        batch_size = anchor_features.shape[0]  # b
        R = torch.eq(anchor_labels, contrast_labels.T).float().requires_grad_(False).to(self.device)
        positive_mask = R.clone().requires_grad_(False)
        positive_mask[:, :batch_size] -= torch.eye(batch_size).to(self.device)

        positive_mask = positive_mask.detach()
        negative_mask = 1 - R
        negative_mask = negative_mask.detach()

        anchor_dot_contrast = torch.div(torch.mm(anchor_features, contrast_feature.T), self.temperature)

        neg_contrast = (torch.exp(anchor_dot_contrast) * negative_mask).sum(dim=1, keepdim=True)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        anchor_dot_contrast = anchor_dot_contrast - logits_max.detach()

        pos_contrast = torch.log(torch.exp(anchor_dot_contrast)) * positive_mask - torch.log(torch.exp(anchor_dot_contrast) + neg_contrast) * positive_mask

        num = positive_mask.sum(dim=1)
        loss = -torch.div(pos_contrast.sum(dim=1)[num != 0], num[num != 0])

        return loss.mean()


# old
class Prototype_Balanced_Contrastive_Loss_old(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device

    def forward(self, labels, features_old, features, outputs_old, prototypes, num_class, num_old_class):
        B, C, H, W = features.shape[0], features.shape[1], features.shape[2], features.shape[3]
        labels = labels.unsqueeze(dim=1)
        labels_down = (F.interpolate(input=labels.double(), size=(H, W), mode='nearest')).long()
        labels_down_bgr_mask = (labels_down == 0).long()

        # generate pseudo label using old model's output
        threshold = self.config['hyperparameter']['threshold']
        outputs_old[outputs_old[:, :, :, :] < threshold] = 0
        outputs_old = torch.argmax(outputs_old, dim=1, keepdim=True)
        outputs_old_down = (F.interpolate(input=outputs_old.double(), size=(H, W), mode='nearest')).long()
        pseudo_label_old_down = (outputs_old_down * labels_down_bgr_mask).long()

        temperature = self.config['hyperparameter']['temperature']
        decoder_dim = self.config['arch']['args']['decoder_dim']

        class_prototypes_teacher = torch.zeros([num_class + 1, B + 1, decoder_dim]).to(self.device)
        cnt_class_teacher = torch.zeros([num_class + 1]).long().to(self.device)
        for bs in range(B):
            cl_present = torch.unique(input=pseudo_label_old_down[bs]).long()
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
                features_cl = features_old[bs][(pseudo_label_old_down[bs] == cl).expand(C, -1, -1)].view(C, -1)
                # [dim]
                class_prototypes_teacher[cl, cnt_class_teacher[cl], :] = F.normalize(
                    torch.mean(features_cl, dim=-1).float(), p=2, dim=0)
                cnt_class_teacher[cl] += 1

        for i in range(1, num_class + 1):
            class_prototypes_teacher[i, cnt_class_teacher[i], :] = F.normalize(prototypes[i].detach(), p=2, dim=0)
            cnt_class_teacher[i] += 1

        pseudo_label = pseudo_label_old_down + labels_down
        class_prototypes_student = torch.zeros([num_class + 1, B + 1, decoder_dim]).to(self.device)
        cnt_class_student = torch.zeros([num_class + 1]).long().to(self.device)
        for bs in range(B):
            cl_present = torch.unique(input=pseudo_label[bs]).long()
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
                features_cl = features[bs][(pseudo_label[bs] == cl).expand(C, -1, -1)].view(C, -1)
                # [dim]
                class_prototypes_student[cl, cnt_class_student[cl], :] = F.normalize(
                    torch.mean(features_cl, dim=-1).float(), p=2, dim=0)
                cnt_class_student[cl] += 1

        for i in range(1, num_class + 1):
            class_prototypes_student[i, cnt_class_student[i], :] = F.normalize(prototypes[i].detach(), p=2, dim=0)
            cnt_class_student[i] += 1

        loss = torch.tensor(0.).to(self.device)
        cnt_class_old_exist = torch.tensor(0.).to(self.device)
        for i in range(1, num_old_class + 1):
            if cnt_class_student[i] == 1:
                continue
            for j in range(cnt_class_student[i] - 1):
                feature_student = class_prototypes_student[i, j, :]
                similarity_neg = torch.zeros([num_class + 1]).to(self.device)
                for m in range(1, num_class + 1):
                    if m == i:
                        continue
                    for n in range(cnt_class_student[m]):
                        neg = class_prototypes_student[m, n, :]
                        similarity_neg[m] += torch.exp(torch.dot(feature_student, neg) / temperature) / \
                                             cnt_class_student[m]
                for k in range(cnt_class_teacher[i]):
                    feature_teacher = class_prototypes_teacher[i, k, :]
                    similarity_pos = torch.exp(torch.dot(feature_teacher, feature_student) / temperature)

                    numerator = similarity_pos
                    denominator = torch.sum(similarity_neg) + numerator

                    loss += -1.0 * torch.log(numerator / denominator) / cnt_class_teacher[i] / (
                                cnt_class_student[i] - 1)

            cnt_class_old_exist += 1

        if cnt_class_old_exist != 0:
            loss /= cnt_class_old_exist

        return loss


# old + new
class Prototype_Balanced_Contrastive_Loss_old_new(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device

    def forward(self, labels, features_old, features, outputs_old, prototypes, num_class, num_old_class):
        B, C, H, W = features.shape[0], features.shape[1], features.shape[2], features.shape[3]
        labels = labels.unsqueeze(dim=1)
        labels_down = (F.interpolate(input=labels.double(), size=(H, W), mode='nearest')).long()
        labels_down_bgr_mask = (labels_down == 0).long()

        # generate pseudo label using old model's output
        threshold = self.config['hyperparameter']['threshold']
        outputs_old[outputs_old[:, :, :, :] < threshold] = 0
        outputs_old = torch.argmax(outputs_old, dim=1, keepdim=True)
        outputs_old_down = (F.interpolate(input=outputs_old.double(), size=(H, W), mode='nearest')).long()
        pseudo_label_old_down = (outputs_old_down * labels_down_bgr_mask).long()

        temperature = self.config['hyperparameter']['temperature']
        decoder_dim = self.config['arch']['args']['decoder_dim']

        class_prototypes_teacher = torch.zeros([num_class + 1, B + 1, decoder_dim]).to(self.device)
        cnt_class_teacher = torch.zeros([num_class + 1]).long().to(self.device)
        for bs in range(B):
            cl_present = torch.unique(input=pseudo_label_old_down[bs]).long()
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
                features_cl = features_old[bs][(pseudo_label_old_down[bs] == cl).expand(C, -1, -1)].view(C, -1)
                # [dim]
                class_prototypes_teacher[cl, cnt_class_teacher[cl], :] = F.normalize(
                    torch.mean(features_cl, dim=-1).float(), p=2, dim=0)
                cnt_class_teacher[cl] += 1

        for i in range(1, num_class + 1):
            class_prototypes_teacher[i, cnt_class_teacher[i], :] = F.normalize(prototypes[i].detach(), p=2, dim=0)
            cnt_class_teacher[i] += 1

        pseudo_label = pseudo_label_old_down + labels_down
        class_prototypes_student = torch.zeros([num_class + 1, B + 1, decoder_dim]).to(self.device)
        cnt_class_student = torch.zeros([num_class + 1]).long().to(self.device)
        for bs in range(B):
            cl_present = torch.unique(input=pseudo_label[bs]).long()
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
                features_cl = features[bs][(pseudo_label[bs] == cl).expand(C, -1, -1)].view(C, -1)
                # [dim]
                class_prototypes_student[cl, cnt_class_student[cl], :] = F.normalize(
                    torch.mean(features_cl, dim=-1).float(), p=2, dim=0)
                cnt_class_student[cl] += 1

        for i in range(1, num_class + 1):
            class_prototypes_student[i, cnt_class_student[i], :] = F.normalize(prototypes[i].detach(), p=2, dim=0)
            cnt_class_student[i] += 1

        loss_old = torch.tensor(0.).to(self.device)
        cnt_class_old_exist = torch.tensor(0.).to(self.device)
        for i in range(1, num_old_class + 1):
            if cnt_class_student[i] == 1:
                continue
            for j in range(cnt_class_student[i] - 1):
                feature_student = class_prototypes_student[i, j, :]
                similarity_neg = torch.zeros([num_class + 1]).to(self.device)
                for m in range(1, num_class + 1):
                    for n in range(cnt_class_student[m]):
                        neg = class_prototypes_student[m, n, :]
                        similarity_neg[m] += torch.exp(torch.dot(feature_student, neg) / temperature) / cnt_class_student[m]
                for k in range(cnt_class_teacher[i]):
                    feature_teacher = class_prototypes_teacher[i, k, :]
                    similarity_pos = torch.exp(torch.dot(feature_teacher, feature_student) / temperature)

                    numerator = similarity_pos
                    denominator = torch.sum(similarity_neg)

                    loss_old += -1.0 * torch.log(numerator / denominator) / cnt_class_teacher[i] / (cnt_class_student[i] - 1)

            cnt_class_old_exist += 1

        if cnt_class_old_exist != 0:
            loss_old /= cnt_class_old_exist

        loss_new = torch.tensor(0.).to(self.device)
        cnt_class_new_exist = torch.tensor(0.).to(self.device)
        for i in range(num_old_class + 1, num_class + 1):
            if cnt_class_student[i] == 1:
                continue
            for j in range(cnt_class_student[i] - 1):
                anchor = class_prototypes_student[i, j, :]
                similarity_anchor_neg = torch.zeros([num_class + 1]).to(self.device)
                for m in range(1, num_class + 1):
                    for n in range(cnt_class_student[m]):
                        neg = class_prototypes_student[m, n, :]
                        similarity_anchor_neg[m] += torch.exp(torch.dot(anchor, neg) / temperature) / cnt_class_student[m]
                for k in range(cnt_class_student[i]):
                    if j == k:
                        continue
                    pos = class_prototypes_student[i, k, :]
                    similarity_anchor_pos = torch.exp(torch.dot(anchor, pos) / temperature)

                    numerator = similarity_anchor_pos
                    denominator = torch.sum(similarity_anchor_neg)

                    loss_new += -1.0 * torch.log(numerator / denominator) / (cnt_class_student[i] - 1) / (cnt_class_student[i] - 1)

            cnt_class_new_exist += 1

        loss_new /= cnt_class_new_exist
        # print(loss_old, loss_new)
        loss = loss_old +  loss_new

        return loss


class Prototype_Balanced_Contrastive_Loss(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device

    def forward(self, pseudo_label_old_down, pseudo_label, features_old, features, prototypes, num_class, num_old_class):
        B, C, H, W = features.shape[0], features.shape[1], features.shape[2], features.shape[3]
        pseudo_label_old_down = (F.interpolate(input=pseudo_label_old_down.double(), size=(H, W), mode='nearest')).long()

        temperature = self.config['hyperparameter']['temperature']
        decoder_dim = self.config['arch']['args']['decoder_dim']

        class_prototypes_teacher = torch.zeros([num_class + 1, B + 1, decoder_dim]).to(self.device)
        cnt_class_teacher = torch.zeros([num_class + 1]).long().to(self.device)
        for bs in range(B):
            cl_present = torch.unique(input=pseudo_label_old_down[bs]).long()
            if cl_present[0] == 0:
                cl_present = cl_present[1:]
            if len(cl_present) == 0:
                continue
            if cl_present[-1] == 255:
                cl_present = cl_present[:-1]
            if len(cl_present) == 0:
                continue

            _pl_old_down = pseudo_label_old_down[bs].view(-1) # b*h*w -> bhw
            feat_old = einops.rearrange(features_old[bs], 'c h w -> ( h w ) c ')

            for cl in cl_present:
                pl = (_pl_old_down == cl)
                features_cl = feat_old[pl].mean(dim=0)
                class_prototypes_teacher[cl, cnt_class_teacher[cl], :] = F.normalize(features_cl, p=2, dim=0)
                cnt_class_teacher[cl] += 1

        for i in range(1, num_class + 1):
            class_prototypes_teacher[i, cnt_class_teacher[i], :] = F.normalize(prototypes[i].detach(), p=2, dim=0)
            cnt_class_teacher[i] += 1

        pseudo_label = (F.interpolate(input=pseudo_label.double(), size=(H, W), mode='nearest')).long()
        class_prototypes_student = torch.zeros([num_class + 1, B + 1, decoder_dim]).to(self.device)
        cnt_class_student = torch.zeros([num_class + 1]).long().to(self.device)
        for bs in range(B):
            cl_present = torch.unique(input=pseudo_label[bs]).long()
            if cl_present[0] == 0:
                cl_present = cl_present[1:]
            if len(cl_present) == 0:
                continue
            if cl_present[-1] == 255:
                cl_present = cl_present[:-1]
            if len(cl_present) == 0:
                continue

            _pl = pseudo_label[bs].view(-1)  # b*h*w -> bhw
            feat = einops.rearrange(features[bs], 'c h w -> ( h w ) c ')

            for cl in cl_present:
                pl = (_pl == cl)
                features_cl = feat[pl].mean(dim=0)
                class_prototypes_student[cl, cnt_class_student[cl], :] = F.normalize(features_cl, p=2, dim=0)
                cnt_class_student[cl] += 1

        for i in range(1, num_class + 1):
            class_prototypes_student[i, cnt_class_student[i], :] = F.normalize(prototypes[i].detach(), p=2, dim=0)
            cnt_class_student[i] += 1

        loss_old = torch.tensor(0.).to(self.device)
        cnt_class_old_exist = torch.tensor(0.).to(self.device)
        for i in range(1, num_old_class + 1):
            if cnt_class_student[i] == 1:
                continue
            for j in range(cnt_class_student[i] - 1):
                feature_student = class_prototypes_student[i, j, :]
                similarity_neg = torch.zeros([num_class + 1]).to(self.device)
                for m in range(1, num_class + 1):
                    for n in range(cnt_class_student[m]):
                        neg = class_prototypes_student[m, n, :]
                        similarity_neg[m] += torch.exp(torch.dot(feature_student, neg) / temperature) / cnt_class_student[m]
                for k in range(cnt_class_teacher[i]):
                    feature_teacher = class_prototypes_teacher[i, k, :]
                    similarity_pos = torch.exp(torch.dot(feature_teacher, feature_student) / temperature)

                    numerator = similarity_pos
                    denominator = torch.sum(similarity_neg)

                    loss_old += -1.0 * torch.log(numerator / denominator) / cnt_class_teacher[i] / (cnt_class_student[i] - 1)

            cnt_class_old_exist += 1

        if cnt_class_old_exist != 0:
             loss_old /= cnt_class_old_exist


        loss_new = torch.tensor(0.).to(self.device)
        cnt_class_new_exist = torch.tensor(0.).to(self.device)
        for i in range(num_old_class + 1, num_class + 1):
            if cnt_class_student[i] == 1:
                continue
            for j in range(cnt_class_student[i] - 1):
                anchor = class_prototypes_student[i, j, :]
                similarity_anchor_neg = torch.zeros([num_class + 1]).to(self.device)
                for m in range(1, num_class + 1):
                    for n in range(cnt_class_student[m]):
                        neg = class_prototypes_student[m, n, :]
                        similarity_anchor_neg[m] += torch.exp(torch.dot(anchor, neg) / temperature) / cnt_class_student[m]
                for k in range(cnt_class_student[i]):
                    if j == k:
                        continue
                    pos = class_prototypes_student[i, k, :]
                    similarity_anchor_pos = torch.exp(torch.dot(anchor, pos) / temperature)

                    numerator = similarity_anchor_pos
                    denominator = torch.sum(similarity_anchor_neg)

                    loss_new += -1.0 * torch.log(numerator / denominator) / (cnt_class_student[i] - 1) / (cnt_class_student[i] - 1)

            cnt_class_new_exist += 1

        loss_new /= cnt_class_new_exist
        # print(loss_old, loss_new)
        loss = loss_old + 0.1 * loss_new


        return loss


# base_new + old + new
class Prototype_Balanced_Contrastive_Loss_base_new_old_new(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device

    def forward_base(self, labels, features_old, features, outputs_old, outputs, prototypes, num_class, num_old_class,
                num_new_class, epoch, train_step, len_epoch):
        B, C, H, W = features.shape[0], features.shape[1], features.shape[2], features.shape[3]
        labels = labels.unsqueeze(dim=1)
        labels_down = (F.interpolate(input=labels.double(), size=(H, W), mode='nearest')).long()

        temperature = self.config['hyperparameter']['temperature']
        decoder_dim = self.config['arch']['args']['decoder_dim']

        class_prototypes = torch.zeros([num_class + 1, B + 1, decoder_dim]).to(self.device)
        cnt_class = torch.zeros([num_class + 1]).long().to(self.device)
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
                class_prototypes[cl, cnt_class[cl], :] = F.normalize(torch.mean(features_cl, dim=-1).float(), p=2, dim=0)
                cnt_class[cl] += 1

        for i in range(1, num_class + 1):
            class_prototypes[i, cnt_class[i], :] = F.normalize(prototypes[i].detach(), p=2, dim=0)
            cnt_class[i] += 1

        loss = torch.tensor(0.).to(self.device)
        cnt_class_exist = torch.tensor(0.).to(self.device)
        for i in range(1, num_class + 1):
            if cnt_class[i] == 1:
                continue
            for j in range(cnt_class[i] - 1):
                anchor = class_prototypes[i, j, :]
                similarity_anchor_neg = torch.zeros([num_class + 1]).to(self.device)
                for m in range(1, num_class + 1):
                    for n in range(cnt_class[m]):
                        neg = class_prototypes[m, n, :]
                        similarity_anchor_neg[m] += torch.exp(torch.dot(anchor, neg) / temperature) / cnt_class[m]
                for k in range(cnt_class[i]):
                    if j == k:
                        continue
                    pos = class_prototypes[i, k, :]
                    similarity_anchor_pos = torch.exp(torch.dot(anchor, pos) / temperature)

                    numerator = similarity_anchor_pos
                    denominator = torch.sum(similarity_anchor_neg)

                    loss += -1.0 * torch.log(numerator / denominator) / (cnt_class[i] - 1) / (cnt_class[i] - 1)

            cnt_class_exist += 1

        loss = 0.1 * loss / cnt_class_exist

        return loss

    def forward_incremental(self, labels, features_old, features, outputs_old, outputs, prototypes, num_class, num_old_class,
                num_new_class, epoch, train_step, len_epoch):
        B, C, H, W = features.shape[0], features.shape[1], features.shape[2], features.shape[3]
        labels = labels.unsqueeze(dim=1)
        labels_down = (F.interpolate(input=labels.double(), size=(H, W), mode='nearest')).long()
        labels_down_bgr_mask = (labels_down == 0).long()

        # generate pseudo label using old model's output
        threshold = self.config['hyperparameter']['threshold']
        outputs_old[outputs_old[:, :, :, :] < threshold] = 0
        outputs_old = torch.argmax(outputs_old, dim=1, keepdim=True)
        outputs_old_down = (F.interpolate(input=outputs_old.double(), size=(H, W), mode='nearest')).long()
        pseudo_label_old_down = (outputs_old_down * labels_down_bgr_mask).long()

        temperature = self.config['hyperparameter']['temperature']
        decoder_dim = self.config['arch']['args']['decoder_dim']

        class_prototypes_teacher = torch.zeros([num_class + 1, B + 1, decoder_dim]).to(self.device)
        cnt_class_teacher = torch.zeros([num_class + 1]).long().to(self.device)
        for bs in range(B):
            cl_present = torch.unique(input=pseudo_label_old_down[bs]).long()
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
                features_cl = features_old[bs][(pseudo_label_old_down[bs] == cl).expand(C, -1, -1)].view(C, -1)
                # [dim]
                class_prototypes_teacher[cl, cnt_class_teacher[cl], :] = F.normalize(
                    torch.mean(features_cl, dim=-1).float(), p=2, dim=0)
                cnt_class_teacher[cl] += 1

        for i in range(1, num_class + 1):
            class_prototypes_teacher[i, cnt_class_teacher[i], :] = F.normalize(prototypes[i].detach(), p=2, dim=0)
            cnt_class_teacher[i] += 1

        pseudo_label = pseudo_label_old_down + labels_down
        class_prototypes_student = torch.zeros([num_class + 1, B + 1, decoder_dim]).to(self.device)
        cnt_class_student = torch.zeros([num_class + 1]).long().to(self.device)
        for bs in range(B):
            cl_present = torch.unique(input=pseudo_label[bs]).long()
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
                features_cl = features[bs][(pseudo_label[bs] == cl).expand(C, -1, -1)].view(C, -1)
                # [dim]
                class_prototypes_student[cl, cnt_class_student[cl], :] = F.normalize(
                    torch.mean(features_cl, dim=-1).float(), p=2, dim=0)
                cnt_class_student[cl] += 1

        for i in range(1, num_class + 1):
            class_prototypes_student[i, cnt_class_student[i], :] = F.normalize(prototypes[i].detach(), p=2, dim=0)
            cnt_class_student[i] += 1

        loss_old = torch.tensor(0.).to(self.device)
        cnt_class_old_exist = torch.tensor(0.).to(self.device)
        for i in range(1, num_old_class + 1):
            if cnt_class_student[i] == 1:
                continue
            for j in range(cnt_class_student[i] - 1):
                feature_student = class_prototypes_student[i, j, :]
                similarity_neg = torch.zeros([num_class + 1]).to(self.device)
                for m in range(1, num_class + 1):
                    for n in range(cnt_class_student[m]):
                        neg = class_prototypes_student[m, n, :]
                        similarity_neg[m] += torch.exp(torch.dot(feature_student, neg) / temperature) / cnt_class_student[m]
                for k in range(cnt_class_teacher[i]):
                    feature_teacher = class_prototypes_teacher[i, k, :]
                    similarity_pos = torch.exp(torch.dot(feature_teacher, feature_student) / temperature)

                    numerator = similarity_pos
                    denominator = torch.sum(similarity_neg)

                    loss_old += -1.0 * torch.log(numerator / denominator) / cnt_class_teacher[i] / (cnt_class_student[i] - 1)

            cnt_class_old_exist += 1

        if cnt_class_old_exist != 0:
            loss_old /= cnt_class_old_exist

        loss_new = torch.tensor(0.).to(self.device)
        cnt_class_new_exist = torch.tensor(0.).to(self.device)
        for i in range(1, num_class + 1):
            if cnt_class_student[i] == 1:
                continue
            for j in range(cnt_class_student[i] - 1):
                anchor = class_prototypes_student[i, j, :]
                similarity_anchor_neg = torch.zeros([num_class + 1]).to(self.device)
                for m in range(1, num_class + 1):
                    for n in range(cnt_class_student[m]):
                        neg = class_prototypes_student[m, n, :]
                        similarity_anchor_neg[m] += torch.exp(torch.dot(anchor, neg) / temperature) / cnt_class_student[m]
                for k in range(cnt_class_student[i]):
                    if j == k:
                        continue
                    pos = class_prototypes_student[i, k, :]
                    similarity_anchor_pos = torch.exp(torch.dot(anchor, pos) / temperature)

                    numerator = similarity_anchor_pos
                    denominator = torch.sum(similarity_anchor_neg)

                    loss_new += -1.0 * torch.log(numerator / denominator) / (cnt_class_student[i] - 1) / (cnt_class_student[i] - 1)

            cnt_class_new_exist += 1

        loss_new /= cnt_class_new_exist

        loss = loss_old + 0.1 * loss_new

        return loss

    def forward(self, labels, features_old, features, outputs_old, outputs, prototypes, num_class, num_old_class,
                num_new_class, epoch, train_step, len_epoch):
        if self.config['data_loader']['args']['task']['step'] == 0:
            return self.forward_base(labels, features_old, features, outputs_old, outputs, prototypes, num_class, num_old_class,
                num_new_class, epoch, train_step, len_epoch)
        else:
            return self.forward_incremental(labels, features_old, features, outputs_old, outputs, prototypes, num_class, num_old_class,
                num_new_class, epoch, train_step, len_epoch)


