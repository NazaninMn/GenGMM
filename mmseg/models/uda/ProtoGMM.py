# Obtained from: https://github.com/BIT-DA/SePiCo
# Modifications: ProtoGMM

# ---------------------------------------------------------------
# Copyright (c) 2022 BIT-DA. All rights reserved.
# Licensed under the Apache License, Version 2.0
# -------------------------------------------------
# A copy of the license is available at resources/license_SePiCo

# The ema model update and the domain-mixing are based on:
# https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License.
# A copy of the license is available at resources/license_dacs

import math
import os
import random
from copy import deepcopy

import numpy as np
import torch
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.models import UDA, build_segmentor
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks, get_mean_std, strong_transform)
from mmseg.models.utils.visualization import subplotimg
from mmseg.models.utils.ours_transforms import RandomCrop, RandomCropNoProd

from mmseg.models.utils.proto_estimator import ProtoEstimator
from mmseg.models.losses.contrastive_loss import contrast_preparations
from mmseg.ops import resize   # TODO: Modification GenGMM


def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm


@UDA.register_module()
class ProtoGMM(UDADecorator):

    def __init__(self, **cfg):
        super(ProtoGMM, self).__init__(**cfg)
        # basic setup
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.alpha = cfg['alpha']

        # for ssl
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.mix = cfg['mix']
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.debug_img_interval = cfg['debug_img_interval']
        assert self.mix == 'class'
        self.enable_self_training = cfg['enable_self_training']
        self.enable_strong_aug = cfg['enable_strong_aug']
        self.push_off_self_training = cfg.get('push_off_self_training', False)

        # configs for contrastive
        self.proj_dim = cfg['model']['auxiliary_head']['channels']
        self.contrast_mode = cfg['model']['auxiliary_head']['input_transform']
        self.calc_layers = cfg['model']['auxiliary_head']['in_index']
        self.num_classes = cfg['model']['decode_head']['num_classes']
        self.enable_avg_pool = cfg['model']['auxiliary_head']['loss_decode']['use_avg_pool']
        self.scale_min_ratio = cfg['model']['auxiliary_head']['loss_decode']['scale_min_ratio']

        # iter to start cl
        self.start_distribution_iter = cfg['start_distribution_iter']

        # for prod strategy (CBC)
        self.pseudo_random_crop = cfg.get('pseudo_random_crop', False)
        self.crop_size = cfg.get('crop_size', (640, 640))
        self.cat_max_ratio = cfg.get('cat_max_ratio', 0.75)
        self.regen_pseudo = cfg.get('regen_pseudo', False)
        self.prod = cfg.get('prod', True)

        self.class_dist_source =  torch.zeros(self.num_classes).cuda()   #TODO: Modifications
        self.class_dist_target = torch.zeros(self.num_classes).cuda()  # TODO: Modifications

        self.Amount_source = torch.zeros(self.num_classes).cuda()  # TODO: Modifications
        self.Amount_target = torch.zeros(self.num_classes).cuda()  # TODO: Modifications

        # feature storage for contrastive
        self.feat_distributions = None
        self.ignore_index = 255

        # BankCL memory length
        self.memory_length = cfg.get('memory_length', 0)  # 0 means no memory bank

        self.mean_s = torch.zeros(cfg['max_iters'], 19*5, 64)   #TODO: Modifications GenGMM
        self.cov_s = torch.zeros(cfg['max_iters'], 19*5, 64)   #TODO: Modifications GenGMM
        self.class_dist_source_save = torch.zeros(cfg['max_iters'], self.num_classes)   # TODO: Modifications GenGMM
        self.class_dist_target_save = torch.zeros(cfg['max_iters'], self.num_classes)   # TODO: Modifications GenGMM
        self.scaler = torch.cuda.amp.GradScaler() #TODO: Modifications GenGMM
        self.unlabeled_source = cfg['unlabeled']  # TODO: unlabeled source #TODO: Modifications GenGMM
        self.weak_target = cfg['weak_target']  # TODO: weak labels target #TODO: Modifications GenGMM
        print('self.unlabeled_source, self.weak_target',self.unlabeled_source, self.weak_target)#TODO: Modifications GenGMM
        # init distribution
        if self.contrast_mode == 'multiple_select':
            self.feat_distributions = {}
            for idx in range(len(self.calc_layers)):
                self.feat_distributions[idx] = ProtoEstimator(dim=self.proj_dim, class_num=self.num_classes,
                                                              memory_length=self.memory_length)
        else:  # 'resize_concat' or None
            self.feat_distributions = ProtoEstimator(dim=self.proj_dim, class_num=self.num_classes,
                                                     memory_length=self.memory_length)

        # ema model
        ema_cfg = deepcopy(cfg['model'])
        self.ema_model = build_segmentor(ema_cfg)

    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    def _init_ema_weights(self):
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        optimizer.zero_grad()
        log_vars = self(**data_batch)
        self.scaler.step(optimizer)
        self.scaler.update()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def random_crop(self, image, gt_seg, prod=True):
        if prod:
            RC = RandomCrop(crop_size=self.crop_size, cat_max_ratio=self.cat_max_ratio)
        else:
            RC = RandomCropNoProd(crop_size=self.crop_size, cat_max_ratio=self.cat_max_ratio)
        assert self.pseudo_random_crop
        image = image.permute(0, 2, 3, 1).contiguous()
        gt_seg = gt_seg
        res_img, res_gt = [], []
        for img, gt in zip(image, gt_seg):
            results = {'img': img, 'gt_semantic_seg': gt, 'seg_fields': ['gt_semantic_seg']}
            results = RC(results)
            img, gt = results['img'], results['gt_semantic_seg']
            res_img.append(img.unsqueeze(0))
            res_gt.append(gt.unsqueeze(0))
        image = torch.cat(res_img, dim=0).permute(0, 3, 1, 2).contiguous()
        gt_seg = torch.cat(res_gt, dim=0).long()
        return image, gt_seg

    def random_crop_gt_weak_seg(self, image, gt_seg, prod=True,gt_weak_seg=None):
        if prod:
            RC = RandomCrop(crop_size=self.crop_size, cat_max_ratio=self.cat_max_ratio)
        else:
            RC = RandomCropNoProd(crop_size=self.crop_size, cat_max_ratio=self.cat_max_ratio)
        assert self.pseudo_random_crop
        image = image.permute(0, 2, 3, 1).contiguous()
        gt_seg = gt_seg
        res_img, res_gt, res_gt_weak = [], [], []
        for img, gt,gt_weak in zip(image, gt_seg, torch.squeeze(gt_weak_seg)):
            results = {'img': img, 'gt_semantic_seg': gt, 'gt_weak': gt_weak, 'seg_fields': ['gt_semantic_seg','gt_weak']}
            results = RC(results)
            img, gt, gt_weak = results['img'], results['gt_semantic_seg'], results['gt_weak']
            res_img.append(img.unsqueeze(0))
            res_gt.append(gt.unsqueeze(0))
            res_gt_weak.append(gt_weak.unsqueeze(0))
        image = torch.cat(res_img, dim=0).permute(0, 3, 1, 2).contiguous()
        gt_seg = torch.cat(res_gt, dim=0).long()
        gt_weak = torch.cat(res_gt_weak, dim=0).long()
        return image, gt_seg,gt_weak

    def forward_train(self, img, img_metas, gt_semantic_seg, unlabeled_source_img_metas,unlabeled_source_img,unlabeled_source_gt_semantic_seg,target_img, target_img_metas, target_gt_semantic_seg):  # TODO: Modification GenGMM
    
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training

        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }

        if self.unlabeled_source: # with unlabeled source # TODO: Modification GenGMM
            weak_img, unlabeled_weak_source, weak_target_img = img.clone(), unlabeled_source_img.clone(), target_img.clone()  # TODO: Modification GenGMM
        else: # weak target # TODO: Modification GenGMM
            weak_img, weak_target_img = img.clone(), target_img.clone()  # TODO: Modification GenGMM

        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False

        # Generate pseudo-label
#         self.pseudo_threshold=0 #check
#         self.local_iter = 3333#check
        ema_target_logits = self.get_ema_model().encode_decode(weak_target_img, target_img_metas)
        ema_target_softmax = torch.softmax(ema_target_logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_target_softmax, dim=1)
        # find pseudo labels with prob> pseudo_threshold
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        # find a weight for each image as was proposed in the article
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size
        if self.unlabeled_source:
            # Generate pseudo-label-source_unlabeled  # TODO: Modification GenGMM
            ema_unlabeled_source_logits = self.get_ema_model().encode_decode(unlabeled_weak_source, unlabeled_source_img_metas) # TODO: Modification GenGMM
            ema_unlabeled_source_softmax = torch.softmax(ema_unlabeled_source_logits.detach(), dim=1)# TODO: Modification GenGMM
            pseudo_prob_unlabeled_source, pseudo_label_unlabeled_source = torch.max(ema_unlabeled_source_softmax, dim=1)# TODO: Modification GenGMM
            # find pseudo labels with prob> pseudo_threshold  # TODO: Modification GenGMM
            ps_large_p_unlabeled_source = pseudo_prob_unlabeled_source.ge(self.pseudo_threshold).long() == 1# TODO: Modification GenGMM
            ps_size_unlabeled_source = np.size(np.array(pseudo_label_unlabeled_source.cpu()))# TODO: Modification GenGMM
            pseudo_weight_unlabeled_source = torch.sum(ps_large_p_unlabeled_source).item() / ps_size_unlabeled_source# TODO: Modification GenGMM


        # pseudo RandomCrop
        if self.pseudo_random_crop:
            weak_target_img, pseudo_label,target_gt_semantic_seg = self.random_crop_gt_weak_seg(weak_target_img, pseudo_label, prod=self.prod,gt_weak_seg=target_gt_semantic_seg) # TODO: Modification GenGMM
            if self.regen_pseudo:
                # Re-Generate pseudo-label
                ema_target_logits = self.get_ema_model().encode_decode(weak_target_img, target_img_metas)
                ema_target_softmax = torch.softmax(ema_target_logits.detach(), dim=1)
                pseudo_prob, pseudo_label = torch.max(ema_target_softmax, dim=1)
                # find pseudo labels with prob> pseudo_threshold
                ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
                ps_size = np.size(np.array(pseudo_label.cpu()))
                # find a weight for each image as was proposed in the article
                pseudo_weight = torch.sum(ps_large_p).item() / ps_size
            target_img = weak_target_img.clone()

        if self.enable_strong_aug:
            img, gt_semantic_seg = strong_transform(
                strong_parameters,
                data=img,
                target=gt_semantic_seg
            )
            target_img, _ = strong_transform(
                strong_parameters,
                data=target_img,
                target=pseudo_label.unsqueeze(1)
            )
            # strong augmentation on the unlabeled source data with their pseudo labels  # TODO: Modification GenGMM
            if self.unlabeled_source: # TODO: Modification GenGMM
                unlabeled_source_img, _ = strong_transform( # TODO: Modification GenGMM
                    strong_parameters, # TODO: Modification GenGMM
                    data=unlabeled_source_img, # TODO: Modification GenGMM
                    target=pseudo_label_unlabeled_source.unsqueeze(1) # TODO: Modification GenGMM
                )

        pseudo_weight = pseudo_weight * torch.ones(pseudo_label.shape, device=dev)
        if self.unlabeled_source:
            pseudo_weight_unlabeled_source = pseudo_weight_unlabeled_source * torch.ones(pseudo_label.shape, device=dev)# TODO: Modification GenGMM

        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
            # unlabeled source
            if self.unlabeled_source:
                pseudo_weight_unlabeled_source[:, :self.psweight_ignore_top, :] = 0 # TODO: Modification GenGMM
        if self.psweight_ignore_bottom > 0:
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
            # unlabeled source
            if self.unlabeled_source:
                pseudo_weight_unlabeled_source[:, -self.psweight_ignore_bottom:, :] = 0 # TODO: Modification GenGMM

        gt_pixel_weight = torch.ones(pseudo_weight.shape, device=dev)
        if self.unlabeled_source:
            gt_pixel_weight_unlabeled_source = torch.ones(pseudo_weight_unlabeled_source.shape, device=dev)  # TODO: Modification GenGMM


        ema_source_logits = self.get_ema_model().encode_decode(weak_img, img_metas)
        ema_source_softmax = torch.softmax(ema_source_logits.detach(), dim=1)
        _, source_pseudo_label = torch.max(ema_source_softmax, dim=1)

        weak_gt_semantic_seg = gt_semantic_seg.clone().detach()

        # update distribution
        ema_src_feat = self.get_ema_model().extract_auxiliary_feat(weak_img)
        mean = {}
        covariance = {}
        bank = {}
        if self.contrast_mode == 'multiple_select':
            for idx in range(len(self.calc_layers)):
                feat, mask = contrast_preparations(ema_src_feat[idx], weak_gt_semantic_seg, self.enable_avg_pool,
                                                   self.scale_min_ratio, self.num_classes, self.ignore_index)
                self.feat_distributions[idx].update_proto(features=feat.detach(), labels=mask)
                mean[idx] = self.feat_distributions[idx].Ave
                covariance[idx] = self.feat_distributions[idx].CoVariance
                bank[idx] = self.feat_distributions[idx].MemoryBank
        else:  # 'resize_concat' or None
            feat, mask = contrast_preparations(ema_src_feat, weak_gt_semantic_seg, self.enable_avg_pool,
                                               self.scale_min_ratio, self.num_classes, self.ignore_index)
            out_seg, contrast_logits, contrast_target = self.feat_distributions.update_proto(features=feat.detach(), labels=mask) # TODO: Modification GenGMM
            mean = self.feat_distributions.means   # TODO: Modification GenGMM
            covariance = self.feat_distributions.diagonal # TODO: Modification GenGMM
            bank = self.feat_distributions.MemoryBank
            
            self.mean_s[self.local_iter,:] = self.feat_distributions.means.view(-1,64) # TODO: Modification GenGMM
            self.cov_s[self.local_iter,:] = self.feat_distributions.diagonal.view(-1,64) # TODO: Modification GenGMM
            self.class_dist_source_save[self.local_iter,:] = self.class_dist_source  # TODO: Modification GenGMM
            self.class_dist_target_save[self.local_iter,:] = self.class_dist_target # TODO: Modification GenGMM
        
        if self.local_iter%5000==0:   # TODO: Modification GenGMM
                np.save('mean.npy', self.mean_s.cpu().numpy())   # TODO: Modification GenGMM
                np.save('cov.npy', self.cov_s.cpu().numpy())      # TODO: Modification GenGMM
                np.save('source.npy', self.class_dist_source_save.cpu().numpy()) # TODO: Modification GenGMM
                np.save('target.npy', self.class_dist_target_save.cpu().numpy()) # TODO: Modification GenGMM
        # source ce + cl
        src_mode = 'dec'  # stands for ce only
        if self.local_iter >= self.start_distribution_iter:
            src_mode = 'all'  # stands for ce + cl
        # loss over labeled source
        with torch.cuda.amp.autocast():
            source_losses,_ = self.get_model().forward_train(img, img_metas, gt_semantic_seg, return_feat=False,
                                                           mean=mean, covariance=covariance, bank=bank, mode=src_mode, source=True, class_sourse=self.class_dist_source, class_target= self.class_dist_target,target_memory=None, weak_target=False) # TODO: Modifications ( ProtoGMM)
            source_loss, source_log_vars = self._parse_losses(source_losses)
            log_vars.update(add_prefix(source_log_vars, 'src'))
        self.scaler.scale(source_loss).backward()

        # loss over unlabeled source
        if self.unlabeled_source: # unlabeled source # loss over labeled source
            if self.local_iter >= self.start_distribution_iter:  # loss over labeled source
                # unlabeled source cl(Aligning source and unlabeled domain distribution)
                pseudo_lbl_unlabeled_source = pseudo_label_unlabeled_source.clone()  # pseudo label should not be overwritten   # TODO: Modification GenGMM
                pseudo_lbl_unlabeled_source[pseudo_weight_unlabeled_source == 0.] = self.ignore_index  # TODO: Modification GenGMM
                pseudo_lbl_unlabeled_source = pseudo_lbl_unlabeled_source.unsqueeze(1) # TODO: Modification GenGMM
                unlabeled_source_losses, [weight_gaussian_reshape,unlabele_gaussian] = self.get_model().forward_train(unlabeled_source_img, unlabeled_source_img_metas, [unlabeled_source_gt_semantic_seg, pseudo_lbl_unlabeled_source], return_feat=False, # TODO: Modification GenGMM
                                                               mean=mean, covariance=covariance, bank=bank, mode='aux', # TODO: Modification GenGMM
                                                               source=False, class_sourse=self.class_dist_source, # TODO: Modification GenGMM
                                                               class_target=self.class_dist_source,   # since both of them are from source # TODO: Modification GenGMM
                                                               target_memory=None, ema_unlabeled_source_softmax=ema_unlabeled_source_softmax, weak_target=False)  # TODO: Modification GenGMM

                unlabeled_loss, unlabeled_log_vars = self._parse_losses(unlabeled_source_losses)# TODO: Modification GenGMM
                log_vars.update(add_prefix(unlabeled_log_vars, 'tgt'))# TODO: Modification GenGMM
                self.scaler.scale(unlabeled_loss).backward()# TODO: Modification GenGMM

        ########

        # Source prior dist update (labeled)# TODO: Modification GenGMM
        source_labels = gt_semantic_seg[gt_semantic_seg != 255].clone()# TODO: Modification GenGMM
        N = source_labels.shape[0]  # TODO: Modification GenGMM
        onehot = torch.zeros(N, self.num_classes).cuda()  # TODO: Modification GenGMM
        onehot.scatter_(1, source_labels.view(-1, 1), 1)  # TODO: Modification GenGMM
        self.Amount_source = self.Amount_source + onehot.sum(0)  # TODO: Modification GenGMM

        if self.local_iter%1==0:  # TODO: Modification GenGMM
           current_prob = self.Amount_source/self.Amount_source.sum()  # TODO: Modification GenGMM
           self.class_dist_source = 0.9*self.class_dist_source+(1-0.9)*current_prob  # TODO: Modification GenGMM
           self.Amount_source = torch.zeros(self.num_classes).cuda()  # TODO: Modification GenGMM


        # target prior dist update
        if self.local_iter >= 2000: # TODO: Modification GenGMM
            pseudo_lbl = pseudo_label.clone()  # pseudo label should not be overwritten# TODO: Modification GenGMM
            pseudo_lbl[pseudo_weight == 0.] = self.ignore_index  # TODO: Modification GenGMM
            if self.weak_target: # In the presence of weak target # TODO: Modification GenGMM
                pseudo_lbl[target_gt_semantic_seg < 200] = target_gt_semantic_seg[target_gt_semantic_seg < 200]# TODO: Modification GenGMM
            target_labels = pseudo_lbl[pseudo_lbl != 255].clone() # TODO: Modification GenGMM
            N = target_labels.shape[0]  # TODO: Modification GenGMM
            onehot = torch.zeros(N, self.num_classes).cuda()  # TODO: Modification GenGMM
            onehot.scatter_(1, target_labels.view(-1, 1), 1)  # TODO: Modification GenGMM
            self.Amount_target = self.Amount_target + onehot.sum(0)  # TODO: Modification GenGMM
            if self.weak_target:# In the presence of weak target # TODO: Modification GenGMM
                list_not = set(torch.arange(19).tolist()) - set(torch.unique(target_gt_semantic_seg).tolist())# TODO: Modification GenGMM
                self.Amount_target[list(list_not)] = 0# TODO: Modification GenGMM


            if self.local_iter % 100==0:  # TODO: Modification GenGMM
                current_prob = self.Amount_target / self.Amount_target.sum()  # TODO: Modification GenGMM
                self.class_dist_target = 0.9 * self.class_dist_target + (
                        1 - 0.9) * current_prob  # TODO: Modification GenGMM
                self.Amount_target = torch.zeros(self.num_classes).cuda()  # TODO: Modification GenGMM
                
            # Target
            ema_src_feat_target = self.get_ema_model().extract_auxiliary_feat(weak_target_img)
            # the below line reduce the size of the mask, since labels are integrated, if they ratio is less than 0.75, we add ignore label to it
            feat, mask = contrast_preparations(ema_src_feat_target, pseudo_lbl.unsqueeze(1), self.enable_avg_pool,
                                               self.scale_min_ratio, self.num_classes, self.ignore_index,
                                               remove_ignore=False) # TODO: Modification GenGMM
            # Update target bank
            if self.weak_target: # In the presence of weak labeled targets
                feat_weak, mask_weak = contrast_preparations(ema_src_feat_target, target_gt_semantic_seg.unsqueeze(1), self.enable_avg_pool,
                                                   self.scale_min_ratio, self.num_classes, self.ignore_index,
                                                   remove_ignore=False)  # TODO: Modification GenGMM
                mask[mask_weak<200]=mask_weak[mask_weak<200]# TODO: Modification GenGMM
                for id_label in list_not:# TODO: Modification GenGMM
                    mask[mask==id_label]=255# TODO: Modification GenGMM
            self.feat_distributions.update_target_bank(features=feat.detach(),labels=mask)   # TODO: Modification GenGMM

            ########

        if self.local_iter >= self.start_distribution_iter:
            with torch.cuda.amp.autocast(): 
                # target cl(Aligning source and target domain distribution)
                pseudo_lbl = pseudo_lbl.unsqueeze(1)
                if self.weak_target: # If the target is weakly labeled # TODO: Modification GenGMM
                    target_losses, [local_gauss_target,unlabele_gaussian_target] = self.get_model().forward_train(target_img, target_img_metas,
                                                                                      [target_gt_semantic_seg, pseudo_lbl],
                                                                                      return_feat=False,
                                                                                      mean=mean, covariance=covariance,
                                                                                      bank=bank, mode='aux', source=False,
                                                                                      class_sourse=self.class_dist_source,
                                                                                      class_target=self.class_dist_target,
                                                                                      target_memory=self.feat_distributions.queue_target,
                                                                                      ema_unlabeled_source_softmax=ema_target_softmax,
                                                                                      weak_target=self.weak_target)  # TODO: Modification GenGMM

                else: # If the target is unlabeled # TODO: Modification GenGMM
                    target_losses, [local_gauss_target,unlabele_gaussian_target] = self.get_model().forward_train(target_img, target_img_metas,
                                                                                      [target_gt_semantic_seg,pseudo_lbl], return_feat=False,
                                                                                      mean=mean, covariance=covariance,
                                                                                      bank=bank, mode='aux',
                                                                                      source=False,
                                                                                      class_sourse=self.class_dist_source,
                                                                                      class_target=self.class_dist_target,
                                                                                      target_memory=self.feat_distributions.queue_target,
                                                                                      ema_unlabeled_source_softmax=ema_target_softmax,
                                                                                      weak_target=False)  # TODO: Modification GenGMM                                                      class_target= self.class_dist_target, target_memory=self.feat_distributions.queue_target, ema_unlabeled_source_softmax=ema_target_softmax) # TODO: Modifications

                target_loss, target_log_vars = self._parse_losses(target_losses)
                log_vars.update(add_prefix(target_log_vars, 'tgt'))
            self.scaler.scale(target_loss).backward()

        local_enable_self_training = \
            self.enable_self_training and \
            (not self.push_off_self_training or self.local_iter >= self.start_distribution_iter)

        # mixed ce (ssl=source+target) # TODO: Modification GenGMM
        if local_enable_self_training:
            # Apply mixing
            with torch.cuda.amp.autocast():
                if self.weak_target and self.local_iter >= self.start_distribution_iter: # In the presence of weak labeled # TODO: Modification GenGMM
                    pseudo_weight = pseudo_weight * local_gauss_target.squeeze(1).detach()  # TODO: Modification GenGMM
                    pseudo_weight[(target_gt_semantic_seg < 200)] = 1 # TODO: Modification GenGMM
                    pseudo_label[(target_gt_semantic_seg < 200)]=target_gt_semantic_seg[(target_gt_semantic_seg < 200)] # TODO: Modification GenGMM

                mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
                mix_masks = get_class_masks(gt_semantic_seg)

                for i in range(batch_size):
                    strong_parameters['mix'] = mix_masks[i]
                    mixed_img[i], mixed_lbl[i] = strong_transform(
                        strong_parameters,
                        data=torch.stack((weak_img[i], weak_target_img[i])),
                        target=torch.stack((gt_semantic_seg[i][0], pseudo_label[i])))
                    _, pseudo_weight[i] = strong_transform(
                        strong_parameters,
                        target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
                mixed_img = torch.cat(mixed_img)
                mixed_lbl = torch.cat(mixed_lbl)

                # Train on mixed images
                mix_losses, seg_logits = self.get_model().forward_train(mixed_img, img_metas, mixed_lbl, pseudo_weight,
                                                            return_feat=False, mode='dec')
                mix_loss, mix_log_vars = self._parse_losses(mix_losses)
                log_vars.update(add_prefix(mix_log_vars, 'mix'))
            self.scaler.scale(mix_loss).backward()

        ################# mixed ce (ssl=source+unlabeled source) # TODO: Modification GenGMM
        if self.unlabeled_source: # If there is unlabeled source data # TODO: Modification GenGMM
            if self.local_iter >= self.start_distribution_iter:  # TODO: Modification GenGMM
                pseudo_weight_unlabeled_source = pseudo_weight_unlabeled_source*weight_gaussian_reshape.squeeze(1) # TODO: Modification GenGMM
                mixed_img_unlabeled_source, mixed_lbl_unlabeled_source = [None] * batch_size, [None] * batch_size  # TODO: Modification GenGMM
                mix_masks = get_class_masks(gt_semantic_seg) # TODO: Modification GenGMM

                for i in range(batch_size): # TODO: Modification GenGMM
                        strong_parameters['mix'] = mix_masks[i] # TODO: Modification GenGMM
                        mixed_img_unlabeled_source[i], mixed_lbl_unlabeled_source[i] = strong_transform( # TODO: Modification GenGMM
                            strong_parameters, # TODO: Modification GenGMM
                            data=torch.stack((weak_img[i], unlabeled_weak_source[i])), # TODO: Modification GenGMM
                            target=torch.stack(
                                (gt_semantic_seg[i][0], pseudo_label_unlabeled_source[i])))  # TODO: Modification GenGMM
                        _, pseudo_weight_unlabeled_source[i] = strong_transform( # TODO: Modification GenGMM
                            strong_parameters, # TODO: Modification GenGMM
                            target=torch.stack((gt_pixel_weight[i], pseudo_weight_unlabeled_source[i])))# TODO: Modification GenGMM
                mixed_img_unlabeled_source = torch.cat(mixed_img_unlabeled_source)# TODO: Modification GenGMM
                mixed_lbl_unlabeled_source = torch.cat(mixed_lbl_unlabeled_source)# TODO: Modification GenGMM
                mix_losses_unlabeled_source, seg_logits = self.get_model().forward_train(mixed_img_unlabeled_source, img_metas, mixed_lbl_unlabeled_source, pseudo_weight_unlabeled_source,
                                                            return_feat=False, mode='dec')# TODO: Modification GenGMM
                mix_loss_unlabeled_source, mix_log_vars_unlabeled_source = self._parse_losses(mix_losses_unlabeled_source)# TODO: Modification GenGMM
                log_vars.update(add_prefix(mix_log_vars_unlabeled_source, 'mix'))# TODO: Modification GenGMM
                mix_loss_unlabeled_source.backward()# TODO: Modification GenGMM


        if self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg['work_dir'], 'visualize_meta')
            os.makedirs(out_dir, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
            if local_enable_self_training:
                vis_mixed_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)
            ema_src_logits = self.get_ema_model().encode_decode(weak_img, img_metas)
            ema_softmax = torch.softmax(ema_src_logits.detach(), dim=1)
            _, src_pseudo_label = torch.max(ema_softmax, dim=1)
            for j in range(batch_size):
                rows, cols = 2, 5
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        'hspace': 0.1,
                        'wspace': 0,
                        'top': 0.95,
                        'bottom': 0,
                        'right': 1,
                        'left': 0
                    },
                )
                subplotimg(axs[0][0], vis_img[j], f'{img_metas[j]["ori_filename"]}')
                subplotimg(axs[1][0], vis_trg_img[j],
                           f'{os.path.basename(target_img_metas[j]["ori_filename"]).replace("_leftImg8bit", "")}')
                subplotimg(
                    axs[0][1],
                    src_pseudo_label[j],
                    'Source Pseudo Label',
                    cmap='cityscapes',
                    nc=self.num_classes)
                subplotimg(
                    axs[1][1],
                    pseudo_label[j],
                    'Target Pseudo Label',
                    cmap='cityscapes',
                    nc=self.num_classes)
                subplotimg(
                    axs[0][2],
                    gt_semantic_seg[j],
                    'Source Seg GT',
                    cmap='cityscapes',
                    nc=self.num_classes)
                if target_gt_semantic_seg.dim() > 1:
                    subplotimg(
                        axs[1][2],
                        target_gt_semantic_seg[j],
                        'Target Seg GT',
                        cmap='cityscapes',
                        nc=self.num_classes
                    )
                subplotimg(
                    axs[0][3], pseudo_weight[j], 'Pseudo W.', vmin=0, vmax=1)
                if local_enable_self_training:
                    subplotimg(
                        axs[1][3],
                        mix_masks[j][0],
                        'Mixed Mask',
                        cmap='gray'
                    )
                    subplotimg(
                        axs[0][4],
                        vis_mixed_img[j],
                        'Mixed ST Image')
                    subplotimg(
                        axs[1][4],
                        mixed_lbl[j],
                        'Mixed ST Label',
                        cmap='cityscapes',
                        nc=self.num_classes
                    )
                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir,
                                 f'{(self.local_iter + 1):06d}_{j}.png'))
                plt.close()
        self.local_iter += 1

        return log_vars
