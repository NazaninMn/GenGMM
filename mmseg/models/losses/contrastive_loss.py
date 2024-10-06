# Obtained from: https://github.com/BIT-DA/SePiCo
# Modifications: GenGMM loss function

# ---------------------------------------------------------------
# Copyright (c) 2022 BIT-DA. All rights reserved.
# Licensed under the Apache License, Version 2.0
# -------------------------------------------------
# A copy of the license is available at resources/license_SePiCo

# Note that `downscale_label_ratio` method is adapted from: https://github.com/lhoyer/DAFormer

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque

from ..builder import LOSSES
from .utils import get_class_weight, weight_reduce_loss
from mmseg.utils.distributions import MultivariateNormalDiag  # TODO: Modifications GenGMM
from mmseg.utils.GMMSeg import distributed_sinkhorn_wograd, shifted_var, init_weights, l2_normalize, momentum_update, \
    rnd_sample  # TODO: Modifications GenGMM
from mmseg.ops import resize  # TODO: Modifications GenGMM


def downscale_label_ratio(gt,
                          scale_factor,
                          min_ratio,
                          n_classes,
                          ignore_index=255, weak_target=False):  # TODO: weak labels # TODO: Modifications GenGMM
    assert scale_factor >= 1
    if scale_factor == 1:
        return gt.clone()
    bs, orig_c, orig_h, orig_w = gt.shape
    assert orig_c == 1
    trg_h, trg_w = orig_h // scale_factor, orig_w // scale_factor
    ignore_substitute = n_classes

    if weak_target == True:  # TODO: weak labels# TODO: Modifications GenGMM
        ignore_index = 254      # TODO: Modifications GenGMM
        unlabeled = 255      # TODO: Modifications GenGMM
        out = gt.clone()  # o/w next line would modify original gt      # TODO: Modifications GenGMM
        out[out == ignore_index] = ignore_substitute      # TODO: Modifications GenGMM
        out[out == unlabeled] = ignore_substitute + 1      # TODO: Modifications GenGMM
        out = F.one_hot(      # TODO: Modifications GenGMM
            out.squeeze(1), num_classes=n_classes + 2).permute(0, 3, 1, 2)      # TODO: Modifications GenGMM
        assert list(out.shape) == [bs, n_classes + 2, orig_h, orig_w], out.shape      # TODO: Modifications GenGMM
        out = F.avg_pool2d(out.float(), kernel_size=scale_factor)      # TODO: Modifications GenGMM
        gt_ratio, out = torch.max(out, dim=1, keepdim=True)      # TODO: Modifications GenGMM
        out[out == ignore_substitute] = ignore_index      # TODO: Modifications GenGMM
        out[out == ignore_substitute + 1] = unlabeled      # TODO: Modifications GenGMM
    else:
        out = gt.clone()  # o/w next line would modify original gt
        out[out == ignore_index] = ignore_substitute
        out = F.one_hot(
            out.squeeze(1), num_classes=n_classes + 1).permute(0, 3, 1, 2)
        assert list(out.shape) == [bs, n_classes + 1, orig_h, orig_w], out.shape
        out = F.avg_pool2d(out.float(), kernel_size=scale_factor)
        gt_ratio, out = torch.max(out, dim=1, keepdim=True)
        out[out == ignore_substitute] = ignore_index
        out[gt_ratio < min_ratio] = ignore_index
    assert list(out.shape) == [bs, 1, trg_h, trg_w], out.shape
    return out


def contrast_preparations(feat,
                          mask,
                          use_avg_pool,
                          scale_min_ratio,
                          num_classes,
                          ignore_index, remove_ignore=True,
                          weak_target=False):        # TODO: Modifications GenGMM # weak target labels
    # down-sample mask to fit feat
    if use_avg_pool:
        scale_factor = mask.shape[-1] // feat.shape[-1]
        mask = downscale_label_ratio(mask, scale_factor, scale_min_ratio, num_classes, ignore_index,
                                     weak_target=weak_target).long().detach()  # weak labels      # TODO: Modifications GenGMM
    else:
        mask = F.interpolate(mask.float(), size=feat.shape[-2:], mode='nearest').long()
    # normalize the feat
    # feat = F.normalize(feat, p=2, dim=1)  # already normalized in proj_head.py
    # transpose the feat shape
    A = feat.size(1)
    feat = feat.permute(0, 2, 3, 1).contiguous().view(-1, A)
    mask = mask.contiguous().view(-1)

    msk = (mask != ignore_index)
    # remove ignore_index pixels
    if remove_ignore == True:  # TODO: Modifications GenGMM
        mask = mask[msk]  # TODO: Modifications GenGMM
        feat = feat[msk]  # TODO: Modifications GenGMM
    return feat, mask


def proto_reg(feat,
              mean=None,
              contrast_temp=100.,
              contrast_norm=None,
              **kwargs):
    assert mean is not None, 'Parameter `mean` required'
    assert contrast_norm is not None, 'Parameter `contrast_norm` required'
    assert not mean.requires_grad
    assert feat.requires_grad

    if feat.size(0) == 0:
        return torch.tensor(0., requires_grad=True).cuda()

    mean_feat = torch.mean(feat, 0, keepdim=True)

    # feat (1, A) x Ave (A, C)
    proto_sim = mean_feat.mm(mean.mean(1).view(mean.shape[-1], -1).contiguous()) / contrast_temp  # TODO: Modifications GenGMM

    loss = torch.sum(torch.softmax(proto_sim, dim=1).log()) / contrast_norm  # TODO: Modifications GenGMM

    return loss


def proto_contrastive(feat,
                      mask,
                      mean=None,
                      index=-1,
                      contrast_temp=100.,
                      use_avg_pool=True,
                      scale_min_ratio=0.75,
                      num_classes=19,
                      weight=None,
                      class_weight=None,
                      reduction='mean',
                      avg_factor=None,
                      reg_weight=0,
                      ignore_index=255,
                      **kwargs):
    if index >= 0:
        assert isinstance(feat, list), f'feat list expected for index={index}'
        assert isinstance(mean, (list, dict)), f'mean list expected for index={index}'
        feat = feat[index]
        mean = mean[index]
    feat, mask = contrast_preparations(feat, mask, use_avg_pool, scale_min_ratio, num_classes, ignore_index,
                                       weak_target=kwargs['weak_target'])  # weak labels # TODO: Modifications GenGMM
    assert mean is not None, 'Parameter `mean` required'
    assert not mean.requires_grad
    assert feat.requires_grad
    assert not mask.requires_grad

    if feat.size(0) == 0:
        return torch.tensor(0., requires_grad=True).cuda()

    # feat (N, A) x Ave (A, C)
    proto_sim = feat.mm(mean.permute(1, 0).contiguous()) / contrast_temp

    # The wrapper function for :func:`F.cross_entropy`
    # class_weight is a manual rescaling weight given to each class.
    # If given, has to be a Tensor of size C element-wise losses
    loss = F.cross_entropy(
        proto_sim,
        mask,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    if reg_weight > 0.:
        contrast_norm = (num_classes * 1) * np.log(num_classes * 1) # TODO: Modifications GenGMM
        loss += reg_weight * proto_reg(feat, mean, contrast_temp, contrast_norm=contrast_norm)

    return loss


def ProtoGMM_loss(feat,
                  masks,  # weak source labels# TODO: Modifications GenGMM
                  mean=None,
                  covariance=None,
                  ratio=1.0,
                  index=-1,
                  contrast_temp=100.,
                  use_avg_pool=True,
                  scale_min_ratio=0.75,
                  num_classes=19,
                  weight=None,
                  class_weight=None,
                  reduction='mean',
                  avg_factor=None,
                  reg_weight=0,
                  ignore_index=255,
                  # source=True, # TODO: Modifications GenGMM
                  **kwargs):
    if index >= 0:
        assert isinstance(feat, list), f'feat list expected for index={index}'
        assert isinstance(mean, (list, dict)), f'mean list expected for index={index}'
        assert isinstance(covariance, (list, dict)), f'covariance list expected for index={index}'
        feat = feat[index]
        mean = mean[index]
        covariance = covariance[index]
    if kwargs['source'] == False and kwargs['target_memory'] == None:  # source unlabeled # TODO: Modifications GenGMM
        mask = masks[0]  # source unlabeled # TODO: Modifications GenGMM
        feat_psuedo, mask_psuedo = contrast_preparations(feat, masks[1], use_avg_pool, scale_min_ratio, num_classes,
                                                         ignore_index,
                                                         remove_ignore=False,
                                                         weak_target=kwargs[
                                                             'weak_target'])  # source unlabeled # TODO: Modifications GenGMM
    elif kwargs['source'] == False and kwargs['target_memory'] != None:  # TODO: weak/un labeled target data # TODO: Modifications GenGMM
        mask = masks[0].unsqueeze(1)  # weak target labels # TODO: Modifications GenGMM
        feat_psuedo, mask_psuedo = contrast_preparations(feat, masks[1], use_avg_pool, scale_min_ratio, num_classes,
                                                         ignore_index,
                                                         remove_ignore=False,
                                                         weak_target=kwargs[
                                                             'weak_target'])  # pseudo labels from classifier # TODO: Modifications GenGMM

    else:# TODO: Modifications GenGMM
        mask = masks.clone() # TODO: Modifications GenGMM
    feat, mask = contrast_preparations(feat, mask, use_avg_pool, scale_min_ratio, num_classes, ignore_index,
                                       remove_ignore=False,
                                       weak_target=kwargs['weak_target'])  # TODO: Modifications GenGMM
    assert covariance is not None, 'Parameter `covariance` required'
    assert not mean.requires_grad
    assert not covariance.requires_grad
    assert feat.requires_grad
    assert not mask.requires_grad
    criterion = nn.CrossEntropyLoss(reduction='none')

    # GMM
    # Compute log probability p(component/feature)
    _c_gauss = MultivariateNormalDiag(mean.view(-1, mean.shape[-1]), scale_diag=covariance.view(-1,covariance.shape[-1]))  # * c*p multivariate gaussian   # TODO: Modifications GenGMM
    feat = l2_normalize(feat)  # TODO: Modifications GenGMM
    log_prob = _c_gauss.log_prob(feat.detach()[:,None,:].cpu()).cuda()   # TODO: Modifications GenGMM
    log_prob = log_prob.view(-1, mean.shape[0], mean.shape[1]) # TODO: Modifications GenGMM
    if kwargs['source']: # Labeled source data, TODO: Modifications GenGMM
        msk = mask.clone() # TODO: Modifications GenGMM
        mask_ = mask.clone()  # TODO: Modifications GenGMM
        # mask=> ground truth labels for source domain data
        # index => GMM component
        softmax = torch.nn.Softmax()
    else:  # weakly or unlabeled source|target domain data, TODO: Modifications GenGMM
        softmax = torch.nn.Softmax()  # TODO: Modifications GenGMM
        # p(component, class/feature)
        value = softmax(log_prob.reshape(log_prob.shape[0], -1)).reshape(log_prob.shape[0], log_prob.shape[1],
                                                                 log_prob.shape[2]) # TODO: Modifications GenGMM
        # un/weak labeled target data
        if kwargs['target_memory']!=None: # TODO: Modifications GenGMM
            # distribution shift = target prior/source prior    # TODO: Modifications GenGMM
            dist_ratio = kwargs['class_target'] / kwargs['class_sourse']  # TODO: Modifications GenGMM
            dist_ratio = torch.nan_to_num(dist_ratio, nan=0.0)  # TODO: Modifications GenGMM
            #  p'(component, class/feature) = p(component, class/feature)*distribution shift   # TODO: Modifications GenGMM
            value = (dist_ratio.unsqueeze(0).unsqueeze(2) * value)  # TODO: Modifications GenGMM
            # similarity to target prototypes               # TODO: Modifications GenGMM
            mean_target = l2_normalize(kwargs['target_memory'].mean(2))  # TODO: Modifications GenGMM
            sim_to_target_mean = feat.detach().mm(mean_target.T.cuda())  # TODO: Modifications GenGMM
            sim_to_target_mean = softmax(sim_to_target_mean)  # TODO: Modifications GenGMM
            #  p'(component, class/feature) =  p'(component, class/feature)*similarity to target prototypes    # TODO: Modifications GenGMM
            value = value * sim_to_target_mean.unsqueeze(2)  # TODO: Modifications GenGMM
            msk = mask.clone()  # TODO: Modifications GenGMM
            if kwargs['weak_target']:  # weak labeled target data# TODO: Modifications GenGMM
                mean_loc = torch.zeros(19, 64)  # weak labeled target data# TODO: Modifications GenGMM
                unique_labels = set(torch.unique(mask).tolist()) - {254, 255}  # weak labeled target data# TODO: Modifications GenGMM
                local_gauss = torch.zeros(mask.shape[0], 19)  # weak labeled target data# TODO: Modifications GenGMM
                for i in range(19):  # weak labeled target data# TODO: Modifications GenGMM
                    if i in unique_labels:  # weak labeled target data# TODO: Modifications GenGMM
                        # image_i = mask_160_160# weak labeled target data# TODO: Modifications GenGMM
                        mean_loc[i] = feat[mask == i].detach().mean(0)  # weak labeled target data# TODO: Modifications GenGMM
                        diff = (feat.detach() - mean_loc[i].cuda()).cuda()  # weak labeled target data# TODO: Modifications GenGMM
                        local_gauss[:, i] = torch.exp(  # weak labeled target data# TODO: Modifications GenGMM
                            -torch.matmul(diff.unsqueeze(1),
                                          diff.unsqueeze(2))).squeeze()  # p*(feat-mean)^2 # weak labeled target data# TODO: Modifications GenGMM
                        local_gauss[:, i][mask == i] = 1  # weak labeled target data# TODO: Modifications GenGMM
            if kwargs['weak_target']!=True: # Unlabeled target data # TODO: Modifications GenGMM
                mask_ = torch.argmax(value.sum(-1), axis=-1)  # Unlabeled target data # TODO: Modifications GenGMM

            classifier = kwargs['ema_unlabeled_source_softmax']  # Unlabeled target data # TODO: Modifications GenGMM


        else:   #unlabeled source# TODO: Modifications GenGMM

            ######mask for d^2/zigma^2   #unlabeled source# TODO: Modifications GenGMM
            msk = mask.clone()  #unlabeled source# TODO: Modifications GenGMM
            mask = torch.argmax(log_prob.view(-1,95), -1)  #unlabeled source# TODO: Modifications GenGMM
            means = _c_gauss.loc[mask]  #unlabeled source# TODO: Modifications GenGMM
            cov2 = torch.diag_embed(_c_gauss.scale_diag)[mask].cuda()#unlabeled source# TODO: Modifications GenGMM
            # The final output will be a 3D array with shape (95, 64, 64)  #unlabeled source# TODO: Modifications GenGMM
            diff = (feat.detach()-means.cuda()).unsqueeze(1).cuda() #unlabeled source# TODO: Modifications GenGMM
            # Compute the differences between each data point and the mean vectors #unlabeled source# TODO: Modifications GenGMM
            # Transpose cov2 for proper multiplication #unlabeled source# TODO: Modifications GenGMM
            cov2_transposed = 1/(2*cov2.transpose(1, 2))  # Shape: [51200, 64, 64] #unlabeled source# TODO: Modifications GenGMM
            # Check for inf values and create a mask #unlabeled source# TODO: Modifications GenGMM
            inf_mask = torch.isinf(cov2_transposed)  #unlabeled source# TODO: Modifications GenGMM
            # Replace inf values with zero using the mask  #unlabeled source# TODO: Modifications GenGMM
            cov2_transposed[inf_mask] = 0.0  #unlabeled source# TODO: Modifications GenGMM
            # Perform matrix multiplication to calculate the Mahalanobis distances #unlabeled source# TODO: Modifications GenGMM
            # The result will have shape [51200, 1, 1] #unlabeled source# TODO: Modifications GenGMM
            weight_gaussian = torch.exp(-torch.matmul(torch.matmul(diff, cov2_transposed), diff.transpose(1, 2))).squeeze() #unlabeled source# TODO: Modifications GenGMM
            weight_gaussian_reshape = weight_gaussian.reshape(2, 160, 160)  #unlabeled source# TODO: Modifications GenGMM

            classifier = kwargs['ema_unlabeled_source_softmax']  #unlabeled source# TODO: Modifications GenGMM
            weight_gaussian_reshape = resize(  #unlabeled source# TODO: Modifications GenGMM
                input=weight_gaussian_reshape.unsqueeze(1),  #unlabeled source# TODO: Modifications GenGMM
                size=classifier.shape[-2:],  #unlabeled source# TODO: Modifications GenGMM
                mode='bilinear',  #unlabeled source# TODO: Modifications GenGMM
                align_corners=True)  #unlabeled source# TODO: Modifications GenGMM

            mask_ = torch.argmax(value.sum(-1), axis=-1)[
                msk != 255]  #unlabeled source# TODO: Modifications GenGMM

    if feat.size(0) == 0:# TODO: Modifications GenGMM
        return torch.tensor(0., requires_grad=True).cuda()
    mean = mean.cuda() # TODO: Modifications GenGMM
    # Multi ptototype contrastive learning with hard sampling

    logits = feat.mm(
        mean.view(-1, mean.shape[-1]).permute(1, 0).contiguous()) / contrast_temp  # TODO: Modifications GenGMM
    if msk[msk!=255].sum()==0:
        loss = (logits.sum(1)*0).mean()
    else:
        logits = logits.view(-1, mean.shape[0], mean.shape[1]) # TODO: Modifications GenGMM

        if kwargs['source'] == True: # labeled source# TODO: Modifications GenGMM
            logits = logits[
                msk != 255]# TODO: Modifications GenGMM
            mask_ = mask_[
                msk != 255]# TODO: Modifications GenGMM
            mask_one_hot = F.one_hot(mask_, mean.shape[0])  # m# TODO: Modifications GenGMM
            logits_max = logits.max(-1)[0]# TODO: Modifications GenGMM
            logits = mask_one_hot * logits_max + (1 - mask_one_hot) * logits_max  # max over both# TODO: Modifications GenGMM
            loss = F.cross_entropy(
                logits,
                mask_,
                weight=class_weight,
                reduction='none',
                ignore_index=ignore_index)# TODO: Modifications GenGMM
        if kwargs['source'] == False and kwargs['target_memory'] != None: # un/weak labeled target# TODO: Modifications GenGMM
            if kwargs['weak_target']:#weak labeled target# TODO: Modifications GenGMM
                label_gaussian = value.max(-1)[0].max(-1)[1]  #weak labeled target# TODO: Modifications GenGMM
                label_gaussian[mask < 200] = mask[mask < 200]#weak labeled target# TODO: Modifications GenGMM
                label_gaussian = label_gaussian.cuda()#weak labeled target# TODO: Modifications GenGMM
                local_gauss = local_gauss.cuda()#weak labeled target# TODO: Modifications GenGMM
                local_gauss_pseudo = local_gauss.cuda().clone()#weak labeled target# TODO: Modifications GenGMM
                local_gauss = local_gauss[torch.arange(local_gauss.shape[0]), label_gaussian]  #weak labeled target# TODO: Modifications GenGMM
                mask_psuedo_ = mask_psuedo.clone()#weak labeled target# TODO: Modifications GenGMM
                mask_psuedo_[mask_psuedo == 255] = 0#weak labeled target# TODO: Modifications GenGMM
                local_gauss_pseudo = local_gauss_pseudo[
                    torch.arange(local_gauss_pseudo.shape[0]), mask_psuedo_]  #weak labeled target# TODO: Modifications GenGMM
                local_gauss[mask < 200] = 1  # put weak labels guass 1 #weak labeled target# TODO: Modifications GenGMM
                local_gauss_pseudo[mask < 200] = 1  # put weak labels guass 1#weak labeled target# TODO: Modifications GenGMM
                mask_psuedo[mask < 200] = mask[
                    mask < 200]  # put weak labels in psudo labels, to be sure weak labels not being ignored in pseudo labels
                label_gaussian_ = label_gaussian[mask_psuedo != 255] #weak labeled target# TODO: Modifications GenGMM
                logits = logits[mask_psuedo != 255]#weak labeled target# TODO: Modifications GenGMM

                mask_one_hot = F.one_hot(label_gaussian_, mean.shape[0])  # m#weak labeled target# TODO: Modifications GenGMM
                logits_max = logits.max(-1)[0]#weak labeled target# TODO: Modifications GenGMM
                logits = mask_one_hot * logits_max + (1 - mask_one_hot) * logits_max  # max over both#weak labeled target# TODO: Modifications GenGMM

                loss = F.cross_entropy(
                    logits,
                    label_gaussian_,
                    weight=class_weight,
                    reduction='none',
                    ignore_index=ignore_index)#weak labeled target# TODO: Modifications GenGMM
            else: #unlabeled target# TODO: Modifications GenGMM
                logits = logits[
                    msk != 255]#unlabeled target# TODO: Modifications GenGMM
                mask_ = mask_[
                    msk != 255]#unlabeled target# TODO: Modifications GenGMM
                mask_one_hot = F.one_hot(mask_, mean.shape[0])  # m#unlabeled target# TODO: Modifications GenGMM
                logits_max = logits.max(-1)[0]#unlabeled target# TODO: Modifications GenGMM
                logits = mask_one_hot * logits_max + (1 - mask_one_hot) * logits_max  # max over both#unlabeled target# TODO: Modifications GenGMM
                loss = F.cross_entropy(
                    logits,
                    mask_,
                    weight=class_weight,
                    reduction='none',
                    ignore_index=ignore_index)#unlabeled target# TODO: Modifications GenGMM

        if kwargs['source'] == False and kwargs[
            'target_memory'] == None:    # unlabeled source# TODO: Modifications GenGMM
            # TODO: Gaussian weighting private labeles ######## plogp  #TODO: weak source labels

            value = softmax(log_prob.reshape(log_prob.shape[0], -1)).reshape(log_prob.shape[0], log_prob.shape[1],
                                                                 log_prob.shape[2]) # unlabeled source# TODO: Modifications GenGMM
            label_gaussian = torch.argmax(value.sum(-1), axis=-1)# unlabeled source# TODO: Modifications GenGMM
            label_gaussian_ = label_gaussian[mask_psuedo != 255]# unlabeled source# TODO: Modifications GenGMM
            logits = logits[mask_psuedo != 255]# unlabeled source# TODO: Modifications GenGMM
            local_gauss = weight_gaussian# unlabeled source# TODO: Modifications GenGMM

            mask_one_hot = F.one_hot(label_gaussian_, mean.shape[0])  # m# unlabeled source# TODO: Modifications GenGMM
            logits_max = logits.max(-1)[0]# unlabeled source# TODO: Modifications GenGMM
            logits = mask_one_hot * logits_max + (1 - mask_one_hot) * logits_max  # max over both# unlabeled source# TODO: Modifications GenGMM
            #             soft_logit = softmax(logits)  ######## plogp

            loss = F.cross_entropy(
                logits,
                label_gaussian_,
                weight=class_weight,
                reduction='none',
                ignore_index=ignore_index)# unlabeled source# TODO: Modifications GenGMM

            loss = loss * local_gauss[mask_psuedo != 255]  ######mask for d^2/zigma^2  # unlabeled source# TODO: Modifications GenGMM

        # apply weights and do the reduction
        if weight is not None:
            weight = weight.float()

        if kwargs['target_memory'] != None and kwargs['source'] == False:# TODO: Modifications GenGMM
            if kwargs['weak_target']:# TODO: Modifications GenGMM
                loss = (loss * local_gauss[mask_psuedo != 255].detach().cuda()).sum() / local_gauss[
                mask_psuedo != 255].detach().cuda().sum()# TODO: Modifications GenGMM
            else:# TODO: Modifications GenGMM
                loss = weight_reduce_loss(
                    loss, weight=weight, reduction=reduction, avg_factor=avg_factor)# TODO: Modifications GenGMM

        else:# TODO: Modifications GenGMM
            loss = weight_reduce_loss(
                loss, weight=weight, reduction=reduction, avg_factor=avg_factor)# TODO: Modifications GenGMM

        if reg_weight > 0. and kwargs['source'] == True:  # TODO: Modifications GenGMM
            contrast_norm = (num_classes) * np.log(num_classes)  # TODO: Modifications GenGMM
            loss += reg_weight * proto_reg(feat, mean, contrast_temp,
                                           contrast_norm=contrast_norm)  # TODO: Modifications GenGMM
        if reg_weight > 0. and kwargs['source'] == False and kwargs[
            'target_memory'] != None:  # TODO: Modifications GenGMM
            contrast_norm = (num_classes) * np.log(num_classes)  # TODO: Modifications GenGMM
            loss += reg_weight * proto_reg(feat, mean, contrast_temp,
                                           contrast_norm=contrast_norm) # TODO: Modifications GenGMM
    if kwargs['source']:  # labeeld source# TODO: Modifications GenGMM
        return loss  # TODO: Modifications GenGMM
    elif kwargs['target_memory'] != None:  # weak labeled target
        if kwargs['weak_target']:
            local_gauss = resize(  # TODO: Modifications GenGMM
                input=local_gauss.reshape(2, 160, 160).unsqueeze(1),
                ## TODO: Modifications GenGMM
                size=classifier.shape[-2:],  # TODO: Modifications GenGMM
                mode='bilinear',  # TODO: Modifications GenGMM
                align_corners=True)  # TODO: Modifications GenGMM

            local_gauss_pseudo = resize(  # TODO: Modifications GenGMM
                input=local_gauss_pseudo.reshape(2, 160, 160).unsqueeze(1).float(),  # TODO: Modifications GenGMM
                size=classifier.shape[-2:],  # TODO: Modifications GenGMM
                mode='nearest')  # TODO: Modifications GenGMM
        else:
            local_gauss_pseudo=torch.tensor(1) # jost to not be empty
            label_gaussian=torch.tensor(1) # jost to not be empty
        return loss, [local_gauss_pseudo.detach().cuda(), label_gaussian.detach().cuda()]  # TODO: Modifications GenGMM

    else: # unlabeled source

        local_gauss = resize(  # TODO: Modifications GenGMM
            input=local_gauss.reshape(2, 160, 160).unsqueeze(1),
            # TODO: Modifications GenGMM
            size=classifier.shape[-2:],  # TODO: Modifications GenGMM
            mode='bilinear',  # TODO: Modifications GenGMM
            align_corners=True)  # TODO: Modifications GenGMM

        label_gaussian = resize(  # TODO: Modifications GenGMM
            input=label_gaussian.reshape(2, 160, 160).unsqueeze(1).float(),  # TODO: Modifications GenGMM
            size=classifier.shape[-2:],  # TODO: Modifications GenGMM
            mode='nearest')  # TODO: Modifications GenGMM
        return loss, [local_gauss.detach().cuda(),
                      label_gaussian.detach().cuda()]  # TODO: Modifications GenGMM


def bank_contrastive(feat,
                     mask,
                     bank=None,
                     mean=None,
                     index=-1,
                     contrast_temp=100.,
                     use_avg_pool=True,
                     scale_min_ratio=0.75,
                     num_classes=19,
                     weight=None,
                     reduction='mean',
                     avg_factor=None,
                     reg_weight=0,
                     ignore_index=255,
                     **kwargs):
    if index >= 0:
        assert isinstance(feat, list), f'feat list expected for index={index}'
        assert isinstance(bank, (list, dict)) \
               and isinstance(bank[index], deque), f'bank list expected for index={index}'
        feat = feat[index]
        bank = bank[index]
        if reg_weight > 0.:
            assert isinstance(mean, (list, dict)), f'mean list expected for index={index}'
            mean = mean[index]
    feat, mask = contrast_preparations(feat, mask, use_avg_pool, scale_min_ratio, num_classes, ignore_index)
    assert bank is not None, 'Parameter `bank` required'
    if reg_weight > 0.:
        assert mean is not None, 'Parameter `mean` required'
        assert not mean.requires_grad
    assert feat.requires_grad
    assert not mask.requires_grad

    if feat.size(0) == 0:
        return torch.tensor(0., requires_grad=True).cuda()

    loss = []
    # calculate per class
    for cls in range(num_classes):
        cls_filter = (mask == cls)
        cls_feat = feat[cls_filter]  # NcxA
        pos, neg = [], []
        for idx in range(num_classes):
            idx_bank = list(bank[idx])
            cls_bank = torch.cat(idx_bank, dim=0)
            bank_sim = cls_feat.mm(cls_bank.permute(1, 0).contiguous()) / contrast_temp
            if idx == cls:
                pos = bank_sim  # NcxMp
            else:
                neg.append(bank_sim.mean(1, keepdim=True))  # NcxMn -> Ncx1
        neg = torch.cat(neg, dim=1)  # Ncx(C-1)
        exp_pos = pos.exp()  # NcxMp
        sum_exp_neg = neg.exp().sum(1, keepdim=True)  # Ncx1
        softmax_term = exp_pos / (exp_pos + sum_exp_neg)  # NcxMp
        cls_loss = - softmax_term.log().mean(dim=1)  # Nc
        loss.append(cls_loss)

    loss = torch.cat(loss, dim=0)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    if reg_weight > 0.:
        contrast_norm = num_classes * np.log(num_classes)
        loss += reg_weight * proto_reg(feat, mean, contrast_temp, contrast_norm=contrast_norm)

    return loss


@LOSSES.register_module()
class ContrastiveLoss(nn.Module):
    """ContrastiveLoss.

    Args:
        use_dist (bool, optional): Whether to use distribution based contrastive loss.
            Defaults to False.
        use_bank (bool, optional): Whether to use memory bank based contrastive loss.
            Defaults to False.
        use_reg (bool, optional): Whether to use regularization term.
            Defaults to False.
        use_avg_pool (bool, optional): Whether to use average pooling for down sampling.
            Defaults to True.
        contrast_temp (double, optional): Temperature used in contrastive loss.
            Defaults to 100.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 use_dist=False,
                 use_bank=False,
                 use_reg=False,
                 use_avg_pool=True,
                 scale_min_ratio=0.75,
                 num_classes=None,
                 contrast_temp=100.,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 reg_relative_weight=1.0,
                 source=True):  # TODO: Modifications
        super(ContrastiveLoss, self).__init__()
        assert (use_dist is False) or (use_bank is False)
        assert num_classes is not None
        self.use_dist = use_dist
        self.use_bank = use_bank
        self.use_reg = use_reg
        self.use_avg_pool = use_avg_pool
        self.scale_min_ratio = scale_min_ratio
        self.contrast_temp = contrast_temp
        self.num_classes = num_classes
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.reg_weight = reg_relative_weight
        self.class_weight = get_class_weight(class_weight)
        self.source = source  # TODO: Modifications

        if self.use_dist:
            self.contrast_criterion = ProtoGMM_loss
        elif self.use_bank:
            self.contrast_criterion = bank_contrastive
        else:
            self.contrast_criterion = proto_contrastive

    def forward(self,
                feat,
                mask,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function."""
        # Parameters mean, covariance are sometimes required
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = feat.new_tensor(self.class_weight)
        else:
            class_weight = None
        if isinstance(feat, list):
            if not isinstance(self.loss_weight, list):
                self.loss_weight = [self.loss_weight for _ in range(len(feat))]
            loss_contrast = [self.loss_weight[i] * self.contrast_criterion(
                feat,
                mask,
                weight=weight,
                index=i,
                contrast_temp=self.contrast_temp,
                use_avg_pool=self.use_avg_pool,
                scale_min_ratio=self.scale_min_ratio,
                num_classes=self.num_classes,
                class_weight=class_weight,
                reduction=reduction,
                avg_factor=avg_factor,
                reg_weight=self.reg_weight if self.use_reg else 0,
                # source=self.source,  # TODO: Modifications
                **kwargs) for i in range(len(feat))]
            loss_contrast = sum(loss_contrast)
        else:
            if kwargs['source']:  # TODO: Modifications GenGMM
                loss_contrast = self.loss_weight * self.contrast_criterion(  # TODO: Modifications GenGMM
                    feat,
                    mask,
                    weight=weight,
                    contrast_temp=self.contrast_temp,
                    use_avg_pool=self.use_avg_pool,
                    scale_min_ratio=self.scale_min_ratio,
                    num_classes=self.num_classes,
                    class_weight=class_weight,
                    reduction=reduction,
                    avg_factor=avg_factor,
                    reg_weight=self.reg_weight if self.use_reg else 0,
                    # source=self.source,  # TODO: Modifications
                    **kwargs)  # TODO: Modifications
                return loss_contrast  # TODO: Modifications GenGMM
            else:  # TODO: Modifications GenGMM
                loss_contrast, label_contrast = self.contrast_criterion(  # TODO: Modifications GenGMM
                    feat,
                    mask,
                    weight=weight,
                    contrast_temp=self.contrast_temp,
                    use_avg_pool=self.use_avg_pool,
                    scale_min_ratio=self.scale_min_ratio,
                    num_classes=self.num_classes,
                    class_weight=class_weight,
                    reduction=reduction,
                    avg_factor=avg_factor,
                    reg_weight=self.reg_weight if self.use_reg else 0,
                    # source=self.source,  # TODO: Modifications
                    **kwargs)  # TODO: Modifications
                loss_contrast = self.loss_weight * loss_contrast  # TODO: Modifications GenGMM
                return loss_contrast, label_contrast  # TODO: Modifications GenGMM
