# Obtained from: https://github.com/BIT-DA/SePiCo
# Modifications: GenGMM

# ---------------------------------------------------------------
# Copyright (c) 2022 BIT-DA. All rights reserved.
# Licensed under the Apache License, Version 2.0
# -------------------------------------------------
# A copy of the license is available at resources/license_SePiCo
# Noted compute_log_prob,online_contrast, update_GMM,_dequeue_and_enqueue_k obtained from https://github.com/leonnnop/GMMSeg
# A copy of the license is available at resources/license_GMMSeg

import torch
import torch.nn as nn  # TODO: Modifications GenGMM
import torch.utils.data
import torch.distributed
import torch.backends.cudnn
from collections import deque
from mmseg.utils.GMMSeg import distributed_sinkhorn_wograd, shifted_var, init_weights, l2_normalize, momentum_update, rnd_sample  # TODO: Modifications GenGMM
from timm.models.layers import trunc_normal_  # TODO: Modifications GenGMM
from einops import rearrange, repeat # TODO: Modifications GenGMM
from mmseg.utils.distributions import MultivariateNormalDiag  # TODO: Modifications GenGMM


class ProtoEstimator:
    def __init__(self, dim, class_num, memory_length=100, resume=""):
        super(ProtoEstimator, self).__init__()
        self.dim = dim
        self.class_num = class_num

        # init mean and covariance
        if resume:
            print("Loading checkpoint from {}".format(resume))
            checkpoint = torch.load(resume, map_location=torch.device('cpu'))
            self.CoVariance = checkpoint['CoVariance'].cuda()
            self.Ave = checkpoint['Ave'].cuda()
            self.Amount = checkpoint['Amount'].cuda()
            if 'MemoryBank' in checkpoint:
                self.MemoryBank = checkpoint['MemoryBank'].cuda()
        else:
            self.CoVariance = torch.zeros(self.class_num, self.dim).cuda()
            self.Ave = torch.zeros(self.class_num, self.dim).cuda()
            self.Amount = torch.zeros(self.class_num).cuda()
            self.MemoryBank = [deque([self.Ave[cls].unsqueeze(0).detach()], maxlen=memory_length)
                               for cls in range(self.class_num)]
        self.embedding_dim=64 # TODO: Modifications GenGMM
        self.num_components = 5 # TODO: Modifications GenGMM
        self.feat_norm = nn.LayerNorm(self.embedding_dim).cuda()   # TODO: Modifications GenGMM
        self.mask_norm = nn.LayerNorm(self.class_num).cuda()   # TODO: Modifications GenGMM
        self.means = nn.Parameter(torch.zeros(self.class_num, self.num_components, self.embedding_dim),
                                  requires_grad=False) # TODO: Modifications GenGMM
        trunc_normal_(self.means, std=0.02)  # TODO: Modifications GenGMM
        self.num_prob_n = self.num_components  # TODO: Modifications GenGMM
        self.diagonal = nn.Parameter(torch.ones(self.class_num, self.num_components, self.embedding_dim),
                                     requires_grad=False) # Modifications GenGMM
        self.eye_matrix = nn.Parameter(torch.ones(self.embedding_dim), requires_grad=False) # Modifications GenGMM
        self.factors = [2, 1, 1]    # TODO: Modifications GenGMM
        self.iteration_counter = nn.Parameter(torch.zeros(1), requires_grad=False) # TODO: Modifications GenGMM
        self.update_GMM_interval = 5  # TODO: Modifications GenGMM
        self.max_sample_size = 20  # TODO: Modifications GenGMM
        self.K = 32000 # TODO: Modifications GenGMM
        self.queue = torch.randn(self.class_num * self.num_components, self.embedding_dim, self.K) # TODO: Modifications GenGMM
        self.queue = nn.functional.normalize(self.queue, dim=-2) # TODO: Modifications GenGMM
        self.queue_ptr = torch.zeros(self.class_num * self.num_components, dtype=torch.long) # TODO: Modifications GenGMM
        
        self.K_target = 16000  # TODO: Modifications GenGMM
        self.queue_target = torch.randn(self.class_num , self.embedding_dim,
                                 self.K_target)  # TODO: Modifications GenGMM
        self.queue_target = nn.functional.normalize(self.queue_target, dim=-2)  # TODO: Modifications GenGMM
        self.queue_ptr_target = torch.zeros(self.class_num, dtype=torch.long)  # TODO: Modifications GenGMM
        self.Ks_target = torch.tensor([self.K_target for _c in range(self.class_num)], dtype=torch.long)  # TODO: Modifications GenGMM
        self.means_target = nn.Parameter(torch.zeros(self.class_num, self.embedding_dim),
                                  requires_grad=False)  # TODO: Modifications GenGMM


        self.Ks = torch.tensor([self.K for _c in range(self.class_num * self.num_components)], dtype=torch.long) # TODO: Modifications GenGMM
        self.gamma_mean = 0.999 # TODO: Modifications GenGMM
        self.gamma_cov = 0 # TODO: Modifications GenGMM

    # TODO: Modifications, update GMM's componnents per category
    def update_proto(self, features, labels):
        """Update variance and mean

        Args:
            features (Tensor): feature map, shape [B, A, H, W]  N = B*H*W
            labels (Tensor): shape [B, 1, H, W]
        """
        with torch.no_grad():# TODO: Modifications
#             features = self.feat_norm(features.detach())  # * n, d  # TODO: Modifications
            features = l2_normalize(features)  # TODO: Modifications
            self.means.data.copy_(l2_normalize(self.means))  # TODO: Modifications
            _log_prob = self.compute_log_prob(features.cpu()) # TODO: Modifications
            final_probs = _log_prob.contiguous().view(-1, self.class_num, self.num_prob_n) # TODO: Modifications
            _m_prob = torch.amax(final_probs.cuda(), dim=-1) # TODO: Modifications
            out_seg = self.mask_norm(_m_prob) # TODO: Modifications
            # out_seg = rearrange(out_seg, "(b h w) k -> b k h w", b=base_feature.shape[0], h=base_feature.shape[2]) # TODO: Modifications

            contrast_logits, contrast_target, qs = self.online_contrast(labels, final_probs.cuda(), features, out_seg)  # TODO: Modifications

            _c_mem = features  # TODO: Modifications
            _gt_seg_mem = labels  # TODO: Modifications
            _qs = qs   # TODO: Modifications

            unique_c_list = _gt_seg_mem.unique().int() # TODO: Modifications
            for k in unique_c_list: # TODO: Modifications
                if k == 255: continue # TODO: Modifications
                self._dequeue_and_enqueue_k(k.item(), _c_mem, _qs.bool(), (_gt_seg_mem == k.item()))  # TODO: Modifications

            # * EM
            if self.iteration_counter % self.update_GMM_interval == 0:  # TODO: Modifications
                self.update_GMM(unique_c_list)   # TODO: Modifications

        return out_seg, contrast_logits, contrast_target# TODO: Modifications GenGMM




    @torch.no_grad()
    def update_target_bank(self,features, labels, num = 10):  # TODO: Modifications GenGMM
        features = self.feat_norm(features)  # * n, d  # TODO: Modifications GenGMM
        features = l2_normalize(features)   # TODO: Modifications GenGMM
        unique_c_list = labels.unique().int()  # TODO: Modifications GenGMM
        for k in unique_c_list:  ## TODO: Modifications GenGMM
            if k == 255: continue  # TODO: Modifications GenGMM
            features_selected = features[labels == k] # TODO: Modifications GenGMM
            # labels = labels[labels == k]# TODO: Modifications GenGMM
            mean_f = features_selected.mean(0)# TODO: Modifications GenGMM
            mean_f = l2_normalize(mean_f.unsqueeze(0)).T# TODO: Modifications GenGMM
            similarity = features_selected.mm(mean_f) #  dot product# TODO: Modifications GenGMM
            if similarity.shape[0]<num: continue# TODO: Modifications GenGMM
            values, indices = torch.topk(similarity, k=num, dim=0)# TODO: Modifications GenGMM
            selected = features_selected[indices.squeeze(1)]# TODO: Modifications GenGMM

            ptr = int(self.queue_ptr_target[k])# TODO: Modifications GenGMM
            if ptr + num >= self.Ks_target[k]:# TODO: Modifications GenGMM
                _fir = self.Ks_target[k] - ptr# TODO: Modifications GenGMM
                _sec = num - self.Ks_target[k] + ptr# TODO: Modifications GenGMM
                self.queue_target[k, :, ptr:self.Ks_target[k]] = selected[:_fir].T# TODO: Modifications GenGMM
                self.queue_target[k, :, :_sec] = selected[_fir:].T# TODO: Modifications GenGMM
            else:# TODO: Modifications GenGMM
                self.queue_target[k, :, ptr:ptr + num] = selected.T# TODO: Modifications GenGMM
            ptr = (ptr + num) % self.Ks_target[k].item()  # move pointer # TODO: Modifications GenGMM
            self.queue_ptr_target[k] = ptr# TODO: Modifications GenGMM


        
    @torch.no_grad()# TODO: Modifications GenGMM
    def _dequeue_and_enqueue_k(self, _c, _c_embs, _c_cluster_q, _c_mask):   # TODO: Modifications GenGMM

        if _c_mask is None: _c_mask = torch.ones(_c_embs.shape[0]).detach_()# TODO: Modifications GenGMM

        _k_max_sample_size = self.max_sample_size# TODO: Modifications GenGMM
        _embs = _c_embs[_c_mask > 0]   # embedding that their ground truth labels are k (_c_mask > 0)# TODO: Modifications GenGMM
        _cluster = _c_cluster_q[_c_mask > 0]  # _c_cluster_q already contains the correct qu's and know it considers those that are correct and their labels are equal to k# TODO: Modifications GenGMM
        for q_index in range(self.num_components):# TODO: Modifications GenGMM
            _q_ptr = _c * self.num_components + q_index  # it is used to convert to 1 to n*componnent labels# TODO: Modifications GenGMM
            ptr = int(self.queue_ptr[_q_ptr])# TODO: Modifications GenGMM

            if torch.sum(_cluster[:, q_index]) == 0: continue# TODO: Modifications GenGMM
            assert _cluster[:, q_index].shape[0] == _embs.shape[0]# TODO: Modifications GenGMM
            _q_embs = _embs[_cluster[:, q_index]]  # find embedding which are correct# TODO: Modifications GenGMM

            _q_sample_size = _q_embs.shape[0]# TODO: Modifications GenGMM
            assert _q_sample_size == torch.sum(_cluster[:, q_index])# TODO: Modifications GenGMM

            if self.max_sample_size != -1 and _q_sample_size > _k_max_sample_size:# TODO: Modifications GenGMM
                _rnd_sample = rnd_sample(_q_sample_size, _k_max_sample_size, _uniform=True, _device=_c_embs.device)# TODO: Modifications GenGMM
                _q_embs = _q_embs[_rnd_sample, ...]# TODO: Modifications GenGMM
                _q_sample_size = _k_max_sample_size# TODO: Modifications GenGMM

            # replace the embs at ptr (dequeue and enqueue)# TODO: Modifications GenGMM
            if ptr + _q_sample_size >= self.Ks[_q_ptr]:# TODO: Modifications GenGMM
                _fir = self.Ks[_q_ptr] - ptr# TODO: Modifications GenGMM
                _sec = _q_sample_size - self.Ks[_q_ptr] + ptr# TODO: Modifications GenGMM
                self.queue[_q_ptr, :, ptr:self.Ks[_q_ptr]] = _q_embs[:_fir].T# TODO: Modifications GenGMM
                self.queue[_q_ptr, :, :_sec] = _q_embs[_fir:].T# TODO: Modifications GenGMM
            else:# TODO: Modifications GenGMM
                self.queue[_q_ptr, :, ptr:ptr + _q_sample_size] = _q_embs.T# TODO: Modifications GenGMM

            ptr = (ptr + _q_sample_size) % self.Ks[_q_ptr].item()  # move pointer# TODO: Modifications GenGMM
            self.queue_ptr[_q_ptr] = ptr# TODO: Modifications GenGMM

    @torch.no_grad()# TODO: Modifications GenGMM
    def update_GMM(self, unique_c_list):   # TODO: Modifications GenGMM
        components = self.means.data.clone()# TODO: Modifications GenGMM
        covs = self.diagonal.data.clone()# TODO: Modifications GenGMM

        for _c in unique_c_list:# TODO: Modifications GenGMM
            if _c == 255: continue# TODO: Modifications GenGMM
            _c = _c if isinstance(_c, int) else _c.item()# TODO: Modifications GenGMM

            for _p in range(self.num_components):# TODO: Modifications GenGMM
                _p_ptr = _c * self.num_components + _p# TODO: Modifications GenGMM
                _mem_fea_q = self.queue[_p_ptr, :, :self.Ks[_c]].transpose(-1, -2)  # n,d# TODO: Modifications GenGMM

                f = l2_normalize(torch.sum(_mem_fea_q, dim=0))  # d,# TODO: Modifications GenGMM

                new_value = momentum_update(old_value=components[_c, _p, ...], new_value=f, momentum=self.gamma_mean,
                                            debug=False)# TODO: Modifications GenGMM
                components[_c, _p, ...] = new_value# TODO: Modifications GenGMM

                _shift_fea = _mem_fea_q - f[None, ...]  # * n, d# TODO: Modifications GenGMM

                _cov = shifted_var(_shift_fea, rowvar=False)# TODO: Modifications GenGMM
                _cov = _cov + 1e-2 * self.eye_matrix# TODO: Modifications GenGMM
                _cov = _cov.sqrt()# TODO: Modifications GenGMM

                new_covariance = momentum_update(old_value=covs[_c, _p, ...], new_value=_cov, momentum=self.gamma_cov,
                                                 debug=False)# TODO: Modifications GenGMM
                covs[_c, _p, ...] = new_covariance# TODO: Modifications GenGMM

        self.means = nn.Parameter(components, requires_grad=False)# TODO: Modifications GenGMM
        self.diagonal = nn.Parameter(covs, requires_grad=False)# TODO: Modifications GenGMM
        # * NOTE: need not to sync across gpus. memory is shared across all gpus


    def online_contrast(self, gt_seg, simi_logits, _c, out_seg): # TODO: Modifications GenGMM
        # find pixels that are correctly classified
        pred_seg = torch.max(out_seg, 1)[1]# TODO: Modifications GenGMM
        mask = (gt_seg == pred_seg.view(-1))  #find the correct predictions# TODO: Modifications GenGMM

        # compute logits
        contrast_logits = simi_logits.flatten(1) # * n, c*p# TODO: Modifications GenGMM
        contrast_target = gt_seg.clone().float()# TODO: Modifications GenGMM

        return_qs = torch.zeros(size=(simi_logits.shape[0], self.num_components), device=gt_seg.device)# TODO: Modifications GenGMM
        # clustering for each class
        for k in gt_seg.unique().long():# TODO: Modifications GenGMM
            if k == 255: continue# TODO: Modifications GenGMM
            # get initial assignments for the k-th class
            init_q = simi_logits[:, k, :]# TODO: Modifications GenGMM
            init_q = init_q[gt_seg == k, ...] # n,p# TODO: Modifications GenGMM
            init_q = init_q[:,:self.num_components]# TODO: Modifications GenGMM
            init_q = init_q / torch.abs(init_q).max()# TODO: Modifications GenGMM

            # * init_q: [gt_n, p]
            # clustering q.shape = n x self.num_components
            q, indexs = distributed_sinkhorn_wograd(init_q)# TODO: Modifications GenGMM
            try:
                assert torch.isnan(q).int().sum() <= 0# TODO: Modifications GenGMM
            except:
                # * process nan
                q[torch.isnan(q)] = 0# TODO: Modifications GenGMM
                indexs[torch.isnan(q).int().sum(dim=1)>0] = 255 - (self.num_prob_n * k)# TODO: Modifications GenGMM

            # binary mask for pixels of the k-th class
            m_k = mask[gt_seg == k]# TODO: Modifications GenGMM

            m_k_tile = repeat(m_k, 'n -> n tile', tile=self.num_components)# TODO: Modifications GenGMM
            # mask the incorrect q with zero
            q = q * m_k_tile  # n x self.num_prob_n# TODO: Modifications GenGMM

            contrast_target[gt_seg == k] = indexs.float() + (self.num_prob_n * k)# TODO: Modifications GenGMM

            return_qs[gt_seg == k] = q# TODO: Modifications GenGMM

        return contrast_logits, contrast_target, return_qs# TODO: Modifications GenGMM
    def compute_log_prob(self, _fea):  # TODO: Modifications GenGMM
        covariances = self.diagonal.detach_() # * c,p,d,d

        _prob_n = []
        _n_group = _fea.shape[0] // self.factors[0]
        _c_group = self.class_num // self.factors[1]
        for _c in range(0,self.class_num,_c_group):# TODO: Modifications GenGMM
            _prob_c = []# TODO: Modifications GenGMM
            _c_means = self.means[_c:_c+_c_group]# TODO: Modifications GenGMM
            _c_covariances = covariances[_c:_c+_c_group]# TODO: Modifications GenGMM

            _c_gauss = MultivariateNormalDiag(_c_means.view(-1, self.embedding_dim), scale_diag=_c_covariances.view(-1,self.embedding_dim)) # * c*p multivariate gaussian# TODO: Modifications GenGMM
            for _n in range(0,_fea.shape[0],_n_group):# TODO: Modifications GenGMM
                _prob_c.append(_c_gauss.log_prob(_fea[_n:_n+_n_group,None,...]))# TODO: Modifications GenGMM
            _c_probs = torch.cat(_prob_c, dim=0) # n, cp# TODO: Modifications GenGMM
            _c_probs = _c_probs.contiguous().view(_c_probs.shape[0], -1, self.num_prob_n)# TODO: Modifications GenGMM
            _prob_n.append(_c_probs)# TODO: Modifications GenGMM
        probs = torch.cat(_prob_n, dim=1)# TODO: Modifications GenGMM

        return probs.contiguous().view(probs.shape[0],-1)# TODO: Modifications GenGMM
    def save_proto(self, path):
        torch.save({'CoVariance': self.CoVariance.cpu(),
                    'Ave': self.Ave.cpu(),
                    'Amount': self.Amount.cpu()
                    }, path)
