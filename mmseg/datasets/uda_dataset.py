# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------
#
import json
import os.path as osp

import mmcv
import numpy as np
import torch

from . import CityscapesDataset
from .builder import DATASETS


def get_rcs_class_probs(data_root, temperature):
    with open(osp.join(data_root, 'sample_class_stats.json'), 'r') as of:
        sample_class_stats = json.load(of)
    overall_class_stats = {}
    for s in sample_class_stats:
        s.pop('file')
        for c, n in s.items():
            c = int(c)
            if c not in overall_class_stats:
                overall_class_stats[c] = n
            else:
                overall_class_stats[c] += n
    overall_class_stats = {
        k: v
        for k, v in sorted(
            overall_class_stats.items(), key=lambda item: item[1])
    }
    freq = torch.tensor(list(overall_class_stats.values()))
    freq = freq / torch.sum(freq)
    freq = 1 - freq
    freq = torch.softmax(freq / temperature, dim=-1)

    return list(overall_class_stats.keys()), freq.numpy()


@DATASETS.register_module()
class UDADataset(object):

    def __init__(self, source, target, cfg, weak=False):# TODO: Modifications GenGMM
        self.unlabeled = cfg['unlabeled']  # TODO: Modifications GenGMM
        self.source = source
        self.target = target
        self.ignore_index = target.ignore_index
        self.CLASSES = target.CLASSES
        self.PALETTE = target.PALETTE
        assert target.ignore_index == source.ignore_index
        assert target.CLASSES == source.CLASSES
        assert target.PALETTE == source.PALETTE

        rcs_cfg = cfg.get('rare_class_sampling')
        self.rcs_enabled = rcs_cfg is not None
        if self.rcs_enabled:
            self.rcs_class_temp = rcs_cfg['class_temp']
            self.rcs_min_crop_ratio = rcs_cfg['min_crop_ratio']
            self.rcs_min_pixels = rcs_cfg['min_pixels']

            self.rcs_classes, self.rcs_classprob = get_rcs_class_probs(
                cfg['source']['data_root'], self.rcs_class_temp)
            mmcv.print_log(f'RCS Classes: {self.rcs_classes}', 'mmseg')
            mmcv.print_log(f'RCS ClassProb: {self.rcs_classprob}', 'mmseg')

            with open(
                    osp.join(cfg['source']['data_root'],
                             'samples_with_class.json'), 'r') as of:
                samples_with_class_and_n = json.load(of)
            samples_with_class_and_n = {
                int(k): v
                for k, v in samples_with_class_and_n.items()
                if int(k) in self.rcs_classes
            }
            self.samples_with_class = {}
            for c in self.rcs_classes:
                self.samples_with_class[c] = []
                for file, pixels in samples_with_class_and_n[c]:
                    if pixels > self.rcs_min_pixels:
                        self.samples_with_class[c].append(file.split('/')[-1])
                assert len(self.samples_with_class[c]) > 0
            if self.unlabeled ==True: # TODO: unlabeled source data # TODO: Modifications GenGMM
                #######New split
                # use samples_with_class_ with self.rcs_min_pixels=0 because I do not know the number of pixcels in each image associated with each class
                self.samples_with_class_ = {}# TODO: Modifications GenGMM
                for c in self.rcs_classes:# TODO: Modifications GenGMM
                    self.samples_with_class_[c] = []# TODO: Modifications GenGMM
                    for file, pixels in samples_with_class_and_n[c]:# TODO: Modifications GenGMM
                        if pixels > 0:# TODO: Modifications GenGMM
                            self.samples_with_class_[c].append(file.split('/')[-1])# TODO: Modifications GenGMM
                    assert len(self.samples_with_class_[c]) > 0# TODO: Modifications GenGMM

                self.samples_with_class_original = self.samples_with_class_.copy()# TODO: Modifications GenGMM
                if 'gta' in cfg.source.data_root:# TODO: Modifications GenGMM
                    num_images_per_class = {i: len(self.samples_with_class_[i]) for i in# TODO: Modifications GenGMM
                                            range(19)}  # TODO: Modifications GenGMM
                else:# TODO: Modifications GenGMM
                    num_images_per_class = {i: len(self.samples_with_class_[i]) for i in    # TODO: Modifications GenGMM
                                            list(self.samples_with_class_.keys())}  # TODO: Modifications GenGMM
                sum_ = sum(num_images_per_class.values())# TODO: Modifications GenGMM
                for key, value in num_images_per_class.items():# TODO: Modifications GenGMM
                    num_images_per_class[key] = value / sum_    # TODO: Modifications GenGMM
                all_values = [item for values in self.samples_with_class_.values() for item in values]# TODO: Modifications GenGMM
                from collections import Counter  # TODO: Modifications GenGMM
                import random# TODO: Modifications GenGMM
                percentage = 0.5# TODO: Modifications GenGMM
                num_labeled = int(percentage*source.__len__())# TODO: Modifications GenGMM
                labeled = []# TODO: Modifications GenGMM

                # Read data from JSON file
                with open('/data/labeled_samples_gta'+str(percentage)+'.json', 'r') as json_file:# TODO: Modifications GenGMM
                    labeled_samples = json.load(json_file)   # TODO: Modifications GenGMM
                # convert to integer keys# TODO: Modifications GenGMM
                self.labeled_samples = {int(key): value for key, value in labeled_samples.items()}   # TODO: Modifications GenGMM
                # Create the remaining_samples dictionary containing the non-filtered sublists
                unlabeled_samples = {
                    key: [sublist for sublist in sublists if sublist not in self.labeled_samples.get(key, [])]# TODO: Modifications GenGMM
                    for key, sublists in self.samples_with_class_original.items()# TODO: Modifications GenGMM
                }    # TODO: Modifications GenGMM


                # Read data from JSON file
                with open('/data/unlabeled_samples_gta'+str(percentage)+'.json', 'r') as json_file: # TODO: Modifications GenGMM
                    unlabeled_samples = json.load(json_file) # TODO: Modifications GenGMM
                # convert to integer keys
                self.unlabeled_samples = {int(key): value for key, value in unlabeled_samples.items()} ## TODO: Modifications GenGMM
                all_values_unlabeled = [item for values in self.unlabeled_samples.values() for item in values]   # TODO: Modifications GenGMM
                self.unlabeled_samples = list(set(all_values_unlabeled))  # TODO: Modifications GenGMM

            self.file_to_idx = {}
            for i, dic in enumerate(self.source.img_infos):
                file = dic['ann']['seg_map']
                if isinstance(self.source, CityscapesDataset):
                    file = file.split('/')[-1]
                self.file_to_idx[file] = i

    def get_rare_class_sample(self):
        c = np.random.choice(self.rcs_classes, p=self.rcs_classprob)
        # f1 = np.random.choice(self.samples_with_class[c])   # TODO: Modifications GenGMM
        if self.unlabeled:# TODO:weak labeles target
            f1 = np.random.choice(self.labeled_samples[c])  # TODO: Modifications GenGMM
        else:# TODO:weak labeles target
            f1 = np.random.choice(self.samples_with_class[c])
        i1 = self.file_to_idx[f1]   # find index from source
        s1 = self.source[i1]
        if self.rcs_min_crop_ratio > 0:
            for j in range(10):
                n_class = torch.sum(s1['gt_semantic_seg'].data == c)
                # mmcv.print_log(f'{j}: {n_class}', 'mmseg')
                if n_class > self.rcs_min_pixels * self.rcs_min_crop_ratio:
                    break
                s1 = self.source[i1]
        i2 = np.random.choice(range(len(self.target)))
        s2 = self.target[i2]
        #unlabeled from source   # TODO: Modifications GenGMM
        if self.unlabeled:  # TODO:weak labeles target
            i3 = np.random.choice(range(len(self.unlabeled_samples))) # TODO: Modifications GenGMM
            s3 = self.source[i3]   # TODO: Modifications GenGMM
        else:  #TODO:weak labeles target
            i3 = i2  #TODO:weak labeles target
            s3 = s2.copy() #TODO:weak labeles target

        return {
            **s1, 'unlabeled_source_img_metas': s3['img_metas'],'unlabeled_source_img': s3['img'], 'unlabeled_source_gt_semantic_seg': s3.get('gt_semantic_seg', -1), 'target_img_metas': s2['img_metas'],
            'target_img': s2['img'], 'target_gt_semantic_seg': s2.get('gt_semantic_seg', -1)  # TODO: Modifications GenGMM
        }# TODO: Modifications GenGMM

    def __getitem__(self, idx):
        if self.rcs_enabled:
            return self.get_rare_class_sample()
        else:
            s1 = self.source[idx // len(self.target)]
            s2 = self.target[idx % len(self.target)]
            return {
                **s1, 'target_img_metas': s2['img_metas'],
                'target_img': s2['img'], 'target_gt_semantic_seg': s2.get('gt_semantic_seg', -1)# TODO: Modifications GenGMM
            }

    def __len__(self):
        return len(self.source) * len(self.target)