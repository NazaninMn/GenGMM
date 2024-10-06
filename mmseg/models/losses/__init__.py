# Obtained from: https://github.com/BIT-DA/SePiCo
# ---------------------------------------------------------------
# Copyright (c) 2022 BIT-DA. All rights reserved.
# Licensed under the Apache License, Version 2.0
# -------------------------------------------------
# A copy of the license is available at resources/license_SePiCo


from .accuracy import Accuracy, accuracy
from .cross_entropy_loss import (CrossEntropyLoss, binary_cross_entropy,
                                 cross_entropy, mask_cross_entropy)
from .contrastive_loss import ContrastiveLoss
from .utils import reduce_loss, weight_reduce_loss, weighted_loss

__all__ = [
    'accuracy', 'Accuracy', 'cross_entropy', 'binary_cross_entropy',
    'mask_cross_entropy', 'CrossEntropyLoss', 'reduce_loss',
    'weight_reduce_loss', 'weighted_loss', 'ContrastiveLoss'
]
