# Obtained from: https://github.com/BIT-DA/SePiCo
# ---------------------------------------------------------------
# Copyright (c) 2022 BIT-DA. All rights reserved.
# Licensed under the Apache License, Version 2.0
# -------------------------------------------------
# A copy of the license is available at resources/license_SePiCo

from .backbones import *  # noqa: F401,F403
from .builder import (BACKBONES, HEADS, LOSSES, SEGMENTORS, UDA,
                      build_backbone, build_head, build_loss, build_segmentor)
from .decode_heads import *  # noqa: F401,F403
from .losses import *  # noqa: F401,F403
from .necks import *  # noqa: F401,F403
from .segmentors import *  # noqa: F401,F403
from .uda import *  # noqa: F401,F403

__all__ = [
    'BACKBONES', 'HEADS', 'LOSSES', 'SEGMENTORS', 'UDA', 'build_backbone',
    'build_head', 'build_loss', 'build_segmentor'
]
