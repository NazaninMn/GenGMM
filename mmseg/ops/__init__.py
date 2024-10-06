# Obtained from: https://github.com/BIT-DA/SePiCo
# ---------------------------------------------------------------
# Copyright (c) 2022 BIT-DA. All rights reserved.
# Licensed under the Apache License, Version 2.0
# -------------------------------------------------
# A copy of the license is available at resources/license_SePiCo

from .encoding import Encoding
from .wrappers import Upsample, resize

__all__ = ['Upsample', 'resize', 'Encoding']
