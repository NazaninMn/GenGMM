# Obtained from: https://github.com/BIT-DA/SePiCo

# ---------------------------------------------------------------
# Copyright (c) 2022 BIT-DA. All rights reserved.
# Licensed under the Apache License, Version 2.0
# -------------------------------------------------
# A copy of the license is available at resources/license_SePiCo

from .collect_env import collect_env
from .logger import get_root_logger

__all__ = ['get_root_logger', 'collect_env']
