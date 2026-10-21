"""
Store constants here that are used across the package. This is to avoid circular
imports and to have a single source of truth for these values.
"""

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

BoolType: TypeAlias = bool
FloatType: TypeAlias = float


@dataclass
class MaskValues:
    """
    Stores mask values for nodata. Special sentinel values can be stored in
    here, such as the -9999.0 for MetaSWAP.
    """

    bool: BoolType = False
    float: FloatType = np.nan
    integer: int = 0
    msw_default: FloatType = -9999.0
