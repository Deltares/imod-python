"""
Store constants here that are used across the package. This is to avoid circular
imports and to have a single source of truth for these values.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class MaskValues:
    """
    Stores mask values for nodata. Special sentinel values can be stored in
    here, such as the -9999.0 for MetaSWAP.
    """

    bool = False
    float = np.nan
    integer = 0
    msw_default = -9999.0
