"""
Store constants here that are used across the package. This is to avoid circular
imports and to have a single source of truth for these values.
"""

import numpy as np


from dataclasses import dataclass


@dataclass
class MaskValues:
    """
    Stores mask values for nodata. Special sentinel values can be stored in
    here, such as the -9999.0 for MetaSWAP.
    """

    float = np.nan
    integer = 0
    msw_default = -9999.0