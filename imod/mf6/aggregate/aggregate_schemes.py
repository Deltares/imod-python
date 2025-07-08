from typing import Callable

import numpy as np
from pydantic.dataclasses import dataclass

from imod.common.utilities.dataclass_type import (
    _CONFIG,
    DataclassType,
)


@dataclass(config=_CONFIG)
class EmptyAggregationMethod(DataclassType):
    pass


@dataclass(config=_CONFIG)
class RiverAggregationMethod(DataclassType):
    """
    Object containing aggregation methods for the :class:`imod.mf6.River` package.
    This can be provided to the ``aggregate`` method to regrid with custom
    settings.

    Parameters
    ----------
    stage: Callable, default np.nanmean
    conductance: Callable, default np.nansum
    bottom_elevation: Callable, default np.nanmean
    concentration: Callable, default np.nanmean

    Examples
    --------
    Aggregate with custom settings:

    >>> agg_method = RiverRegridMethod(stage=np.nanmedian)
    >>> planar_data = riv.aggregate(aggregate_method=agg_method)

    """

    stage: Callable = np.nanmean
    conductance: Callable = np.nansum
    bottom_elevation: Callable = np.nanmean
    concentration: Callable = np.nanmean


@dataclass(config=_CONFIG)
class DrainageAggregationMethod(DataclassType):
    """
    Object containing aggregation methods for the :class:`imod.mf6.Drainage` package.
    This can be provided to the ``aggregate`` method to regrid with custom
    settings.

    Parameters
    ----------
    elevation: Callable, default np.nanmean
    conductance: Callable, default np.nansum
    concentration: Callable, default np.nanmean

    Examples
    --------
    Aggregate with custom settings:

    >>> agg_method = DrainageRegridMethod(elevation=np.nanmedian)
    >>> planar_data = drn.aggregate(aggregate_method=agg_method)

    """

    elevation: Callable = np.nanmean
    conductance: Callable = np.nansum
    concentration: Callable = np.nanmean


@dataclass(config=_CONFIG)
class GeneralHeadBoundaryAggregationMethod(DataclassType):
    """
    Object containing aggregation methods for the
    :class:`imod.mf6.GeneralHeadBoundary` package. This can be provided to the
    ``aggregate`` method to regrid with custom settings.

    Parameters
    ----------
    head: Callable, default np.nanmean
    conductance: Callable, default np.nansum
    concentration: Callable, default np.nanmean

    Examples
    --------
    Aggregate with custom settings:

    >>> agg_method = GeneralHeadBoundaryAggregationMethod(head=np.nanmedian)
    >>> planar_data = ghb.aggregate(aggregate_method=agg_method)

    """

    head: Callable = np.nanmean
    conductance: Callable = np.nansum
    concentration: Callable = np.nanmean


@dataclass(config=_CONFIG)
class RechargeAggregationMethod(DataclassType):
    """
    Object containing aggregation methods for the :class:`imod.mf6.Recharge`
    package. This can be provided to the ``aggregate`` method to regrid with
    custom settings.

    Parameters
    ----------
    rate: Callable, default np.nansum
    concentration: Callable, default np.nanmean

    Examples
    --------
    Aggregate with custom settings:

    >>> agg_method = RechargeAggregationMethod(rate=np.nanmedian)
    >>> planar_data = rch.aggregate(aggregate_method=agg_method)

    """

    rate: Callable = np.nansum
    concentration: Callable = np.nanmean
