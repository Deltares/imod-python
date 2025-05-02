import textwrap
from typing import Optional
from warnings import warn

from imod.mf6.simulation import Modflow6Simulation
from imod.typing import GridDataArray


def get_label_array(
    simulation: Modflow6Simulation,
    npartitions: int,
    weights: Optional[GridDataArray] = None,
):
    """
    To preserve backwards compatibility for older training scripts.
    """

    from imod.prepare.partition import create_partition_labels

    msg = textwrap.dedent(
        """get_label_array is deprecated, the function has been moved and
        renamed to imod.prepare.create_partition_labels."""
    )
    warn(
        msg,
        DeprecationWarning,
    )
    return create_partition_labels(
        simulation=simulation,
        npartitions=npartitions,
        weights=weights,
    )
