from enum import IntEnum


class SurfaceFlowRouting(IntEnum):
    """
    Usage of surface-flow routing (SFR) ability from the associated file
    \\*.ISD\\*. If ASFR=0, a default ISG file is used, if ASFR=1, a SFR
    compliant version of the ISG is used.
    """

    RIV = 0
    """Information for the RIV package (ASFR=0)."""
    SFR = 1
    """Information for the SFR package (ASFR=1)."""


class CalculationOption(IntEnum):
    """
    Method determines how stream depth and width are calculated for each reach
    in a segment (-).
    """

    SPECIFIED_DEPTH = 1
    """
    Stream depth in each reach is specified at the beginning of a stress period
    and remains unchanged unless flow at the midpoint of a reach is zero, then
    depth is set to zero in that reach;
    """
    MANNING_WIDE_RECTANGULAR = 2
    """
    Stream depth is calculated and updated each iteration of the solver within
    a time step and is calculated from Manning's equation assuming a wide
    rectangular channel, it uses the WIDTH parameter;
    """
    MANNING_EIGHT_POINT = 3
    """
    Stream depth and width is calculated and updated each iteration of the
    solver within a time step and are calculated from Manning's equation using
    an eight-point cross section. These cross-sections are specified in the
    ISC1 and ISC2-section;
    """
    POWER_FUNCTION = 4
    """
    Stream depth and width is calculated and updated each iteration of the
    solver within a time step and are calculated from a power function;
    """
    TABLE = 5
    """
    Stream depth and width is calculated and updated each iteration of the
    solver within a time step and are calculated from a table of values as
    entered in the Q-DEPTH/WIDTH relationships; These are specified in the ISQ1
    and ISQ2-section;
    """


class DiversionOption(IntEnum):
    """
    DIVOPT defines the prioritization system for diversion, such as when
    insufficient water is available to meet all diversion stipulations, and is
    used in conjunction with the value of QFLOW (specified below) (-).
    """

    REDUCE = 1
    """
    If the specified diversion flow (QFLOW) is greater than the flow available
    in the stream segment from which the diversion is made, the diversion is
    reduced to the amount available, which will leave no flow available for
    tributary flow into a downstream tributary of segment IUPSEG;
    """
    PRIORITY = 2
    """
    If the specified diversion flow (QFLOW) is greater than the flow available
    in the stream segment from which the diversion is made, no water is
    diverted from the stream. This approach assumes that once flow in the
    stream is sufficiently low, diversions from the stream cease, and is the
    "priority" algorithm that originally was programmed into the STR1 Package
    (Prudic, 1989);
    """
    FRACTION = 3
    """
    The amount of the diversion is computed as a fraction of the available flow
    in segment IUPSEG; in this case, 0.0 < QFLOW < 1.0;
    """
    FLOOD_CONTROL = 4
    """
    A diversion is made only if the streamflow leaving segment IUPSEG exceeds
    the value of QFLOW. If this occurs, then the quantity of water diverted is
    the excess flow and the quantity that flows from the last reach of segment
    IUPSEG into its downstream tributary (IDOWNSEG) is equal to QFLOW. This
    represents a flood-control type of diversion.
    """
