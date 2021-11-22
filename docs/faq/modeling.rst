Groundwater Modeling with iMOD
==============================

General
-------

What are specific yield, specific storage, storage coefficient (storativity)? 
*****************************************************************************

There are multiple terms describing aquifer water storage. Their use can be
confusing, especially when combined with MODFLOW's concept of convertible
layers for unconfined flow and common workarounds.

    Specific yield
        Specific yield describes the volume of storage change (|m3|) for a unit
        length of head change (m), per unit area (|m2|), while becoming
        unsatured. It describes the process of water draining out of a porous
        medium and being replaced by air. As such, it does not occur in
        confined aquifers, as no air can enter the pore space there. It it less
        or equal to the effective porosity. Specific yield values vary based on
        the effective porosity and soil capillary suction which keeps all the
        water from draining out. Values commonly range from 0.05 to 0.40. Unit:
        ((|m3| / m) / |m2|) or (m / m) or (-).
        
    Specific storage
        Specific storage describes the volume of storage change (|m3|) for a
        unit length of head change (m), per unit volume of porous medium
        (|m3|), while remaining saturated. It describes the process of water
        draining out of a porous Specific storage describes the head change (m)
        in a (unit) volume (m) of porous medium for a volume of storage change
        (|m3|) while remaining saturated. It describes two processes.  The
        first is the elastic response of the aquifer: as the head increases,
        the porous material is compressed and this creates a slightly larger
        pore space. The second is the (negligible) compressibility of water.
        Specific storage values vary based on the compressibility or stiffness
        of the soil. Specific storage values are generally very small compared
        to specific yield values; (stiff) sandy soils estimated to have values
        of 1.0e-5 to 1.0E-4. However, soft clays or peats may show considerably
        more compressibility and larger specific storage values. Unit: ((|m3| /
        m) / (|m3|) or (|m-1|).
  
    Storage coefficient
        The storage coefficient is the specific storage of the porous material
        multiplied by the layer thickness, per unit area. This gives the
        storage response of the entire aquifer thickness. Unit: ((|m3| /
        m) / (|m3|) m ) or (m / m) or (-).

    Storativity
        Synonym for storage coefficient.
        
.. note::

    * specific yield and specific storage: both contain the word **specific**,
      and they are both specific to the material and describe a property for a
      unit volume of material (compare with specific mass or volume). 
    * storativity: compare with transmissivity, which also describes
      another property of the porous medium (conductivity) for the entire
      aquifer thickness.
    * This arguably makes storativity a better term than storage coefficient, as
      "coefficient" is not very informative.
      
See also the `Wikipedia article`_ for more background.

In MODFLOW models, we often set layers to be inconvertible even though they
represent unconfined aquifers. This has a numerical reason: for MODFLOW, a
convertible layer has a transmissivity that varies with head -- the ground
water table in case of an unconfined aquifer -- and this creates non-linearity.
Linear problems are (much) easier to solve than non-linear problems, and so
non-linearity can result in non-convergence.

In many cases a pragmatic solution is acceptable: when the groundwater table
show little relative change, the transmissivity of an unconfined aquifer can be
approximated by a constant. We set all layers of the model to be inconvertible,
which restores linearity and will result in a model that converges better.
However, MODFLOW uses specific storage or storativity rather than specific
yield when computing storage terms for inconvertible (confined) layers.

When using storativity in the input settings to represent specific yield using
inconvertible layers, the specific yield can be entered directly. However, when
using specific storage to represent specific yield using inconvertible layers,
**specific yield must be divided by layer thickness**, as MODFLOW will
internally multiply the entered number by layer thickness to compute a
storativity and this will result in a much greater specific yield.

Furthermore, when representing specific yield with either specific storage or
storativity, **the value must only be assigned to a single layer**. All other
layers must be given normal specific storage terms representing only the
compressibility of the aquifer and the water. If this is not done, we
effectively simulate a much greater specific yield: most of the storage occurs
at the water table, where air is replaced by water.

.. note::

    In the BCF package of older MODFLOW versions, storage can only be
    configured via the storage coefficient values; in the LPF package, storage
    can only be configured via specific storage values. In MODFLOW2005's LPF
    package and MODFLOW6's NPF package, storage values are read as **specific**
    storage by default and will be multiplied by layer thickness by MODFLOW
    internally.  By using a "STORAGECOEFFICIENT" option, MODFLOW will read the
    values as storativity instead. In LPF and NPF, specific yield is a separate
    parameter (and will only be used if there are convertible layers).
    
MODFLOW6
--------

Use a river infiltration factor in MODFLOW6
*******************************************
    
.. |m-1| replace:: m\ :sup:`-1`\
.. |m2| replace:: m\ :sup:`2`\
.. |m3| replace:: m\ :sup:`3`\
.. _Wikipedia article: https://en.wikipedia.org/wiki/Specific_storage