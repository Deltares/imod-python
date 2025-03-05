API design FAQ
==============

This document contains answers to common questions we get on our API design. 

Why does the ``"layer"`` coordinate need to start at 1 and not 0?
-----------------------------------------------------------------

There is a difference between coordinate labels and indices. In Python, indices
start at 0. Coordinate labels have no direct meaning to Python, so instead iMOD
Python decides how coordinates should be labeled. The ``"layer"`` coordinate
contains labels, just like the ``"x"`` and ``"y"`` coordinates. Because these
labels are not used for indexing, we decided to keep the ``"layer"`` labels the
same as the MODFLOW6 layers. MODFLOW6 is written in Fortran, in which the
indices start at 1.
