"""
Schemata to help validation of input.

This code is based on: https://github.com/carbonplan/xarray-schema

which has the following MIT license:

    MIT License

    Copyright (c) 2021 carbonplan

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

In the future, we may be able to replace this module by whatever the best
validation xarray library becomes.
"""

import abc
import operator
from functools import partial
from typing import Any, Callable, Dict, Tuple, Union

import numpy as np
import xarray as xr
import xugrid as xu
from numpy.typing import DTypeLike  # noqa: F401

DimsT = Tuple[Union[str, None]]
ShapeT = Tuple[Union[int, None]]
ChunksT = Union[bool, Dict[str, Union[int, None]]]


OPERATORS = {
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    ">": operator.gt,
}


def partial_operator(op, value):
    # partial doesn't allow us to insert the 1st arg on call, and
    # operators don't work with kwargs, so resort to lambda to swap
    # args a and b around.
    # https://stackoverflow.com/a/37468215
    return partial(lambda b, a: OPERATORS[op](a, b), value)


class ValidationError(Exception):
    pass


class BaseSchema(abc.ABC):
    @abc.abstractmethod
    def validate(self):
        pass

    def __or__(self, other):
        """
        This allows us to write:

        DimsSchema("layer", "y", "x") | DimsSchema("layer")

        And get a SchemaUnion back.
        """
        return SchemaUnion(self, other)


class SchemaUnion:
    """
    Succesful validation only requires a single succes.

    Used to validate multiple options.
    """

    def __init__(self, *args):
        ntypes = len(set(type(arg) for arg in args))
        if ntypes > 1:
            raise TypeError("schemata in a union should have the same type")
        self.schemata = tuple(args)

    def validate(self, obj: Any, **kwargs):
        errors = []
        for schema in self.schemata:
            try:
                schema.validate(obj, **kwargs)
            except ValidationError as e:
                errors.append(e)

        if len(errors) == len(self.schemata):  # All schemata failed
            message = "\n\t" + "\n\t".join(str(error) for error in errors)
            raise ValidationError(f"No option succeeded:{message}")

    def __or__(self, other):
        return SchemaUnion(*self.schemata, other)


class DTypeSchema(BaseSchema):
    def __init__(self, dtype: DTypeLike) -> None:
        if dtype in [
            np.floating,
            np.integer,
            np.signedinteger,
            np.unsignedinteger,
            np.generic,
        ]:
            self.dtype = dtype
        else:
            self.dtype = np.dtype(dtype)

    def validate(self, obj: xr.DataArray, **kwargs) -> None:
        """Validate dtype
        Parameters
        ----------
        dtype : Any
            Dtype of the DataArray.
        """
        # If scalar with value None (optional variable basically)
        if (len(obj.shape) == 0) & (~obj.notnull()).all():
            return

        if not np.issubdtype(obj.dtype, self.dtype):
            raise ValidationError(f"dtype {obj.dtype} != {self.dtype}")


class DimsSchema(BaseSchema):
    def __init__(self, *dims: DimsT) -> None:
        self.dims = dims

    def _fill_in_face_dim(self, obj: Union[xr.DataArray, xu.UgridDataArray]):
        """
        Return dims with a filled in face dim if necessary.
        """
        if "{face_dim}" in self.dims and isinstance(obj, xu.UgridDataArray):
            return tuple(
                (
                    obj.ugrid.grid.face_dimension if i == "{face_dim}" else i
                    for i in self.dims
                )
            )
        else:
            return self.dims

    def validate(self, obj: Union[xr.DataArray, xu.UgridDataArray], **kwargs) -> None:
        """Validate dimensions
        Parameters
        ----------
        dims : Tuple[Union[str, None]]
            Dimensions of the DataArray. `None` may be used as a wildcard value.
        """
        dims = self._fill_in_face_dim(obj)

        if len(dims) != len(obj.dims):
            raise ValidationError(
                f"length of dims does not match: {len(obj.dims)} != {len(dims)}"
            )

        for i, (actual, expected) in enumerate(zip(obj.dims, dims)):
            if expected is not None and actual != expected:
                raise ValidationError(
                    f"dim mismatch in axis {i}: {actual} != {expected}"
                )


class IndexesSchema(BaseSchema):
    """
    Verify indexes, check if no dims with zero size are included and that
    indexes are monotonic. Skips unstructured grid dimensions.
    """

    def __init__(self) -> None:
        pass

    def validate(self, obj: Union[xr.DataArray, xu.UgridDataArray], **kwargs) -> None:
        dims_to_validate = list(obj.dims)

        # Remove face dim from list to validate, as it has no ``indexes``
        # attribute.
        if isinstance(obj, xu.UgridDataArray):
            dims_to_validate.remove(obj.ugrid.grid.face_dimension)

        for dim in dims_to_validate:
            if len(obj.indexes[dim]) == 0:
                raise ValidationError(f"provided dimension {dim} with size 0")

        for dim in dims_to_validate:
            if dim == "y":
                if not obj.indexes[dim].is_monotonic_decreasing:
                    raise ValidationError(
                        f"coord {dim} which is not monotonically decreasing"
                    )

            else:
                if not obj.indexes[dim].is_monotonic_increasing:
                    raise ValidationError(
                        f"coord {dim} which is not monotonically increasing"
                    )


class ShapeSchema(BaseSchema):
    def __init__(self, shape: ShapeT) -> None:
        """
        Validate shape

        Parameters
        ----------
        shape : ShapeT
            Shape of the DataArray. `None` may be used as a wildcard value.
        """
        self.shape = shape

    def validate(self, obj: xr.DataArray, **kwargs) -> None:
        if len(self.shape) != len(obj.shape):
            raise ValidationError(
                f"number of dimensions in shape ({len(obj.shape)}) o!= da.ndim ({len(self.shape)})"
            )

        for i, (actual, expected) in enumerate(zip(obj.shape, self.shape)):
            if expected is not None and actual != expected:
                raise ValidationError(
                    f"shape mismatch in axis {i}: {actual} != {expected}"
                )


class CoordsSchema(BaseSchema):
    """
    Validate coords

    Parameters
    ----------
    coords : dict_like
        coords of the DataArray. `None` may be used as a wildcard value.
    """

    def __init__(
        self,
        coords: Tuple[str],
        require_all_keys: bool = True,
        allow_extra_keys: bool = True,
    ) -> None:
        self.coords = coords
        self.require_all_keys = require_all_keys
        self.allow_extra_keys = allow_extra_keys

    def validate(self, obj: xr.DataArray, **kwargs) -> None:

        coords = list(obj.coords.keys())

        if self.require_all_keys:
            missing_keys = set(self.coords) - set(coords)
            if missing_keys:
                raise ValidationError(f"coords has missing keys: {missing_keys}")

        if not self.allow_extra_keys:
            extra_keys = set(coords) - set(self.coords)
            if extra_keys:
                raise ValidationError(f"coords has extra keys: {extra_keys}")

        for key in self.coords:
            if key not in coords:
                raise ValidationError(f"key {key} not in coords")


class OtherCoordsSchema(BaseSchema):
    def __init__(
        self,
        other: str,
        require_all_keys: bool = True,
        allow_extra_keys: bool = True,
    ):
        self.other = other
        self.require_all_keys = require_all_keys
        self.allow_extra_keys = allow_extra_keys

    def validate(self, obj: xr.DataArray, **kwargs):
        other_obj = kwargs[self.other]
        other_coords = list(other_obj.coords.keys())
        return CoordsSchema(
            other_coords,
            self.require_all_keys,
            self.allow_extra_keys,
        ).validate(obj)


class ValueSchema(BaseSchema, abc.ABC):
    def __init__(
        self,
        operator: str,
        other: Any,
    ):
        self.operator = OPERATORS[operator]
        self.operator_str = operator
        self.other = other


class AllValueSchema(ValueSchema):
    def validate(self, obj: xr.DataArray, **kwargs):
        if isinstance(self.other, str):
            other_obj = kwargs[self.other]
        else:
            other_obj = self.other
        condition = self.operator(obj, other_obj)
        condition = condition | np.isnan(obj)  # ignore nan by setting to True
        if not condition.all():
            raise ValidationError(
                f"values exceed condition: {self.operator_str} {self.other}"
            )


class AnyValueSchema(ValueSchema):
    def validate(self, obj: xr.DataArray, **kwargs):
        if isinstance(self.other, str):
            other_obj = kwargs[self.other]
        else:
            other_obj = self.other
        condition = self.operator(obj, other_obj)
        condition = condition | ~np.isnan(obj)  # ignore nan by setting to False
        if not condition.any():
            raise ValidationError(
                f"no values exceed condition: {self.operator_str} {self.other}"
            )


class NoDataSchema(BaseSchema):
    def __init__(
        self,
        is_notnull: Union[Callable, Tuple[str, Any]] = xr.DataArray.notnull,
    ):

        if isinstance(is_notnull, tuple):
            op, value = is_notnull
            self.is_notnull = partial_operator(op, value)
        else:
            self.is_notnull = is_notnull


class AllNoDataSchema(NoDataSchema):
    def validate(self, obj: xr.DataArray, **kwargs):
        valid = self.is_notnull(obj)
        if ~valid.any():
            raise ValidationError("all nodata")


class NoDataComparisonSchema(BaseSchema):
    def __init__(
        self,
        other: str,
        is_notnull: Union[Callable, Tuple[str, Any]] = xr.DataArray.notnull,
        is_other_notnull: Union[Callable, Tuple[str, Any]] = xr.DataArray.notnull,
    ):
        self.other = other
        if isinstance(is_notnull, tuple):
            op, value = is_notnull
            self.is_notnull = partial_operator(op, value)
        else:
            self.is_notnull = is_notnull

        if isinstance(is_other_notnull, tuple):
            op, value = is_other_notnull
            self.is_other_notnull = partial_operator(op, value)
        else:
            self.is_other_notnull = is_other_notnull


class IdentityNoDataSchema(NoDataComparisonSchema):
    """
    Checks that the NoData values are located at exactly the same locations.
    """

    def validate(self, obj: xr.DataArray, **kwargs):
        other_obj = kwargs[self.other]
        valid = self.is_notnull(obj)
        other_valid = self.is_other_notnull(other_obj)
        if (valid ^ other_valid).any():
            raise ValidationError(f"nodata is not aligned with {self.other}")


class AllInsideNoDataSchema(NoDataComparisonSchema):
    """
    Checks that that notnull values all occur within other.
    """

    def validate(self, obj: xr.DataArray, **kwargs):
        other_obj = kwargs[self.other]
        valid = self.is_notnull(obj)
        other_valid = self.is_other_notnull(other_obj)
        if (valid & ~other_valid).any():
            print(valid)
            print(other_valid)
            raise ValidationError(f"data values found at nodata values of {self.other}")
