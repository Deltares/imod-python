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
from typing import Any, Callable, Dict, Mapping, Tuple, Union

import numpy as np
import xarray as xr
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
            message = "\n".join(str(error) for error in errors)
            raise ValidationError(f"No option succeeded: {message}")

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
        if not np.issubdtype(obj.dtype, self.dtype):
            raise ValidationError(f"dtype {obj.dtype} != {self.dtype}")


class DimsSchema(BaseSchema):
    def __init__(self, *dims: DimsT) -> None:
        self.dims = dims

    def validate(self, obj: xr.DataArray, **kwargs) -> None:
        """Validate dimensions
        Parameters
        ----------
        dims : Tuple[Union[str, None]]
            Dimensions of the DataArray. `None` may be used as a wildcard value.
        """
        if len(self.dims) != len(obj.dims):
            raise ValidationError(
                f"length of dims does not match: {len(obj.dims)} != {len(self.dims)}"
            )

        # TODO: Add check for dims with size 0 on object

        for i, (actual, expected) in enumerate(zip(obj.dims, self.dims)):
            if expected is not None and actual != expected:
                raise ValidationError(
                    f"dim mismatch in axis {i}: {actual} != {expected}"
                )


class ShapeSchema(BaseSchema):
    def __init__(self, shape: ShapeT) -> None:
        self.shape = shape

    def validate(self, obj: xr.DataArray, **kwargs) -> None:
        """Validate shape
        Parameters
        ----------
        shape : ShapeT
            Shape of the DataArray. `None` may be used as a wildcard value.
        """
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
    def __init__(
        self,
        coords: Mapping[str, Any],
        require_all_keys: bool = True,
        allow_extra_keys: bool = True,
    ) -> None:
        self.coords = coords
        self.require_all_keys = require_all_keys
        self.allow_extra_keys = allow_extra_keys

    def validate(self, obj: xr.DataArray, **kwargs) -> None:
        """Validate coords
        Parameters
        ----------
        coords : dict_like
            coords of the DataArray. `None` may be used as a wildcard value.
        """
        coords = obj.coords

        if self.require_all_keys:
            missing_keys = set(self.coords) - set(coords)
            if missing_keys:
                raise ValidationError(f"coords has missing keys: {missing_keys}")

        if not self.allow_extra_keys:
            extra_keys = set(coords) - set(self.coords)
            if extra_keys:
                raise ValidationError(f"coords has extra keys: {extra_keys}")

        for key, da_schema in self.coords.items():
            if key not in coords:
                raise ValidationError(f"key {key} not in coords")
            else:
                da_schema.validate(coords[key])


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
        return CoordsSchema(
            other_obj.coords,
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
        if not condition.any():
            raise ValidationError(
                f"no values exceed condition: {self.operator_str} {self.other}"
            )


class NoDataSchema(BaseSchema):
    def __init__(
        self,
        other: str,
        is_nodata: Union[Callable, Tuple[str, Any]] = xr.DataArray.notnull,
        is_other_nodata: Union[Callable, Tuple[str, Any]] = xr.DataArray.notnull,
    ):
        self.other = other
        if isinstance(is_nodata, tuple):
            op, value = is_nodata
            self.is_nodata = partial(OPERATORS[op], value)
        else:
            self.is_nodata = is_nodata

        if isinstance(is_other_nodata, tuple):
            op, value = is_other_nodata
            self.is_nodata = partial(OPERATORS[op], value)
        else:
            self.is_other_nodata = is_other_nodata

    def validate(self, obj: xr.DataArray, **kwargs):
        other_obj = kwargs[self.other]
        valid = self.is_nodata(obj)
        other_valid = self.is_other_nodata(other_obj)
        if (valid ^ other_valid).any():
            raise ValidationError(f"nodata is not aligned with {self.other}")
