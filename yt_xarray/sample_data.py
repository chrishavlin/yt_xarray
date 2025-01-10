from typing import Mapping

import numpy as np
import xarray as xr
from numpy import typing as npt
from typing import Callable

def load_random_xr_data(
    fields: Mapping[str, tuple[str, ...]],
    dims: Mapping[str, tuple[int | float, int | float, int] | npt.NDArray],
    length_unit: str | None = None,
) -> xr.Dataset:
    """

    Parameters
    ----------
    fields : dict[str, tuple]
        A dictionary specifying fields and their dimensions
        {'field1': ('x', 'y', 'z'), 'field2': ('x', 'y')}
    dims
        a dictionary mapping any dimensions to their start, stop and size.
        Any dimensions specified in fields must exist here.
        {'x': (0, 1, 10), 'y': (0, 2, 15)} would create an x and y dimension
        that goes from 0 to 1 with 10 elements for x, 0 to 2 with 15 elements
        for y.
    length_unit
        optional string to indicate length unit for dimensions.

    Returns
    -------
    xr.Dataset
        an xarray Dataset with fields of random values with the supplied names
        and dimensions.

    """

    available_coords = {}
    for dim_name, dim_range in dims.items():
        if isinstance(dim_range, np.ndarray):
            available_coords[dim_name] = dim_range
        elif isinstance(dim_range, tuple):
            available_coords[dim_name] = np.linspace(*dim_range)
        else:
            msg = f"unexpected type for `dim_range`: {dim_range} must be a mapping" \
                  f"from dimension to either a tuple for (start, stop, size) or " \
                  f"a numpy array."
            raise RuntimeError(msg)

    data = {}
    for field, field_dims in fields.items():
        coords = {}
        sz = []
        for dim_name in field_dims:
            if dim_name not in available_coords:
                raise KeyError(
                    f"{dim_name} is specified as a dimension for "
                    f"{field} but does not exist in the dims argument!"
                )
            coords[dim_name] = available_coords[dim_name]
            sz.append(coords[dim_name].size)
        data[field] = xr.DataArray(np.random.rand(*sz), coords=coords, dims=field_dims)

    attrs = {}
    if length_unit is not None:
        attrs["geospatial_vertical_units"] = length_unit
    return xr.Dataset(data, attrs=attrs)
