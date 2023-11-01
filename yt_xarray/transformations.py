import abc
from typing import List, Optional, Tuple

import numpy as np
import xarray as xr
from aglio import coordinate_transformations as ag_ct
from numpy.typing import ArrayLike
from yt.utilities.linear_interpolators import TrilinearFieldInterpolator


class Transformation(abc.ABC):
    ndim = 3
    coord_names: Tuple[str, str, str]
    _default_bbox: list

    def __init__(self, bbox: Optional[ArrayLike]):
        if bbox is None:
            bbox = self._default_bbox

        self.bbox = np.asarray(bbox)
        if self.bbox.shape != (self.ndim, 2):
            raise ValueError(
                f"bounding_box must have shape {(self.ndim, 2)}, found {self.bbox.shape}"
            )

        self.name_to_axis_id = {
            name: axis_id for axis_id, name in enumerate(self.coord_names)
        }
        self.axis_id_to_name = {
            axis_id: name for name, axis_id in self.name_to_axis_id.items()
        }

    @abc.abstractmethod
    def to_cartesian(self, coords: List[ArrayLike]):
        """must implement method to go from native to cartesian coordinates"""

    @abc.abstractmethod
    def to_native(self, x, y, z):
        """must implement method to go from cartesian to native coordinates"""

    def _linear_normalization(self, vals: ArrayLike, min_max_vals: ArrayLike):
        min_val = min_max_vals[0]
        max_val = min_max_vals[1]
        return (vals - min_val) / (max_val - min_val)

    def to_normalized_native(self, coords: List[ArrayLike]):
        coords_native = self.to_native(coords)
        return self.normalize_native(coords_native)

    def normalize_native(self, coords_native: List[ArrayLike]):
        for idim in range(self.ndim):
            coords_native[idim] = self._linear_normalization(
                coords_native[idim], self.bbox[idim]
            )
        return coords_native


class Spherical(Transformation):
    coord_names = ("r", "theta", "phi")
    _default_bbox = [[0.0, 1.0], [0.0, 2.0 * np.pi], [0, np.pi]]

    def to_cartesian(self, coords: List[ArrayLike]):
        r = coords[self.name_to_axis_id["r"]]
        theta = coords[self.name_to_axis_id["theta"]]
        phi = coords[self.name_to_axis_id["phi"]]
        return ag_ct.sphere2cart(phi, theta, r)

    def to_native(self, x, y, z):
        return ag_ct.cart2sphere(x, y, z, deg=False)


class Geocentric(Transformation):
    coord_names = ("r", "longitude", "latitude")
    _default_bbox = [[0.0, 6371000.0], [0.0, 360.0], [-90.0, 90.0]]

    def to_cartesian(self, coords: List[ArrayLike]):
        r = coords[self.name_to_axis_id["r"]]
        lat = coords[self.name_to_axis_id["latitude"]]
        lon = coords[self.name_to_axis_id["longitude"]]
        return ag_ct.geosphere2cart(lat, lon, r)

    def to_native(self, x, y, z):
        return ag_ct.cart2sphere(x, y, z, deg=True, geo=True)


class Sampler(abc.ABC):
    """
    An abstract base class from which all Samplers inherit, not meant
    to be instantiated directly
    """

    ndims: int

    def __init__(self, transformation: Transformation):
        self.transformation = transformation

    def _validate_coordiantes(self, coords: List[ArrayLike]):
        assert len(coords) == self.ndims
        for coord in coords:
            assert coord.shape == coords[0].shape

    @abc.abstractmethod
    def sample_field(self, field_name: str, coords: List[ArrayLike]):
        """must return array of points at the transformed coordinates"""


class Cartesian_3D_xr_Sampler(Sampler):
    ndims = 3

    def __init__(
        self, transformation: Transformation, ds_xr: xr.Dataset, field_subset: List[str]
    ):
        super().__init__(transformation)

        dims = None
        for field in field_subset:
            var = ds_xr.data_vars[field]
            if dims is None:
                dims = var.dims
            assert var.dims == dims

        native_coords = [ds_xr.coords[dim].to_numpy() for dim in dims]
        table = ds_xr.data_vars[field].to_numpy()
        field_names = dims
        self.dims = dims
        self.ds_xr = ds_xr
        self.field_subset = field_subset
        self.current_field = field_subset[0]
        self.interpolator = TrilinearFieldInterpolator(
            table, native_coords, field_names, truncate=True
        )

    def sample_field(self, field, coords: List[ArrayLike]):
        x, y, z = coords
        native_coords = self.transformation.to_native(x, y, z)
        data_obj_dummy = {}
        for idim in range(self.ndims):
            data_obj_dummy[self.dims[idim]] = native_coords[idim]

        if field not in self.field_subset:
            raise RuntimeError("bad")

        if field != self.current_field:
            self.current_field = field
            self.interpolator.table = (
                self.ds_xr.data_vars[field].to_numpy().astype("float64")
            )

        return self.interpolator(data_obj_dummy)
