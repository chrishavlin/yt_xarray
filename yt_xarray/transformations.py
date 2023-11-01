import abc
from typing import Callable, List, Optional, Tuple

import numpy as np
from aglio import coordinate_transformations as ag_ct
from numpy.typing import ArrayLike


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

    def __init__(self, ndims: int):
        self.ndims = ndims

    @abc.abstractmethod
    def sample_field(self, field_name: str, transformed_coordinates: List[ArrayLike]):
        """must return array of points at the transformed coordinates"""

    @abc.abstractmethod
    def get_native_coordinates(self, transformed_coordinates: List[ArrayLike]):
        """provide a method to go from the transformed to the native coordinates"""


class Cartesian_3D_KD_Tree(Sampler):
    def __init__(self, transformation_function: Callable):
        super().__init__(3)

    def _validate_coordiantes(self, coords: List[ArrayLike]):
        assert len(coords) == self.ndims
        for xyz in coords:
            assert xyz.shape == coords[0].shape

    def get_native_coordinates(self, transformed_coordinates: List[ArrayLike]):
        x, y, z = transformed_coordinates

    def sample_field(self, field_name: str, transformed_coordinates: List[ArrayLike]):
        self._validate_coordiantes(transformed_coordinates)

        x, y, z = transformed_coordinates
