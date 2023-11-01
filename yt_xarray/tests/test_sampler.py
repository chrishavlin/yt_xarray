import numpy as np

from yt_xarray.sample_data import load_random_xr_data
from yt_xarray.transformations import Cartesian_3D_xr_Sampler, Geocentric


def test_geocentric_sampling():
    ds = load_random_xr_data(
        {
            "var1": ("r", "latitude", "longitude"),
            "var2": ("r", "latitude", "longitude"),
        },
        {"r": (10, 100, 50), "latitude": (30, 50, 30), "longitude": (200, 250, 35)},
    )

    bbox = ds.yt.get_bbox("var1")

    tform = Geocentric(bbox=bbox)

    pts = []
    for ix in range(2):
        for iy in range(2):
            for iz in range(2):
                pts.append([bbox[0, ix], bbox[1, iy], bbox[2, iz]])

    pts = np.array(pts)
    pts = [pts[:, 0], pts[:, 1], pts[:, 2]]

    cart_pts = tform.to_cartesian(pts)
    bbox_cart = np.array([[cpts.min(), cpts.max()] for cpts in cart_pts])

    sample_points = [
        np.linspace(bbox_cart[idim, 0], bbox_cart[idim, 1], 10) for idim in range(3)
    ]

    sampler = Cartesian_3D_xr_Sampler(tform, ds, ["var1", "var2"])

    _ = sampler.sample_field("var1", sample_points)
