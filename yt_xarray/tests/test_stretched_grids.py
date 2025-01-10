import numpy as np
from yt_xarray.sample_data import load_random_xr_data


def test_stretched_grid_behavior():
    def get_dim(n, offset=0.9):
        elements = np.arange(n)
        dx = 1 - offset * np.sin(elements/elements.max()*np.pi)
        x = np.cumsum(dx)
        x = x / x.max()
        return x

    dims = {'x': get_dim(50),
            'y': get_dim(50, offset=0.5),
            'z': get_dim(60, offset=0.25)}

    fields = {'temperature': ('x', 'y', 'z'), 'pressure': ('x', 'y', 'z')}
    _ = load_random_xr_data(fields, dims, length_unit='m')
