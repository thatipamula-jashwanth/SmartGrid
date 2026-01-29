import numpy as np
from numba import njit, prange

@njit(parallel=True)
def _box_filter_x_inplace(src, dst, r):
    bins, _, _, C = src.shape

    for y in prange(bins):
        for z in range(bins):
            for c in range(C):
                s = 0.0
                for x in range(bins):
                    if x == 0:
                        x0 = 0
                        x1 = min(bins, r + 1)
                        for xx in range(x0, x1):
                            s += src[xx, y, z, c]
                    else:
                        if x - r - 1 >= 0:
                            s -= src[x - r - 1, y, z, c]
                        if x + r < bins:
                            s += src[x + r, y, z, c]
                    dst[x, y, z, c] = s


@njit(parallel=True)
def _box_filter_y_inplace(src, dst, r):
    bins, _, _, C = src.shape

    for x in prange(bins):
        for z in range(bins):
            for c in range(C):
                s = 0.0
                for y in range(bins):
                    if y == 0:
                        y0 = 0
                        y1 = min(bins, r + 1)
                        for yy in range(y0, y1):
                            s += src[x, yy, z, c]
                    else:
                        if y - r - 1 >= 0:
                            s -= src[x, y - r - 1, z, c]
                        if y + r < bins:
                            s += src[x, y + r, z, c]
                    dst[x, y, z, c] = s


@njit(parallel=True)
def _box_filter_z_inplace(src, dst, r):
    bins, _, _, C = src.shape

    for x in prange(bins):
        for y in range(bins):
            for c in range(C):
                s = 0.0
                for z in range(bins):
                    if z == 0:
                        z0 = 0
                        z1 = min(bins, r + 1)
                        for zz in range(z0, z1):
                            s += src[x, y, zz, c]
                    else:
                        if z - r - 1 >= 0:
                            s -= src[x, y, z - r - 1, c]
                        if z + r < bins:
                            s += src[x, y, z + r, c]
                    dst[x, y, z, c] = s


def _compute_pre_votes(grid, r):

    buf1 = np.zeros_like(grid)
    buf2 = np.zeros_like(grid)

    _box_filter_x_inplace(grid, buf1, r)
    _box_filter_y_inplace(buf1, buf2, r)
    _box_filter_z_inplace(buf2, buf1, r)

    return buf1


def build_grid(XYZ, y, bins=20, r=1):
    XYZ = np.asarray(XYZ, dtype=np.float32)
    y = np.asarray(y)

    if XYZ.ndim != 2 or XYZ.shape[1] != 3:
        raise ValueError("XYZ must be (n_samples, 3)")
    if XYZ.shape[0] != y.shape[0]:
        raise ValueError("XYZ and y length mismatch")
    if not (1 <= bins <= 120):
        raise ValueError("bins must be in [1,120]")
    if not (0 <= r <= 4):
        raise ValueError("r must be in [0,4]")

    XYZ = np.nan_to_num(XYZ, nan=0.0)

    classes, y_enc = np.unique(y, return_inverse=True)
    C = len(classes)

    needed_bytes = bins**3 * C * 4
    if needed_bytes > 1_200_000_000:
        raise MemoryError("Grid too large")

    X, Y, Z = XYZ.T
    xmin, xmax = X.min(), X.max()
    ymin, ymax = Y.min(), Y.max()
    zmin, zmax = Z.min(), Z.max()

    eps = 1e-9
    xstep = (xmax - xmin) / bins + eps
    ystep = (ymax - ymin) / bins + eps
    zstep = (zmax - zmin) / bins + eps

    xi = np.clip(((X - xmin) / xstep).astype(np.int32), 0, bins - 1)
    yi = np.clip(((Y - ymin) / ystep).astype(np.int32), 0, bins - 1)
    zi = np.clip(((Z - zmin) / zstep).astype(np.int32), 0, bins - 1)

    flat_idx = (
        xi * (bins * bins * C) +
        yi * (bins * C) +
        zi * C +
        y_enc
    )

    grid = np.zeros(bins * bins * bins * C, dtype=np.float32)
    counts = np.bincount(flat_idx, minlength=grid.size)
    grid[:] = counts
    grid = grid.reshape((bins, bins, bins, C))

    pre_votes = _compute_pre_votes(grid, r)

    return grid, {
        "xmin": xmin, "xstep": xstep,
        "ymin": ymin, "ystep": ystep,
        "zmin": zmin, "zstep": zstep,
        "bins": bins,
        "classes": classes,
        "num_classes": C,
        "pre_votes": np.ascontiguousarray(pre_votes),
        "r": r,
    }
