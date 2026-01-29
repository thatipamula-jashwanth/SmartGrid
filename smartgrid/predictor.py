import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def _predict_batch_precomputed(
    pre_votes,
    num_classes,
    xmin, inv_xstep,
    ymin, inv_ystep,
    zmin, inv_zstep,
    bins,
    XYZ,
    majority_class,
    class_weights,
):
    n = XYZ.shape[0]
    out = np.empty(n, dtype=np.int32)

    for i in prange(n):
        x = XYZ[i, 0]
        y = XYZ[i, 1]
        z = XYZ[i, 2]

        # NaN check
        if x != x or y != y or z != z:
            out[i] = majority_class
            continue

        bx = int((x - xmin) * inv_xstep)
        if bx < 0:
            bx = 0
        elif bx >= bins:
            bx = bins - 1

        by = int((y - ymin) * inv_ystep)
        if by < 0:
            by = 0
        elif by >= bins:
            by = bins - 1

        bz = int((z - zmin) * inv_zstep)
        if bz < 0:
            bz = 0
        elif bz >= bins:
            bz = bins - 1

        votes = pre_votes[bx, by, bz]

        # Empty cell → fallback
        if votes.sum() == 0.0:
            out[i] = majority_class
            continue

        max_cls = 0
        max_val = votes[0] * class_weights[0]

        for c in range(1, num_classes):
            v = votes[c] * class_weights[c]
            if v > max_val:
                max_val = v
                max_cls = c

        out[i] = max_cls

    return out


def predict_batch(ranges, XYZ):
    pre_votes = ranges["pre_votes"]
    if not pre_votes.flags["C_CONTIGUOUS"]:
        pre_votes = np.ascontiguousarray(pre_votes, dtype=np.float32)

    XYZ = np.asarray(XYZ, dtype=np.float32)

    class_weights = ranges["class_weights"]
    if not class_weights.flags["C_CONTIGUOUS"]:
        class_weights = np.ascontiguousarray(class_weights, dtype=np.float32)

    bins = ranges["bins"]
    num_classes = ranges["num_classes"]

    inv_x = 1.0 / (ranges["xstep"] + 1e-12)
    inv_y = 1.0 / (ranges["ystep"] + 1e-12)
    inv_z = 1.0 / (ranges["zstep"] + 1e-12)

    return _predict_batch_precomputed(
        pre_votes,
        num_classes,
        ranges["xmin"], inv_x,
        ranges["ymin"], inv_y,
        ranges["zmin"], inv_z,
        bins,
        XYZ,
        ranges["majority_class"],
        class_weights,
    )
