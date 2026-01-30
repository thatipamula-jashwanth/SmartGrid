import numpy as np
from ._smartgrid_cpp import predict_batch_precomputed


def predict_batch(ranges, XYZ):
    pre_votes = np.ascontiguousarray(
        ranges["pre_votes"], dtype=np.float32
    )
    XYZ = np.ascontiguousarray(XYZ, dtype=np.float32)
    class_weights = np.ascontiguousarray(
        ranges["class_weights"], dtype=np.float32
    )

    inv_x = 1.0 / (ranges["xstep"] + 1e-12)
    inv_y = 1.0 / (ranges["ystep"] + 1e-12)
    inv_z = 1.0 / (ranges["zstep"] + 1e-12)

    return predict_batch_precomputed(
        pre_votes,
        ranges["num_classes"],
        ranges["xmin"], inv_x,
        ranges["ymin"], inv_y,
        ranges["zmin"], inv_z,
        ranges["bins"],
        XYZ,
        ranges["majority_class"],
        class_weights,
    )
