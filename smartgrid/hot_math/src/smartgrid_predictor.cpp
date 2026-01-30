#include "smartgrid_predictor.h"
#include <cmath>
#include <limits>

namespace smartgrid {

static inline int clamp_int(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

void predict_batch_precomputed(
    const float* pre_votes,
    int num_classes,
    float xmin, float inv_xstep,
    float ymin, float inv_ystep,
    float zmin, float inv_zstep,
    int bins,
    const float* XYZ,
    int N,
    int majority_class,
    const float* class_weights,
    int32_t* out_preds
) {
    const int stride_z = num_classes;
    const int stride_y = bins * stride_z;
    const int stride_x = bins * stride_y;

    for (int i = 0; i < N; ++i) {
        const float x = XYZ[i * 3 + 0];
        const float y = XYZ[i * 3 + 1];
        const float z = XYZ[i * 3 + 2];

        if (!(x == x && y == y && z == z)) {
            out_preds[i] = majority_class;
            continue;
        }

        int bx = (int)((x - xmin) * inv_xstep);
        int by = (int)((y - ymin) * inv_ystep);
        int bz = (int)((z - zmin) * inv_zstep);

        bx = clamp_int(bx, 0, bins - 1);
        by = clamp_int(by, 0, bins - 1);
        bz = clamp_int(bz, 0, bins - 1);

        const float* votes =
            pre_votes +
            bx * stride_x +
            by * stride_y +
            bz * stride_z;

        float sum = 0.0f;
        for (int c = 0; c < num_classes; ++c) {
            sum += votes[c];
        }
        if (sum == 0.0f) {
            out_preds[i] = majority_class;
            continue;
        }

        int best_cls = 0;
        float best_val = votes[0] * class_weights[0];

        for (int c = 1; c < num_classes; ++c) {
            float v = votes[c] * class_weights[c];
            if (v > best_val) {
                best_val = v;
                best_cls = c;
            }
        }

        out_preds[i] = best_cls;
    }
}

} // namespace smartgrid