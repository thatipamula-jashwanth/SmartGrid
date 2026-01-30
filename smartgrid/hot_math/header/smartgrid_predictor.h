#pragma once
#include <cstdint>

namespace smartgrid {

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
);

} // namespace smartgrid
