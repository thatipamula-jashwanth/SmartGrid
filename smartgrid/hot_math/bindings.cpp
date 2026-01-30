#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "smartgrid_predictor.h"

namespace py = pybind11;
using namespace smartgrid;

PYBIND11_MODULE(_smartgrid_cpp, m) {
    m.def(
        "predict_batch_precomputed",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> pre_votes,
           int num_classes,
           float xmin, float inv_xstep,
           float ymin, float inv_ystep,
           float zmin, float inv_zstep,
           int bins,
           py::array_t<float, py::array::c_style | py::array::forcecast> XYZ,
           int majority_class,
           py::array_t<float, py::array::c_style | py::array::forcecast> class_weights) {

            const int N = XYZ.shape(0);
            auto out = py::array_t<int32_t>(N);

            predict_batch_precomputed(
                pre_votes.data(),
                num_classes,
                xmin, inv_xstep,
                ymin, inv_ystep,
                zmin, inv_zstep,
                bins,
                XYZ.data(),
                N,
                majority_class,
                class_weights.data(),
                out.mutable_data()
            );

            return out;
        },
        py::arg("pre_votes"),
        py::arg("num_classes"),
        py::arg("xmin"), py::arg("inv_xstep"),
        py::arg("ymin"), py::arg("inv_ystep"),
        py::arg("zmin"), py::arg("inv_zstep"),
        py::arg("bins"),
        py::arg("XYZ"),
        py::arg("majority_class"),
        py::arg("class_weights")
    );
}
