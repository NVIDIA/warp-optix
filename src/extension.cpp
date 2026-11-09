// Copyright (c) 2022 NVIDIA CORPORATION All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// Supplemental bindings only. Standard OptiX bindings come from pyoptix.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>
#define NOMINMAX
#include <optix.h>
static_assert(OPTIX_VERSION == 90100, "OptiX SDK version mismatch: the PyOptiX wrapper bridge requires OptiX 9.1.0");
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <algorithm>
#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;
#define IF_OPTIX72(code) code
#define IF_OPTIX74(code) code
#define COMMA ,
#define PYOPTIX_CHECK(call)                                                                                            \
    do {                                                                                                               \
        OptixResult result = (call);                                                                                   \
        if (result != OPTIX_SUCCESS)                                                                                   \
            throw std::runtime_error(optixGetErrorString(result));                                                     \
    } while (0)
#include "pyoptix_compat.h"

namespace {
py::tuple optix_version() {
    return py::make_tuple(OPTIX_VERSION / 10000, (OPTIX_VERSION % 10000) / 100, OPTIX_VERSION % 100);
}

void ensure_optix_initialized() {
    static std::once_flag initialized;
    std::call_once(initialized, [] { PYOPTIX_CHECK(optixInit()); });
}

size_t checked_add(size_t a, size_t b, const char* what) {
    if (a > SIZE_MAX - b)
        throw py::value_error(std::string(what) + " overflows the address space");
    return a + b;
}

size_t checked_mul(size_t a, size_t b, const char* what) {
    if (a != 0 && b > SIZE_MAX / a)
        throw py::value_error(std::string(what) + " overflows the address space");
    return a * b;
}

// Bytes touched from `base` by `numNetworks` networks of `layers`, spaced `stride` apart.
size_t network_extent(const std::vector<OptixCoopVecMatrixDescription>& layers, size_t stride, unsigned int numNetworks,
                      const char* what) {
    size_t layerEnd = 0;
    for (const OptixCoopVecMatrixDescription& layer : layers)
        layerEnd = std::max(layerEnd, checked_add(layer.offsetInBytes, layer.sizeInBytes, what));
    return checked_add(checked_mul(numNetworks - 1, stride, what), layerEnd, what);
}

// Raw device pointers arrive from Python as integers, so verify that they lie in a
// live CUDA allocation that covers `requiredBytes` before OptiX reads or writes them.
// The driver is resolved at runtime, like OptiX itself, so that building needs only
// cuda.h and the wheel links no CUDA library. Warp has already loaded the driver.
using PointerGetAttributeFn = CUresult(CUDAAPI*)(void*, CUpointer_attribute, CUdeviceptr);

PointerGetAttributeFn pointer_get_attribute() {
    static const PointerGetAttributeFn function = [] {
#ifdef _WIN32
        HMODULE driver = LoadLibraryExA("nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        return driver ? reinterpret_cast<PointerGetAttributeFn>(GetProcAddress(driver, "cuPointerGetAttribute"))
                      : nullptr;
#else
        void* driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        return driver ? reinterpret_cast<PointerGetAttributeFn>(dlsym(driver, "cuPointerGetAttribute")) : nullptr;
#endif
    }();
    if (!function)
        throw std::runtime_error("Cannot load cuPointerGetAttribute from the NVIDIA driver library");
    return function;
}

void require_device_range(uintptr_t ptr, size_t requiredBytes, const char* name) {
    CUdeviceptr rangeStart = 0;
    size_t rangeSize = 0;
    const CUdeviceptr devicePtr = static_cast<CUdeviceptr>(ptr);
    const PointerGetAttributeFn getAttribute = pointer_get_attribute();
    if (getAttribute(&rangeStart, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, devicePtr) != CUDA_SUCCESS ||
        getAttribute(&rangeSize, CU_POINTER_ATTRIBUTE_RANGE_SIZE, devicePtr) != CUDA_SUCCESS)
        throw py::value_error(std::string(name) + " is not a CUDA device allocation");
    const size_t available = static_cast<size_t>(rangeStart + rangeSize - devicePtr);
    if (requiredBytes > available)
        throw py::value_error(std::string(name) + " requires " + std::to_string(requiredBytes) +
                              " bytes but its allocation has " + std::to_string(available) +
                              " bytes after the pointer");
}
} // namespace

namespace warp_optix_ext {

size_t coopVecMatrixComputeSize(pyoptix::DeviceContext context, unsigned int N, unsigned int K,
                                OptixCoopVecElemType elementType, OptixCoopVecMatrixLayout layout,
                                size_t rowColumnStrideInBytes) {
    ensure_optix_initialized();
    if (N == 0 || K == 0)
        throw py::value_error("Cooperative-vector matrix dimensions must be non-zero");

    size_t sizeInBytes = 0;
    PYOPTIX_CHECK(optixCoopVecMatrixComputeSize(context.deviceContext, N, K, elementType, layout,
                                                rowColumnStrideInBytes, &sizeInBytes));
    return sizeInBytes;
}

void coopVecMatrixConvert(pyoptix::DeviceContext context, uintptr_t stream,
                          const std::vector<OptixCoopVecMatrixDescription>& inputLayers, uintptr_t inputNetworks,
                          const std::vector<OptixCoopVecMatrixDescription>& outputLayers, uintptr_t outputNetworks,
                          size_t inputNetworkStrideInBytes, size_t outputNetworkStrideInBytes,
                          unsigned int numNetworks) {
    ensure_optix_initialized();
    if (numNetworks == 0)
        throw py::value_error("numNetworks must be non-zero");
    if (inputLayers.empty())
        throw py::value_error("inputLayers must contain at least one matrix description");
    if (inputLayers.size() != outputLayers.size())
        throw py::value_error("inputLayers and outputLayers must have the same length");
    if (inputNetworks == 0 || outputNetworks == 0)
        throw py::value_error("inputNetworks and outputNetworks must be non-zero CUDA device pointers");
    if (numNetworks > 1 && (inputNetworkStrideInBytes == 0 || outputNetworkStrideInBytes == 0))
        throw py::value_error("network strides must be non-zero when numNetworks is greater than one");
    if (inputNetworkStrideInBytes % 64 != 0 || outputNetworkStrideInBytes % 64 != 0)
        throw py::value_error("cooperative-vector network strides must be multiples of 64 bytes");
    if (inputLayers.size() > UINT_MAX)
        throw py::value_error("too many cooperative-vector matrix descriptions");

    for (size_t i = 0; i < inputLayers.size(); ++i) {
        const OptixCoopVecMatrixDescription& input = inputLayers[i];
        const OptixCoopVecMatrixDescription& output = outputLayers[i];
        if (input.N == 0 || input.K == 0 || output.N == 0 || output.K == 0)
            throw py::value_error("cooperative-vector matrix dimensions must be non-zero");
        if (input.N != output.N || input.K != output.K)
            throw py::value_error("input and output matrix dimensions must match");
        if (input.sizeInBytes == 0 || output.sizeInBytes == 0)
            throw py::value_error("cooperative-vector matrix sizes must be non-zero");
        if ((inputNetworks % 64 + input.offsetInBytes % 64) % 64 != 0)
            throw py::value_error("each input matrix address must be 64-byte aligned");
        if ((outputNetworks % 64 + output.offsetInBytes % 64) % 64 != 0)
            throw py::value_error("each output matrix address must be 64-byte aligned");
    }

    require_device_range(inputNetworks,
                         network_extent(inputLayers, inputNetworkStrideInBytes, numNetworks, "input network extent"),
                         "inputNetworks");
    require_device_range(outputNetworks,
                         network_extent(outputLayers, outputNetworkStrideInBytes, numNetworks, "output network extent"),
                         "outputNetworks");

    OptixNetworkDescription inputDescription{const_cast<OptixCoopVecMatrixDescription*>(inputLayers.data()),
                                             static_cast<unsigned int>(inputLayers.size())};
    OptixNetworkDescription outputDescription{const_cast<OptixCoopVecMatrixDescription*>(outputLayers.data()),
                                              static_cast<unsigned int>(outputLayers.size())};

    OptixResult result;
    {
        py::gil_scoped_release release;
        result = optixCoopVecMatrixConvert(context.deviceContext, reinterpret_cast<CUstream>(stream), numNetworks,
                                           &inputDescription, static_cast<CUdeviceptr>(inputNetworks),
                                           inputNetworkStrideInBytes, &outputDescription,
                                           static_cast<CUdeviceptr>(outputNetworks), outputNetworkStrideInBytes);
    }
    if (result != OPTIX_SUCCESS)
        throw std::runtime_error(optixGetErrorString(result));
}

} // namespace warp_optix_ext

namespace {

template <typename T> void require_upstream_type(py::handle cls) {
    const auto* registered = py::detail::get_type_info(typeid(T), false);
    const auto* actual = py::detail::get_type_info(reinterpret_cast<PyTypeObject*>(cls.ptr()));
    if (!registered || registered != actual || registered->type_size != sizeof(T)) {
        throw py::import_error("warp-optix requires ABI-compatible pyoptix 9.1.1 bindings. "
                               "Rebuild pyoptix and warp-optix using the same pybind11 major version and C++ ABI.");
    }
}

template <typename T, typename Field> void add_field(py::handle cls, const char* name, Field T::* member) {
    if (py::hasattr(cls, name))
        return;
    cls.attr(name) =
        py::module_::import("builtins")
            .attr("property")(
                py::cpp_function([member](const T& self) { return self.*member; }, py::is_method(cls)),
                py::cpp_function([member](T& self, const Field& value) { self.*member = value; }, py::is_method(cls)));
}

void sync_module_options(pyoptix::ModuleCompileOptions& options) {
    // Upstream 9.1.1's sync() returns before forwarding this metadata.
    options.boundValues.clear();
    options.boundValues.reserve(options.pyboundValues.size());
    for (auto& value : options.pyboundValues) {
        value.sync();
        options.boundValues.push_back(value.entry);
    }
    options.options.boundValues = options.boundValues.empty() ? nullptr : options.boundValues.data();
    options.options.numBoundValues = static_cast<unsigned int>(options.boundValues.size());
    options.payloadTypes.clear();
    options.payloadTypes.reserve(options.pypayloadTypes.size());
    for (auto& payload : options.pypayloadTypes) {
        payload.sync();
        options.payloadTypes.push_back(payload.payload_type);
    }
    options.options.payloadTypes = options.payloadTypes.empty() ? nullptr : options.payloadTypes.data();
    options.options.numPayloadTypes = static_cast<unsigned int>(options.payloadTypes.size());
}

void install_pyoptix_extensions(py::module_ upstream) {
    const py::tuple upstream_version = upstream.attr("version")();
    if (!upstream_version.equal(optix_version())) {
        throw py::import_error("OptiX SDK version mismatch: pyoptix was compiled with " +
                               py::str(upstream_version).cast<std::string>() + ", but warp-optix was compiled with " +
                               py::str(optix_version()).cast<std::string>() +
                               ". Rebuild both packages with matching OptiX headers.");
    }
    // Validate all shared layouts before installing any descriptors or reading
    // an upstream object. PyOptiX remains the sole owner of these native types.
    require_upstream_type<pyoptix::DeviceContext>(upstream.attr("DeviceContext"));
    require_upstream_type<pyoptix::ModuleCompileBoundValueEntry>(upstream.attr("ModuleCompileBoundValueEntry"));
    require_upstream_type<pyoptix::PayloadType>(upstream.attr("PayloadType"));
    require_upstream_type<pyoptix::ModuleCompileOptions>(upstream.attr("ModuleCompileOptions"));
    require_upstream_type<OptixDenoiserGuideLayer>(upstream.attr("DenoiserGuideLayer"));
    require_upstream_type<OptixDenoiserParams>(upstream.attr("DenoiserParams"));
    require_upstream_type<OptixDenoiserSizes>(upstream.attr("DenoiserSizes"));

    auto context = upstream.attr("DeviceContext");
    // Supplemental bindings only add missing API; never replace PyOptiX methods.
    for (const char* name : {"coopVecMatrixComputeSize", "coopVecMatrixConvert"}) {
        if (py::hasattr(context, name) && !py::hasattr(upstream, "_warp_optix_native_extensions_installed"))
            throw py::import_error(std::string("pyoptix already defines DeviceContext.") + name +
                                   "; refusing to replace it");
    }
    context.attr("coopVecMatrixComputeSize") =
        py::cpp_function(&warp_optix_ext::coopVecMatrixComputeSize, py::is_method(context), py::arg("N"), py::arg("K"),
                         py::arg("elementType"), py::arg("layout"), py::arg("rowColumnStrideInBytes") = 0u);
    context.attr("coopVecMatrixConvert") =
        py::cpp_function(&warp_optix_ext::coopVecMatrixConvert, py::is_method(context), py::arg("stream"),
                         py::arg("inputLayers"), py::arg("inputNetworks"), py::arg("outputLayers"),
                         py::arg("outputNetworks"), py::arg("inputNetworkStrideInBytes") = 0u,
                         py::arg("outputNetworkStrideInBytes") = 0u, py::arg("numNetworks") = 1u);

    auto guides = upstream.attr("DenoiserGuideLayer");
    add_field(guides, "previousOutputInternalGuideLayer", &OptixDenoiserGuideLayer::previousOutputInternalGuideLayer);
    add_field(guides, "outputInternalGuideLayer", &OptixDenoiserGuideLayer::outputInternalGuideLayer);
    add_field(upstream.attr("DenoiserParams"), "temporalModeUsePreviousLayers",
              &OptixDenoiserParams::temporalModeUsePreviousLayers);
    auto sizes = upstream.attr("DenoiserSizes");
    add_field(sizes, "computeIntensitySizeInBytes", &OptixDenoiserSizes::computeIntensitySizeInBytes);
    add_field(sizes, "computeAverageColorSizeInBytes", &OptixDenoiserSizes::computeAverageColorSizeInBytes);
    add_field(sizes, "internalGuideLayerPixelSizeInBytes", &OptixDenoiserSizes::internalGuideLayerPixelSizeInBytes);
    upstream.attr("_warp_optix_native_extensions_installed") = true;
}

} // namespace

PYBIND11_MODULE(_ext, m) {
    m.doc() = "Additional OptiX bindings for the pyoptix dependency";
    m.def("optix_version", &optix_version, "OptiX SDK version used to compile this supplemental extension.");
    m.def("install_pyoptix_extensions", &install_pyoptix_extensions);
    m.def("sync_module_options", &sync_module_options);
    m.def("coop_vec_property", [](const pyoptix::DeviceContext& context) {
        ensure_optix_initialized();
        uint32_t flags = 0;
        PYOPTIX_CHECK(optixDeviceContextGetProperty(context.deviceContext, OPTIX_DEVICE_PROPERTY_COOP_VEC, &flags,
                                                    sizeof(flags)));
        return flags;
    });
    // Our private OptiX dispatch table is initialized lazily: importing the
    // package must also work on machines without an NVIDIA driver.
    py::enum_<OptixDevicePropertyCoopVecFlags>(m, "DevicePropertyCoopVecFlags", py::arithmetic())
        .value("DEVICE_PROPERTY_COOP_VEC_FLAG_NONE", OPTIX_DEVICE_PROPERTY_COOP_VEC_FLAG_NONE)
        .value("DEVICE_PROPERTY_COOP_VEC_FLAG_STANDARD", OPTIX_DEVICE_PROPERTY_COOP_VEC_FLAG_STANDARD)
        .export_values();

    py::enum_<OptixCoopVecElemType>(m, "CoopVecElemType", py::arithmetic())
        .value("COOP_VEC_ELEM_TYPE_UNKNOWN", OPTIX_COOP_VEC_ELEM_TYPE_UNKNOWN)
        .value("COOP_VEC_ELEM_TYPE_FLOAT16", OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16)
        .value("COOP_VEC_ELEM_TYPE_FLOAT32", OPTIX_COOP_VEC_ELEM_TYPE_FLOAT32)
        .value("COOP_VEC_ELEM_TYPE_UINT8", OPTIX_COOP_VEC_ELEM_TYPE_UINT8)
        .value("COOP_VEC_ELEM_TYPE_INT8", OPTIX_COOP_VEC_ELEM_TYPE_INT8)
        .value("COOP_VEC_ELEM_TYPE_UINT32", OPTIX_COOP_VEC_ELEM_TYPE_UINT32)
        .value("COOP_VEC_ELEM_TYPE_INT32", OPTIX_COOP_VEC_ELEM_TYPE_INT32)
        .value("COOP_VEC_ELEM_TYPE_FLOAT8_E4M3", OPTIX_COOP_VEC_ELEM_TYPE_FLOAT8_E4M3)
        .value("COOP_VEC_ELEM_TYPE_FLOAT8_E5M2", OPTIX_COOP_VEC_ELEM_TYPE_FLOAT8_E5M2)
        .export_values();

    py::enum_<OptixCoopVecMatrixLayout>(m, "CoopVecMatrixLayout", py::arithmetic())
        .value("COOP_VEC_MATRIX_LAYOUT_ROW_MAJOR", OPTIX_COOP_VEC_MATRIX_LAYOUT_ROW_MAJOR)
        .value("COOP_VEC_MATRIX_LAYOUT_COLUMN_MAJOR", OPTIX_COOP_VEC_MATRIX_LAYOUT_COLUMN_MAJOR)
        .value("COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL", OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL)
        .value("COOP_VEC_MATRIX_LAYOUT_TRAINING_OPTIMAL", OPTIX_COOP_VEC_MATRIX_LAYOUT_TRAINING_OPTIMAL)
        .export_values();

    py::class_<OptixCoopVecMatrixDescription>(m, "CoopVecMatrixDescription")
        .def(py::init(
            []() { return std::unique_ptr<OptixCoopVecMatrixDescription>(new OptixCoopVecMatrixDescription{}); }))
        .def_readwrite("N", &OptixCoopVecMatrixDescription::N)
        .def_readwrite("K", &OptixCoopVecMatrixDescription::K)
        .def_readwrite("offsetInBytes", &OptixCoopVecMatrixDescription::offsetInBytes)
        .def_readwrite("elementType", &OptixCoopVecMatrixDescription::elementType)
        .def_readwrite("layout", &OptixCoopVecMatrixDescription::layout)
        .def_readwrite("rowColumnStrideInBytes", &OptixCoopVecMatrixDescription::rowColumnStrideInBytes)
        .def_readwrite("sizeInBytes", &OptixCoopVecMatrixDescription::sizeInBytes);

    m.attr("DEVICE_PROPERTY_COOP_VEC") = py::int_(static_cast<unsigned int>(OPTIX_DEVICE_PROPERTY_COOP_VEC));
    m.attr("PRIMITIVE_TYPE_ROUND_CATMULLROM") =
        py::int_(static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM));
    m.attr("PRIMITIVE_TYPE_FLAT_QUADRATIC_BSPLINE") =
        py::int_(static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_FLAT_QUADRATIC_BSPLINE));
    m.attr("PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER") =
        py::int_(static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER));
    m.attr("PRIMITIVE_TYPE_FLAGS_ROUND_CATMULLROM") =
        py::int_(static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CATMULLROM));
    m.attr("PRIMITIVE_TYPE_FLAGS_FLAT_QUADRATIC_BSPLINE") =
        py::int_(static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_FLAGS_FLAT_QUADRATIC_BSPLINE));
    m.attr("PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BEZIER") =
        py::int_(static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BEZIER));
    m.attr("PIXEL_FORMAT_FLOAT2") = py::int_(static_cast<unsigned int>(OPTIX_PIXEL_FORMAT_FLOAT2));
    m.attr("PIXEL_FORMAT_INTERNAL_GUIDE_LAYER") =
        py::int_(static_cast<unsigned int>(OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER));
    m.attr("DENOISER_MODEL_KIND_TEMPORAL") = py::int_(static_cast<unsigned int>(OPTIX_DENOISER_MODEL_KIND_TEMPORAL));
    m.attr("DENOISER_MODEL_KIND_TEMPORAL_AOV") =
        py::int_(static_cast<unsigned int>(OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV));
    m.attr("DENOISER_MODEL_KIND_UPSCALE2X") = py::int_(static_cast<unsigned int>(OPTIX_DENOISER_MODEL_KIND_UPSCALE2X));
    m.attr("DENOISER_MODEL_KIND_TEMPORAL_UPSCALE2X") =
        py::int_(static_cast<unsigned int>(OPTIX_DENOISER_MODEL_KIND_TEMPORAL_UPSCALE2X));
}
