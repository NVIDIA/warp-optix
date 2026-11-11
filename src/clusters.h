// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Included by extension.cpp: reuse its checked PyOptiX context bridge and OptiX table.
// These are supplemental types, never a second registration of upstream types.
inline void bind_clusters(py::module_& m) {
    m.def("setAllowClusteredGeometry", [](pyoptix::PipelineCompileOptions& options, bool enabled) {
        options.options.allowClusteredGeometry = enabled;
    });
    m.def("getClusterProperties", [](const pyoptix::DeviceContext& context) {
        ensure_optix_initialized();
        py::dict result;
        for (auto [name, property] :
             {std::pair{"flags", OPTIX_DEVICE_PROPERTY_CLUSTER_ACCEL},
              {"max_vertices", OPTIX_DEVICE_PROPERTY_LIMIT_MAX_CLUSTER_VERTICES},
              {"max_triangles", OPTIX_DEVICE_PROPERTY_LIMIT_MAX_CLUSTER_TRIANGLES},
              {"max_grid_resolution", OPTIX_DEVICE_PROPERTY_LIMIT_MAX_STRUCTURED_GRID_RESOLUTION},
              {"max_sbt_index", OPTIX_DEVICE_PROPERTY_LIMIT_MAX_CLUSTER_SBT_INDEX},
              {"max_clusters_per_gas", OPTIX_DEVICE_PROPERTY_LIMIT_MAX_CLUSTERS_PER_GAS}}) {
            unsigned int value = 0;
            PYOPTIX_CHECK(optixDeviceContextGetProperty(context.deviceContext, property, &value, sizeof(value)));
            result[name] = value;
            if (property == OPTIX_DEVICE_PROPERTY_CLUSTER_ACCEL && value == 0)
                break;
        }
        return result;
    });

    // clang-format off
#define CLUSTER_FIELD(type, field) .def_readwrite(#field, &type::field)
    py::class_<OptixClusterAccelBuildInputTriangles>(m, "ClusterAccelBuildInputTriangles")
        .def(py::init<>())
        CLUSTER_FIELD(OptixClusterAccelBuildInputTriangles, flags)
        CLUSTER_FIELD(OptixClusterAccelBuildInputTriangles, maxArgCount)
        CLUSTER_FIELD(OptixClusterAccelBuildInputTriangles, vertexFormat)
        CLUSTER_FIELD(OptixClusterAccelBuildInputTriangles, maxSbtIndexValue)
        CLUSTER_FIELD(OptixClusterAccelBuildInputTriangles, maxUniqueSbtIndexCountPerArg)
        CLUSTER_FIELD(OptixClusterAccelBuildInputTriangles, maxTriangleCountPerArg)
        CLUSTER_FIELD(OptixClusterAccelBuildInputTriangles, maxVertexCountPerArg)
        CLUSTER_FIELD(OptixClusterAccelBuildInputTriangles, maxTotalTriangleCount)
        CLUSTER_FIELD(OptixClusterAccelBuildInputTriangles, maxTotalVertexCount)
        CLUSTER_FIELD(OptixClusterAccelBuildInputTriangles, minPositionTruncateBitCount);
    py::class_<OptixClusterAccelBuildInputGrids>(m, "ClusterAccelBuildInputGrids")
        .def(py::init<>())
        CLUSTER_FIELD(OptixClusterAccelBuildInputGrids, flags)
        CLUSTER_FIELD(OptixClusterAccelBuildInputGrids, maxArgCount)
        CLUSTER_FIELD(OptixClusterAccelBuildInputGrids, vertexFormat)
        CLUSTER_FIELD(OptixClusterAccelBuildInputGrids, maxSbtIndexValue)
        CLUSTER_FIELD(OptixClusterAccelBuildInputGrids, maxWidth)
        CLUSTER_FIELD(OptixClusterAccelBuildInputGrids, maxHeight);
    py::class_<OptixClusterAccelBuildInputClusters>(m, "ClusterAccelBuildInputClusters")
        .def(py::init<>())
        CLUSTER_FIELD(OptixClusterAccelBuildInputClusters, flags)
        CLUSTER_FIELD(OptixClusterAccelBuildInputClusters, maxArgCount)
        CLUSTER_FIELD(OptixClusterAccelBuildInputClusters, maxTotalClusterCount)
        CLUSTER_FIELD(OptixClusterAccelBuildInputClusters, maxClusterCountPerArg);
    py::class_<OptixClusterAccelBuildInput>(m, "ClusterAccelBuildInput")
        .def(py::init<>())
        CLUSTER_FIELD(OptixClusterAccelBuildInput, type)
        CLUSTER_FIELD(OptixClusterAccelBuildInput, triangles)
        CLUSTER_FIELD(OptixClusterAccelBuildInput, grids)
        CLUSTER_FIELD(OptixClusterAccelBuildInput, clusters);

    py::class_<OptixClusterAccelBuildModeDescImplicitDest>(m, "ClusterAccelBuildModeDescImplicitDest")
        .def(py::init<>())
        CLUSTER_FIELD(OptixClusterAccelBuildModeDescImplicitDest, outputBuffer)
        CLUSTER_FIELD(OptixClusterAccelBuildModeDescImplicitDest, outputBufferSizeInBytes)
        CLUSTER_FIELD(OptixClusterAccelBuildModeDescImplicitDest, tempBuffer)
        CLUSTER_FIELD(OptixClusterAccelBuildModeDescImplicitDest, tempBufferSizeInBytes)
        CLUSTER_FIELD(OptixClusterAccelBuildModeDescImplicitDest, outputHandlesBuffer)
        CLUSTER_FIELD(OptixClusterAccelBuildModeDescImplicitDest, outputHandlesStrideInBytes)
        CLUSTER_FIELD(OptixClusterAccelBuildModeDescImplicitDest, outputSizesBuffer)
        CLUSTER_FIELD(OptixClusterAccelBuildModeDescImplicitDest, outputSizesStrideInBytes);
    py::class_<OptixClusterAccelBuildModeDescExplicitDest>(m, "ClusterAccelBuildModeDescExplicitDest")
        .def(py::init<>())
        CLUSTER_FIELD(OptixClusterAccelBuildModeDescExplicitDest, tempBuffer)
        CLUSTER_FIELD(OptixClusterAccelBuildModeDescExplicitDest, tempBufferSizeInBytes)
        CLUSTER_FIELD(OptixClusterAccelBuildModeDescExplicitDest, destAddressesBuffer)
        CLUSTER_FIELD(OptixClusterAccelBuildModeDescExplicitDest, destAddressesStrideInBytes)
        CLUSTER_FIELD(OptixClusterAccelBuildModeDescExplicitDest, outputHandlesBuffer)
        CLUSTER_FIELD(OptixClusterAccelBuildModeDescExplicitDest, outputHandlesStrideInBytes)
        CLUSTER_FIELD(OptixClusterAccelBuildModeDescExplicitDest, outputSizesBuffer)
        CLUSTER_FIELD(OptixClusterAccelBuildModeDescExplicitDest, outputSizesStrideInBytes);
    py::class_<OptixClusterAccelBuildModeDescGetSize>(m, "ClusterAccelBuildModeDescGetSize")
        .def(py::init<>())
        CLUSTER_FIELD(OptixClusterAccelBuildModeDescGetSize, outputSizesBuffer)
        CLUSTER_FIELD(OptixClusterAccelBuildModeDescGetSize, outputSizesStrideInBytes)
        CLUSTER_FIELD(OptixClusterAccelBuildModeDescGetSize, tempBuffer)
        CLUSTER_FIELD(OptixClusterAccelBuildModeDescGetSize, tempBufferSizeInBytes);
    py::class_<OptixClusterAccelBuildModeDesc>(m, "ClusterAccelBuildModeDesc")
        .def(py::init<>())
        CLUSTER_FIELD(OptixClusterAccelBuildModeDesc, mode)
        CLUSTER_FIELD(OptixClusterAccelBuildModeDesc, implicitDest)
        CLUSTER_FIELD(OptixClusterAccelBuildModeDesc, explicitDest)
        CLUSTER_FIELD(OptixClusterAccelBuildModeDesc, getSize);
#undef CLUSTER_FIELD
    // clang-format on

    py::enum_<OptixClusterAccelBuildType>(m, "ClusterAccelBuildType")
        .value("CLUSTER_ACCEL_BUILD_TYPE_GASES_FROM_CLUSTERS", OPTIX_CLUSTER_ACCEL_BUILD_TYPE_GASES_FROM_CLUSTERS)
        .value("CLUSTER_ACCEL_BUILD_TYPE_CLUSTERS_FROM_TRIANGLES",
               OPTIX_CLUSTER_ACCEL_BUILD_TYPE_CLUSTERS_FROM_TRIANGLES)
        .value("CLUSTER_ACCEL_BUILD_TYPE_TEMPLATES_FROM_TRIANGLES",
               OPTIX_CLUSTER_ACCEL_BUILD_TYPE_TEMPLATES_FROM_TRIANGLES)
        .value("CLUSTER_ACCEL_BUILD_TYPE_CLUSTERS_FROM_TEMPLATES",
               OPTIX_CLUSTER_ACCEL_BUILD_TYPE_CLUSTERS_FROM_TEMPLATES)
        .value("CLUSTER_ACCEL_BUILD_TYPE_TEMPLATES_FROM_GRIDS", OPTIX_CLUSTER_ACCEL_BUILD_TYPE_TEMPLATES_FROM_GRIDS)
        .export_values();
    py::enum_<OptixClusterAccelBuildMode>(m, "ClusterAccelBuildMode")
        .value("CLUSTER_ACCEL_BUILD_MODE_IMPLICIT_DESTINATIONS", OPTIX_CLUSTER_ACCEL_BUILD_MODE_IMPLICIT_DESTINATIONS)
        .value("CLUSTER_ACCEL_BUILD_MODE_EXPLICIT_DESTINATIONS", OPTIX_CLUSTER_ACCEL_BUILD_MODE_EXPLICIT_DESTINATIONS)
        .value("CLUSTER_ACCEL_BUILD_MODE_GET_SIZES", OPTIX_CLUSTER_ACCEL_BUILD_MODE_GET_SIZES)
        .export_values();
    py::enum_<OptixClusterAccelBuildFlags>(m, "ClusterAccelBuildFlags", py::arithmetic())
        .value("CLUSTER_ACCEL_BUILD_FLAG_NONE", OPTIX_CLUSTER_ACCEL_BUILD_FLAG_NONE)
        .value("CLUSTER_ACCEL_BUILD_FLAG_PREFER_FAST_TRACE", OPTIX_CLUSTER_ACCEL_BUILD_FLAG_PREFER_FAST_TRACE)
        .value("CLUSTER_ACCEL_BUILD_FLAG_PREFER_FAST_BUILD", OPTIX_CLUSTER_ACCEL_BUILD_FLAG_PREFER_FAST_BUILD)
        .value("CLUSTER_ACCEL_BUILD_FLAG_ALLOW_OPACITY_MICROMAPS",
               OPTIX_CLUSTER_ACCEL_BUILD_FLAG_ALLOW_OPACITY_MICROMAPS)
        .export_values();

    m.def(
        "clusterAccelComputeMemoryUsage",
        [](const pyoptix::DeviceContext& context, OptixClusterAccelBuildMode mode,
           const OptixClusterAccelBuildInput& input) {
            ensure_optix_initialized();
            OptixAccelBufferSizes sizes{};
            PYOPTIX_CHECK(optixClusterAccelComputeMemoryUsage(context.deviceContext, mode, &input, &sizes));
            return py::make_tuple(sizes.outputSizeInBytes, sizes.tempSizeInBytes);
        },
        py::arg("context"), py::arg("mode"), py::arg("input"));
    m.def(
        "clusterAccelBuild",
        [](const pyoptix::DeviceContext& context, uintptr_t stream, const OptixClusterAccelBuildModeDesc& mode,
           const OptixClusterAccelBuildInput& input, uintptr_t args, uintptr_t args_count, unsigned int stride) {
            ensure_optix_initialized();
            // Device descriptors and their pointees follow the native API's lifetime contract.
            // The caller retains them and all output allocations until stream completion.
            // args_count is a device pointer to a uint32 argument count; 0 uses maxArgCount.
            PYOPTIX_CHECK(optixClusterAccelBuild(context.deviceContext, reinterpret_cast<CUstream>(stream), &mode,
                                                 &input, args, args_count, stride));
        },
        py::arg("context"), py::arg("stream"), py::arg("mode"), py::arg("input"), py::arg("args"),
        py::arg("args_count") = 0, py::arg("stride") = 0);
}
