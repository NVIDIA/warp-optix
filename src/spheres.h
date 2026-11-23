// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// PyOptiX 9.1.1 binds no sphere build input, so sphere structures are sized and built here.
// Included by extension.cpp: reuses its checked PyOptiX context bridge and OptiX table.
struct SphereArray {
    uintptr_t centers = 0;
    uintptr_t radii = 0;
    unsigned int count = 0;
    bool singleRadius = false;
    unsigned int flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
};

namespace {
// The OptiX inputs point into `inputs` and `pointers`, which must outlive the call using them.
std::vector<OptixBuildInput> sphere_build_inputs(const std::vector<SphereArray>& inputs,
                                                 std::vector<CUdeviceptr>& pointers) {
    pointers.resize(2 * inputs.size());
    std::vector<OptixBuildInput> result(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        const SphereArray& spheres = inputs[i];
        OptixBuildInput& input = result[i];
        input.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;
        OptixBuildInputSphereArray& array = input.sphereArray;
        array.flags = &spheres.flags;
        array.numSbtRecords = 1;
        // OptiX requires null buffers for empty geometry, which builds a structure nothing intersects.
        if (spheres.count == 0)
            continue;
        require_device_range(spheres.centers, checked_mul(spheres.count, 12, "sphere centers"), "centers");
        require_device_range(spheres.radii, spheres.singleRadius ? 4 : checked_mul(spheres.count, 4, "radii"), "radii");
        pointers[2 * i] = static_cast<CUdeviceptr>(spheres.centers);
        pointers[2 * i + 1] = static_cast<CUdeviceptr>(spheres.radii);
        array.vertexBuffers = &pointers[2 * i];
        array.numVertices = spheres.count;
        array.radiusBuffers = &pointers[2 * i + 1];
        array.singleRadius = spheres.singleRadius;
    }
    return result;
}

OptixAccelBuildOptions sphere_build_options(unsigned int buildFlags, unsigned int operation) {
    OptixAccelBuildOptions options{};
    options.buildFlags = buildFlags;
    options.operation = static_cast<OptixBuildOperation>(operation);
    return options;
}
} // namespace

inline void bind_spheres(py::module_& m) {
    py::class_<SphereArray>(m, "SphereArray")
        .def(py::init<uintptr_t, uintptr_t, unsigned int, bool, unsigned int>(), py::arg("centers"), py::arg("radii"),
             py::arg("count"), py::arg("singleRadius") = false,
             py::arg("flags") = static_cast<unsigned int>(OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT))
        .def_readonly("count", &SphereArray::count);
    m.attr("PRIMITIVE_TYPE_SPHERE") = static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_SPHERE);
    m.attr("PRIMITIVE_TYPE_FLAGS_SPHERE") = static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE);

    m.def(
        "sphereAccelComputeMemoryUsage",
        [](const pyoptix::DeviceContext& context, unsigned int buildFlags, unsigned int operation,
           const std::vector<SphereArray>& inputs) {
            ensure_optix_initialized();
            std::vector<CUdeviceptr> pointers;
            const std::vector<OptixBuildInput> native = sphere_build_inputs(inputs, pointers);
            const OptixAccelBuildOptions options = sphere_build_options(buildFlags, operation);
            OptixAccelBufferSizes sizes{};
            PYOPTIX_CHECK(optixAccelComputeMemoryUsage(context.deviceContext, &options, native.data(),
                                                       static_cast<unsigned int>(native.size()), &sizes));
            return py::make_tuple(sizes.outputSizeInBytes, sizes.tempSizeInBytes, sizes.tempUpdateSizeInBytes);
        },
        py::arg("context"), py::arg("buildFlags"), py::arg("operation"), py::arg("inputs"));
    m.def(
        "sphereAccelBuild",
        [](const pyoptix::DeviceContext& context, uintptr_t stream, unsigned int buildFlags, unsigned int operation,
           const std::vector<SphereArray>& inputs, uintptr_t temp, size_t tempSize, uintptr_t output, size_t outputSize,
           uintptr_t compactedSize) {
            ensure_optix_initialized();
            std::vector<CUdeviceptr> pointers;
            const std::vector<OptixBuildInput> native = sphere_build_inputs(inputs, pointers);
            const OptixAccelBuildOptions options = sphere_build_options(buildFlags, operation);
            OptixAccelEmitDesc emitted{static_cast<CUdeviceptr>(compactedSize), OPTIX_PROPERTY_TYPE_COMPACTED_SIZE};
            OptixTraversableHandle handle = 0;
            PYOPTIX_CHECK(optixAccelBuild(context.deviceContext, reinterpret_cast<CUstream>(stream), &options,
                                          native.data(), static_cast<unsigned int>(native.size()), temp, tempSize,
                                          output, outputSize, &handle, compactedSize ? &emitted : nullptr,
                                          compactedSize ? 1u : 0u));
            return static_cast<uintptr_t>(handle);
        },
        py::arg("context"), py::arg("stream"), py::arg("buildFlags"), py::arg("operation"), py::arg("inputs"),
        py::arg("temp"), py::arg("tempSize"), py::arg("output"), py::arg("outputSize"), py::arg("compactedSize") = 0);
}
