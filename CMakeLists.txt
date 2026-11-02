# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Use of this source code is governed by a BSD-style
# license that can be found in LICENSE.txt.

cmake_minimum_required(VERSION 3.17...3.31)

# Use scikit-build-core provided project name and version
project(
    ${SKBUILD_PROJECT_NAME}
    VERSION ${SKBUILD_PROJECT_VERSION}
    LANGUAGES CXX
)

#------------------------------------------------------------------------------
# Python and pybind11
#------------------------------------------------------------------------------

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

#------------------------------------------------------------------------------
# Fetch OptiX headers from NVIDIA/optix-dev repository
#------------------------------------------------------------------------------

include(FetchContent)
include(CheckOptixVersion)

# Use the dependency pin as the SDK version for fetching and validation.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" _project_metadata)
string(REGEX MATCH "\"pyoptix==([0-9]+\\.[0-9]+\\.[0-9]+)\""
       _pyoptix_requirement "${_project_metadata}")
if(NOT _pyoptix_requirement)
  message(FATAL_ERROR "warp-optix requires an exact pyoptix SDK version pin in pyproject.toml")
endif()
set(WARP_OPTIX_OPTIX_VERSION "${CMAKE_MATCH_1}")

# The headers are fetched as an archive of an immutable commit and verified
# against a SHA512 hash, so a moved tag or tampered download fails the build
# instead of compiling unverified code. The header digest also covers local
# copies supplied through FETCHCONTENT_SOURCE_DIR_OPTIX_HEADERS. Update all
# values together with the pyoptix pin (the commit is what the optix-dev tag
# v<version> points to).
set(_optix_headers_version "9.1.0")
set(_optix_headers_commit "f1f6dd803f3159992d248178f6e09421c6eb8b6d")
set(_optix_headers_sha512 "54f98235206d0156552f8004438c072e0796c16d75a2cb9abc0367ce5d00b281d2ec34173316bf8c34e81e302bc0042f1dcbe3d17ac7a0ee980a14493991d7a1")
set(_optix_headers_digest "119d657b3401b5ce5b1e81115c0f181b75d83a9a57811fa0a1ebc9edb4f68a58")
if(NOT WARP_OPTIX_OPTIX_VERSION VERSION_EQUAL _optix_headers_version)
  message(FATAL_ERROR
    "pyoptix==${WARP_OPTIX_OPTIX_VERSION} requires updating the pinned OptiX "
    "headers commit and hash in CMakeLists.txt (currently ${_optix_headers_version})")
endif()

message(STATUS "Fetching OptiX headers from NVIDIA/optix-dev@${_optix_headers_commit}...")
FetchContent_Declare(
    optix_headers
    URL      https://github.com/NVIDIA/optix-dev/archive/${_optix_headers_commit}.tar.gz
    URL_HASH SHA512=${_optix_headers_sha512}
)
FetchContent_MakeAvailable(optix_headers)
warp_optix_check_optix_version("${optix_headers_SOURCE_DIR}/include/optix.h"
                              "${WARP_OPTIX_OPTIX_VERSION}")
warp_optix_check_optix_headers("${optix_headers_SOURCE_DIR}/include"
                               "${_optix_headers_digest}")

# Create OptiX::OptiX interface target using fetched headers
add_library(OptiX::OptiX INTERFACE IMPORTED)
target_include_directories(OptiX::OptiX INTERFACE ${optix_headers_SOURCE_DIR}/include)

message(STATUS "Using OptiX headers from: ${optix_headers_SOURCE_DIR}/include")

# Find CUDA Toolkit (required for CUDA runtime)
if(NOT TARGET CUDA::cuda_driver)
  find_package(CUDAToolkit 10.0 REQUIRED)
endif()

#------------------------------------------------------------------------------
# Main build
#------------------------------------------------------------------------------

python_add_library(_ext MODULE src/extension.cpp WITH_SOABI)
set_target_properties(_ext PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

target_link_libraries(_ext PRIVATE
    OptiX::OptiX
    CUDA::cuda_driver
    CUDA::cudart
    pybind11::headers
)


target_compile_features(_ext PRIVATE cxx_std_17)

# The dependency owns optix._optix. Install only our supplemental extension,
# and shader headers under warp_optix.
set(WARP_OPTIX_INSTALL_DIR "${SKBUILD_PLATLIB_DIR}/warp_optix")
install(TARGETS _ext DESTINATION ${WARP_OPTIX_INSTALL_DIR})
# Warp compiles OptiX kernels at runtime, so wheels need the unmodified SDK
# headers as well as the extension. These retain their separate SDK license.
install(DIRECTORY "${optix_headers_SOURCE_DIR}/include/"
        DESTINATION ${WARP_OPTIX_INSTALL_DIR}/include)
install(FILES "${optix_headers_SOURCE_DIR}/LICENSE.txt"
        DESTINATION ${WARP_OPTIX_INSTALL_DIR}/include RENAME OptiX_LICENSE.txt)
