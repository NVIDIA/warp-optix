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

# The headers are fetched as an archive of an immutable commit and verified
# against a SHA512 hash, so a moved tag or tampered download fails the build
# instead of compiling unverified code. The header digest also covers local
# copies supplied through FETCHCONTENT_SOURCE_DIR_OPTIX_HEADERS. Update all
# values together when the pyoptix pin moves to a new OptiX SDK (the commit is
# what the optix-dev tag v<version> points to). PyOptiX releases are numbered
# after the SDK they wrap, so pyoptix 9.1.x wraps OptiX 9.1.0.
set(WARP_OPTIX_OPTIX_VERSION "9.1.0")
set(_optix_headers_commit "f1f6dd803f3159992d248178f6e09421c6eb8b6d")
set(_optix_headers_sha512 "54f98235206d0156552f8004438c072e0796c16d75a2cb9abc0367ce5d00b281d2ec34173316bf8c34e81e302bc0042f1dcbe3d17ac7a0ee980a14493991d7a1")
set(_optix_headers_digest "119d657b3401b5ce5b1e81115c0f181b75d83a9a57811fa0a1ebc9edb4f68a58")

string(REGEX MATCH "^[0-9]+\\.[0-9]+" _optix_major_minor "${WARP_OPTIX_OPTIX_VERSION}")
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" _project_metadata)
string(REGEX MATCH "\"pyoptix==([0-9]+\\.[0-9]+)\\.[0-9]+\"" _pyoptix_requirement "${_project_metadata}")
if(NOT _pyoptix_requirement OR NOT CMAKE_MATCH_1 STREQUAL _optix_major_minor)
  message(FATAL_ERROR
    "The pyoptix pin in pyproject.toml must wrap OptiX ${WARP_OPTIX_OPTIX_VERSION}; "
    "update the pinned OptiX headers commit and hashes in CMakeLists.txt together with it")
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

# The OptiX function table includes cuda.h for driver API types. The extension
# resolves the one driver function it calls at runtime, so it links no CUDA
# library and needs no CUDA Toolkit: the header comes from the
# nvidia-cuda-runtime-cu12 build requirement, or from CUDA_PATH when set.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
            "import nvidia.cuda_runtime, pathlib; print(pathlib.Path(nvidia.cuda_runtime.__path__[0], 'include').as_posix())"
    OUTPUT_VARIABLE _cuda_runtime_wheel_include
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
# Isolated build environments are temporary, so search again on every configure.
unset(WARP_OPTIX_CUDA_INCLUDE_DIR CACHE)
find_path(WARP_OPTIX_CUDA_INCLUDE_DIR cuda.h
          HINTS "${_cuda_runtime_wheel_include}" "$ENV{CUDA_PATH}/include"
          REQUIRED)
message(STATUS "Using cuda.h from: ${WARP_OPTIX_CUDA_INCLUDE_DIR}")

#------------------------------------------------------------------------------
# Main build
#------------------------------------------------------------------------------

python_add_library(_ext MODULE src/extension.cpp WITH_SOABI)
set_target_properties(_ext PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

target_include_directories(_ext SYSTEM PRIVATE "${WARP_OPTIX_CUDA_INCLUDE_DIR}")
target_link_libraries(_ext PRIVATE
    OptiX::OptiX
    pybind11::headers
    ${CMAKE_DL_LIBS}
)


target_compile_features(_ext PRIVATE cxx_std_17)

# MSVC 17.10+ constructs std::mutex constexpr, which crashes on lock when an
# msvcp140.dll older than 14.40 is loaded. Keep the wheel safe on such systems.
if(MSVC)
  target_compile_definitions(_ext PRIVATE _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
endif()

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
