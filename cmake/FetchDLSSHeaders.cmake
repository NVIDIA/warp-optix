# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Provides DLSS::Headers, an interface target with the NGX SDK headers needed for
# DLSS Ray Reconstruction in header-only mode (NVSDK_NGX_HEADER_ONLY), and
# DLSS_LICENSE_FILE, the SDK's license (NVIDIA RTX SDKs License).
#
# Header-only mode compiles portions of the SDK into the extension: the NGX
# runtime loader (nvsdk_ngx_loader.h, nvsdk_ngx_standalone_*.h) and the
# declarations and parameter-name strings of the remaining headers. No SDK
# library or DLSS runtime is linked or installed; the driver (R620+) ships the
# Ray Reconstruction runtime. Builds with DLSS are therefore governed by the
# NVIDIA RTX SDKs License, which is installed alongside the extension (see
# THIRD_PARTY_NOTICES.txt).
#
# The headers are proprietary NVIDIA SDK files, so they are downloaded at build
# time into the git-ignored third_party/ directory and never committed. Each file
# is fetched from an immutable commit and verified against its SHA256.
#
# Set DLSS_ROOT to an existing SDK (with include/nvsdk_ngx.h and LICENSE.txt) to
# skip the download.

if(TARGET DLSS::Headers)
  return()
endif()

set(WARP_OPTIX_DLSS_VERSION "310.9.1")
# The commit the annotated NVIDIA/DLSS tag v310.9.1 points to (refs/tags/v310.9.1^{}).
set(_dlss_commit "374959484e79a640feaba44c93ac8cfb0a03f5b5")
# Paths relative to the SDK root.
set(_dlss_files
  include/nvsdk_ngx.h                   dc38e7467cf415379c9d12ae1b6e4a494c453ed92720fb53e92aecb523e7b848
  include/nvsdk_ngx_defs.h              ea23f33497cd274860d1c25a97644fce807dcb0037c594547203343103fad03e
  include/nvsdk_ngx_params.h            943bc8cc5cdae03b6303016fbad3183636f2335ae27a2d18776798c3b4efabbc
  include/nvsdk_ngx_loader.h            abdf5f66a692620538357d29fdf78e3e2a677ee8140b7a638962ecb766585e75
  include/nvsdk_ngx_standalone_common.h 38413c3338b0aea0202c07da95d426bfee7080a42913823fa2d9dd585b087b53
  include/nvsdk_ngx_standalone_cuda.h   b23436ed5040051ac8e81554c9af080f2463c941435bbc3b82222e5431a09499
  include/nvsdk_ngx_defs_dlssd.h        2e97fe1595b1334f71b51724f2cb67496e8c296a7a2eec5606da53055882e716
  LICENSE.txt                           d4216e39ebef5f9b50a6712ebb37beeb5379862a67733a9999c651f21592aaf0
)

if(DEFINED ENV{DLSS_ROOT} AND NOT DEFINED DLSS_ROOT)
  set(DLSS_ROOT "$ENV{DLSS_ROOT}")
endif()

if(DLSS_ROOT)
  file(TO_CMAKE_PATH "${DLSS_ROOT}" _dlss_root)
  foreach(_required include/nvsdk_ngx.h LICENSE.txt)
    if(NOT EXISTS "${_dlss_root}/${_required}")
      message(FATAL_ERROR "DLSS_ROOT=${DLSS_ROOT} does not contain ${_required}")
    endif()
  endforeach()
  message(STATUS "Using DLSS SDK headers from ${_dlss_root}/include")
else()
  set(_dlss_root "${CMAKE_SOURCE_DIR}/third_party/dlss-headers-${WARP_OPTIX_DLSS_VERSION}")
  list(LENGTH _dlss_files _dlss_entries)
  math(EXPR _dlss_last "${_dlss_entries} - 1")
  foreach(_index RANGE 0 ${_dlss_last} 2)
    math(EXPR _hash_index "${_index} + 1")
    list(GET _dlss_files ${_index} _name)
    list(GET _dlss_files ${_hash_index} _hash)
    set(_path "${_dlss_root}/${_name}")
    if(EXISTS "${_path}")
      file(SHA256 "${_path}" _existing)
      if(_existing STREQUAL _hash)
        continue()
      endif()
    endif()
    set(_url "https://raw.githubusercontent.com/NVIDIA/DLSS/${_dlss_commit}/${_name}")
    message(STATUS "Fetching DLSS SDK file ${_name}")
    file(DOWNLOAD "${_url}" "${_path}" EXPECTED_HASH SHA256=${_hash} STATUS _status TLS_VERIFY ON)
    list(GET _status 0 _code)
    if(NOT _code EQUAL 0)
      list(GET _status 1 _message)
      file(REMOVE "${_path}")
      message(FATAL_ERROR
        "Failed to fetch ${_url}: ${_message}. Set DLSS_ROOT to a local DLSS SDK, "
        "or disable DLSS with -DWARP_OPTIX_ENABLE_DLSS=OFF.")
    endif()
  endforeach()
  message(STATUS "Using DLSS SDK ${WARP_OPTIX_DLSS_VERSION} headers from ${_dlss_root}/include")
endif()

set(DLSS_LICENSE_FILE "${_dlss_root}/LICENSE.txt")
# Building with the SDK is use of the SDK, so make its license visible to the builder.
message(STATUS
  "DLSS support uses the NVIDIA DLSS SDK, which is licensed under the NVIDIA RTX SDKs "
  "License (${DLSS_LICENSE_FILE}), not Apache-2.0. Building or distributing warp-optix "
  "with WARP_OPTIX_ENABLE_DLSS=ON means accepting its terms.")

add_library(DLSS::Headers INTERFACE IMPORTED)
target_include_directories(DLSS::Headers INTERFACE "${_dlss_root}/include")
# The header-only loader reads the driver location from the registry (Windows)
# or dlopens libnvidia-ngx.so.1 (Linux).
if(WIN32)
  target_link_libraries(DLSS::Headers INTERFACE advapi32)
else()
  target_link_libraries(DLSS::Headers INTERFACE ${CMAKE_DL_LIBS})
endif()
