# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Provides DLSS::Headers, an interface target with the three NGX SDK headers that
# declare the types, enums, and parameter names used by src/dlss.cpp.
#
# The headers contribute declarations only: no SDK library, loader
# (NVSDK_NGX_HEADER_ONLY), or runtime is compiled in, linked, or installed. The
# driver's NGX core is loaded by warp-optix's own loader, and the driver (R620+)
# ships the Ray Reconstruction runtime. The headers are proprietary NVIDIA SDK
# files, so they are downloaded at build time into the git-ignored third_party/
# directory and never committed or packaged. Each file is fetched from an
# immutable commit and verified against its SHA256; pinning the SDK version also
# pins the NGX API version (NVSDK_NGX_Version_API) warp-optix requests.
#
# Set DLSS_ROOT to an existing SDK (with include/nvsdk_ngx.h) to skip the download.

if(TARGET DLSS::Headers)
  return()
endif()

set(WARP_OPTIX_DLSS_VERSION "310.9.1")
# The commit the annotated NVIDIA/DLSS tag v310.9.1 points to (refs/tags/v310.9.1^{}).
set(_dlss_commit "374959484e79a640feaba44c93ac8cfb0a03f5b5")
set(_dlss_headers
  nvsdk_ngx_defs.h       ea23f33497cd274860d1c25a97644fce807dcb0037c594547203343103fad03e
  nvsdk_ngx_params.h     943bc8cc5cdae03b6303016fbad3183636f2335ae27a2d18776798c3b4efabbc
  nvsdk_ngx_defs_dlssd.h 2e97fe1595b1334f71b51724f2cb67496e8c296a7a2eec5606da53055882e716
)

if(DEFINED ENV{DLSS_ROOT} AND NOT DEFINED DLSS_ROOT)
  set(DLSS_ROOT "$ENV{DLSS_ROOT}")
endif()

if(DLSS_ROOT)
  file(TO_CMAKE_PATH "${DLSS_ROOT}" _dlss_root)
  set(_dlss_include_dir "${_dlss_root}/include")
  if(NOT EXISTS "${_dlss_include_dir}/nvsdk_ngx_defs.h")
    message(FATAL_ERROR "DLSS_ROOT=${DLSS_ROOT} does not contain include/nvsdk_ngx_defs.h")
  endif()
  message(STATUS "Using DLSS SDK headers from ${_dlss_include_dir}")
else()
  set(_dlss_include_dir "${CMAKE_SOURCE_DIR}/third_party/dlss-headers-${WARP_OPTIX_DLSS_VERSION}/include")
  list(LENGTH _dlss_headers _dlss_entries)
  math(EXPR _dlss_last "${_dlss_entries} - 1")
  foreach(_index RANGE 0 ${_dlss_last} 2)
    math(EXPR _hash_index "${_index} + 1")
    list(GET _dlss_headers ${_index} _name)
    list(GET _dlss_headers ${_hash_index} _hash)
    set(_path "${_dlss_include_dir}/${_name}")
    if(EXISTS "${_path}")
      file(SHA256 "${_path}" _existing)
      if(_existing STREQUAL _hash)
        continue()
      endif()
    endif()
    set(_url "https://raw.githubusercontent.com/NVIDIA/DLSS/${_dlss_commit}/include/${_name}")
    message(STATUS "Fetching DLSS SDK header ${_name}")
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
  message(STATUS "Using DLSS SDK ${WARP_OPTIX_DLSS_VERSION} headers from ${_dlss_include_dir}")
endif()

add_library(DLSS::Headers INTERFACE IMPORTED)
target_include_directories(DLSS::Headers INTERFACE "${_dlss_include_dir}")
# src/dlss.cpp dlopens libnvidia-ngx.so.1 on Linux.
if(NOT WIN32)
  target_link_libraries(DLSS::Headers INTERFACE ${CMAKE_DL_LIBS})
endif()
