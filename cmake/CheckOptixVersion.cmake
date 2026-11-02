# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

function(warp_optix_check_optix_version header expected_version)
  if(NOT EXISTS "${header}")
    message(FATAL_ERROR "OptiX SDK header is missing: ${header}")
  endif()
  file(STRINGS "${header}" _version_line
       REGEX "^[ \t]*#[ \t]*define[ \t]+OPTIX_VERSION[ \t]+[0-9]+")
  string(REGEX MATCH "OPTIX_VERSION[ \t]+([0-9]+)" _match "${_version_line}")
  if(NOT _match)
    message(FATAL_ERROR "Cannot determine OptiX SDK version from ${header}")
  endif()
  set(_sdk_version "${CMAKE_MATCH_1}")
  math(EXPR _major "${_sdk_version} / 10000")
  math(EXPR _minor "(${_sdk_version} % 10000) / 100")
  math(EXPR _patch "${_sdk_version} % 100")
  set(_actual_version "${_major}.${_minor}.${_patch}")
  if(NOT "${_actual_version}" STREQUAL "${expected_version}")
    message(FATAL_ERROR
      "OptiX SDK version mismatch: headers are ${_actual_version}, but "
      "pyoptix==${expected_version} requires OptiX ${expected_version}. "
      "Use matching headers and clear any FETCHCONTENT_SOURCE_DIR_OPTIX_HEADERS override. "
      "Header: ${header}")
  endif()
endfunction()

# Verifies the OptiX header tree against a digest of every file's SHA256, so
# headers supplied through FETCHCONTENT_SOURCE_DIR_OPTIX_HEADERS (which bypasses
# the archive URL_HASH) are held to the same content as the pinned download.
function(warp_optix_check_optix_headers include_dir expected_digest)
  file(GLOB_RECURSE _files RELATIVE "${include_dir}" "${include_dir}/*")
  list(SORT _files)
  set(_manifest "")
  foreach(_file IN LISTS _files)
    file(SHA256 "${include_dir}/${_file}" _hash)
    string(APPEND _manifest "${_file} ${_hash}\n")
  endforeach()
  string(SHA256 _digest "${_manifest}")
  if(NOT _digest STREQUAL expected_digest)
    message(FATAL_ERROR
      "OptiX headers in ${include_dir} do not match the pinned OptiX "
      "${WARP_OPTIX_OPTIX_VERSION} headers (digest ${_digest}). Use the pinned "
      "download or an unmodified copy of NVIDIA/optix-dev@${_optix_headers_commit}.")
  endif()
endfunction()
