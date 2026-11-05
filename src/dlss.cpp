// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// DLSS Ray Reconstruction bindings on the NVIDIA driver's NGX runtime.
//
// No NVIDIA code is compiled in or linked, and no NVIDIA binary is shipped:
// - The driver's NGX core (_nvngx.dll / libnvidia-ngx.so.1) is located and
//   loaded by the small loader below, written for warp-optix, and its exported
//   NVSDK_NGX_CUDA_* entry points are called through function pointers.
// - Since driver R620 the driver also ships the Ray Reconstruction runtime
//   (ngx-builtins/nvngx_dlssd), which NGX loads itself.
// - Only declarations come from the DLSS SDK: types, enums, and parameter-name
//   macros in nvsdk_ngx_defs.h, nvsdk_ngx_params.h and nvsdk_ngx_defs_dlssd.h,
//   pinned to SDK 310.9.1 and fetched at build time (never committed). The
//   SDK's own loader (NVSDK_NGX_HEADER_ONLY) and static library are not used.
//
// The driver exports no C parameter setters (NVSDK_NGX_Parameter_SetUI, ...),
// so parameters are set through the NVSDK_NGX_Parameter virtual interface, as
// Blender Cycles does.

#ifdef WARP_OPTIX_ENABLE_DLSS

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <cuda.h>

#include <nvsdk_ngx_defs.h>
#include <nvsdk_ngx_defs_dlssd.h>
#include <nvsdk_ngx_params.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <array>
#include <type_traits>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace py = pybind11;

namespace {

// warp-optix has no NVIDIA-issued application ID, so NGX identifies it by project ID.
constexpr const char* kDefaultProjectId = "dddbee68-a452-4fab-9371-f9575480a154";

// ---------------------------------------------------------------------------
// NGX core loader
// ---------------------------------------------------------------------------

// Exported entry points of the driver's NGX core, by exported symbol name.
struct NgxCore {
    NVSDK_NGX_Result(NVSDK_CONV* Init_ProjectID)(const char*, NVSDK_NGX_EngineType, const char*, const wchar_t*,
                                                 NVSDK_NGX_Version, const NVSDK_NGX_FeatureCommonInfo*) = nullptr;
    NVSDK_NGX_Result(NVSDK_CONV* Shutdown)() = nullptr;
    NVSDK_NGX_Result(NVSDK_CONV* GetCapabilityParameters)(NVSDK_NGX_Parameter**) = nullptr;
    NVSDK_NGX_Result(NVSDK_CONV* AllocateParameters)(NVSDK_NGX_Parameter**) = nullptr;
    NVSDK_NGX_Result(NVSDK_CONV* DestroyParameters)(NVSDK_NGX_Parameter*) = nullptr;
    NVSDK_NGX_Result(NVSDK_CONV* CreateFeature)(NVSDK_NGX_Feature, const NVSDK_NGX_Parameter*,
                                                NVSDK_NGX_Handle**) = nullptr;
    // The progress callback is unused and always null.
    NVSDK_NGX_Result(NVSDK_CONV* EvaluateFeature)(const NVSDK_NGX_Handle*, const NVSDK_NGX_Parameter*, void*) = nullptr;
    NVSDK_NGX_Result(NVSDK_CONV* ReleaseFeature)(NVSDK_NGX_Handle*) = nullptr;
    NVSDK_NGX_Result(NVSDK_CONV* GetFeatureRequirements)(int, const NVSDK_NGX_FeatureDiscoveryInfo*,
                                                         NVSDK_NGX_FeatureRequirement*) = nullptr;
    std::string error; // why loading failed; empty on success
};

#ifdef _WIN32
// The NGX core sits in the display driver's DriverStore folder, next to the CUDA
// driver (nvcuda64.dll) that is already loaded into the process. Using that exact
// folder ties NGX to the driver CUDA runs on and never searches writable paths.
std::wstring driverStoreFolder() {
    for (const wchar_t* name : {L"nvcuda64.dll", L"nvcuda.dll"}) {
        HMODULE module = GetModuleHandleW(name);
        if (!module)
            continue;
        std::wstring path(MAX_PATH, L'\0');
        for (;;) {
            const DWORD size = GetModuleFileNameW(module, path.data(), static_cast<DWORD>(path.size()));
            if (size == 0)
                break;
            if (size < path.size()) {
                path.resize(size);
                const std::filesystem::path folder = std::filesystem::path(path).parent_path();
                if (std::filesystem::exists(folder / L"_nvngx.dll"))
                    return folder.wstring();
                break;
            }
            path.resize(path.size() * 2);
        }
    }
    return {};
}
#endif

NgxCore loadCore() {
    NgxCore core;
    // The CUDA driver must be loaded first: it locates the NGX core on Windows,
    // and NGX itself requires an initialized CUDA driver.
    if (cuInit(0) != CUDA_SUCCESS) {
        core.error = "the CUDA driver could not be initialized";
        return core;
    }
#ifdef _WIN32
    const std::wstring folder = driverStoreFolder();
    if (folder.empty()) {
        core.error = "the NVIDIA driver's NGX library (_nvngx.dll) was not found next to the CUDA driver";
        return core;
    }
    const std::wstring path = folder + L"\\_nvngx.dll";
    // Resolve the core's own dependencies from its folder and System32 only.
    HMODULE library =
        LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
    auto symbol = [&](const char* name) { return reinterpret_cast<void*>(GetProcAddress(library, name)); };
#else
    void* library = dlopen("libnvidia-ngx.so.1", RTLD_NOW | RTLD_LOCAL);
    auto symbol = [&](const char* name) { return dlsym(library, name); };
#endif
    if (!library) {
        core.error = "the NVIDIA driver's NGX library could not be loaded";
        return core;
    }
    // The library stays loaded for the life of the process.
    auto bind = [&](auto& function, const char* name) {
        function = reinterpret_cast<std::remove_reference_t<decltype(function)>>(symbol(name));
        if (!function && core.error.empty())
            core.error = std::string("the NVIDIA driver's NGX library lacks ") + name + " (driver too old)";
    };
    bind(core.Init_ProjectID, "NVSDK_NGX_CUDA_Init_ProjectID");
    bind(core.Shutdown, "NVSDK_NGX_CUDA_Shutdown");
    bind(core.GetCapabilityParameters, "NVSDK_NGX_CUDA_GetCapabilityParameters");
    bind(core.AllocateParameters, "NVSDK_NGX_CUDA_AllocateParameters");
    bind(core.DestroyParameters, "NVSDK_NGX_CUDA_DestroyParameters");
    bind(core.CreateFeature, "NVSDK_NGX_CUDA_CreateFeature");
    bind(core.EvaluateFeature, "NVSDK_NGX_CUDA_EvaluateFeature");
    bind(core.ReleaseFeature, "NVSDK_NGX_CUDA_ReleaseFeature");
    bind(core.GetFeatureRequirements, "NVSDK_NGX_CUDA_GetFeatureRequirements");
    return core;
}

// Loaded once per process; returns null (with the reason in *error) if unavailable.
const NgxCore* ngxCore(std::string* error = nullptr) {
    static const NgxCore core = loadCore();
    if (!core.error.empty()) {
        if (error)
            *error = core.error;
        return nullptr;
    }
    return &core;
}

const NgxCore& requireCore() {
    std::string error;
    const NgxCore* core = ngxCore(&error);
    if (!core)
        throw std::runtime_error("DLSS Ray Reconstruction is unavailable: " + error);
    return *core;
}

// GetNGXResultAsString is a static-library export, so name the results here.
std::string resultString(NVSDK_NGX_Result result) {
    switch (result) {
    case NVSDK_NGX_Result_Success:
        return "Success";
    case NVSDK_NGX_Result_FAIL_FeatureNotSupported:
        return "FeatureNotSupported";
    case NVSDK_NGX_Result_FAIL_PlatformError:
        return "PlatformError (NGX driver library not found?)";
    case NVSDK_NGX_Result_FAIL_FeatureAlreadyExists:
        return "FeatureAlreadyExists";
    case NVSDK_NGX_Result_FAIL_FeatureNotFound:
        return "FeatureNotFound (no DLSS RR runtime; requires driver R620+)";
    case NVSDK_NGX_Result_FAIL_InvalidParameter:
        return "InvalidParameter";
    case NVSDK_NGX_Result_FAIL_ScratchBufferTooSmall:
        return "ScratchBufferTooSmall";
    case NVSDK_NGX_Result_FAIL_NotInitialized:
        return "NotInitialized";
    case NVSDK_NGX_Result_FAIL_UnsupportedInputFormat:
        return "UnsupportedInputFormat";
    case NVSDK_NGX_Result_FAIL_RWFlagMissing:
        return "RWFlagMissing";
    case NVSDK_NGX_Result_FAIL_MissingInput:
        return "MissingInput";
    case NVSDK_NGX_Result_FAIL_UnableToInitializeFeature:
        return "UnableToInitializeFeature";
    case NVSDK_NGX_Result_FAIL_OutOfDate:
        return "OutOfDate (driver too old)";
    case NVSDK_NGX_Result_FAIL_OutOfGPUMemory:
        return "OutOfGPUMemory";
    case NVSDK_NGX_Result_FAIL_UnsupportedFormat:
        return "UnsupportedFormat";
    case NVSDK_NGX_Result_FAIL_UnableToWriteToAppDataPath:
        return "UnableToWriteToAppDataPath";
    case NVSDK_NGX_Result_FAIL_UnsupportedParameter:
        return "UnsupportedParameter";
    case NVSDK_NGX_Result_FAIL_Denied:
        return "Denied";
    case NVSDK_NGX_Result_FAIL_NotImplemented:
        return "NotImplemented";
    default:
        return NVSDK_NGX_FAILED(result) ? "Fail" : "Success";
    }
}

// A fresh parameter block per call, so values never leak between calls.
struct Parameters {
    NVSDK_NGX_Parameter* params = nullptr;
    Parameters();
    Parameters(const Parameters&) = delete;
    Parameters& operator=(const Parameters&) = delete;
    ~Parameters() {
        if (params)
            requireCore().DestroyParameters(params);
    }
    NVSDK_NGX_Parameter* operator->() const { return params; }
};

void check(NVSDK_NGX_Result result, const char* what) {
    if (NVSDK_NGX_FAILED(result))
        throw std::runtime_error(std::string(what) + " failed: " + resultString(result) + " (" +
                                 std::to_string(static_cast<unsigned>(result)) + ")");
}

Parameters::Parameters() {
    check(requireCore().AllocateParameters(&params), "NGX parameter allocation");
}

std::wstring toWide(const std::string& value) {
    return std::filesystem::u8path(value).wstring();
}

// NGX writes logs and other runtime files to the application data path. Default
// to a per-user cache directory rather than the current working directory,
// which may be shared or attacker-writable.
std::filesystem::path defaultDataPath() {
    std::filesystem::path root;
#ifdef _WIN32
    if (const wchar_t* localAppData = _wgetenv(L"LOCALAPPDATA"); localAppData && *localAppData)
        root = localAppData;
#else
    if (const char* cache = std::getenv("XDG_CACHE_HOME"); cache && *cache)
        root = cache;
    else if (const char* home = std::getenv("HOME"); home && *home)
        root = std::filesystem::path(home) / ".cache";
#endif
    if (root.empty() || !root.is_absolute())
        throw std::runtime_error("Cannot determine a per-user cache directory for DLSS; pass applicationDataPath");
    const std::filesystem::path path = root / "warp_optix" / "ngx";
#ifdef _WIN32
    std::filesystem::create_directories(path);
#else
    std::filesystem::create_directories(path.parent_path());
    if (mkdir(path.c_str(), 0700) != 0 && errno != EEXIST)
        throw std::runtime_error("Cannot create DLSS data directory " + path.string());
    // Check and restrict the directory through one descriptor opened without following
    // symlinks, so it cannot be swapped between the ownership check and chmod.
    const int fd = open(path.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    struct stat info{};
    const bool owned = fd >= 0 && fstat(fd, &info) == 0 && S_ISDIR(info.st_mode) && info.st_uid == geteuid();
    const bool restricted = owned && fchmod(fd, 0700) == 0;
    if (fd >= 0)
        close(fd);
    if (!restricted)
        throw std::runtime_error("DLSS data directory " + path.string() +
                                 " must be a real directory owned by the current user");
#endif
    return path;
}

std::wstring dataPath(const std::string& applicationDataPath) {
    return applicationDataPath.empty() ? defaultDataPath().wstring() : toWide(applicationDataPath);
}

// Optional extra folder for a nvngx_dlssd library. By default NGX uses the
// driver's built-in runtime (R620+); an application-supplied copy takes precedence.
std::wstring featurePath(const std::string& featureSearchPath) {
    if (featureSearchPath.empty())
        return {};
    std::filesystem::path path = std::filesystem::u8path(featureSearchPath);
    if (!path.is_absolute())
        throw py::value_error("featureSearchPath must be an absolute path");
    return std::filesystem::weakly_canonical(path).wstring();
}

void NVSDK_CONV logCallback(const char* message, NVSDK_NGX_Logging_Level, NVSDK_NGX_Feature) {
    if (message)
        std::clog << "[NGX] " << message << std::endl;
}

enum DlssRRResource {
    RESOURCE_COLOR_IN = 0,
    RESOURCE_COLOR_OUT,
    RESOURCE_DIFFUSE_ALBEDO,
    RESOURCE_SPECULAR_ALBEDO,
    RESOURCE_NORMALROUGHNESS,
    RESOURCE_MOTIONVECTOR,
    RESOURCE_LINEARDEPTH,
    RESOURCE_SPECULAR_HITDISTANCE,
    RESOURCE_NUM
};

struct DlssRRContext {
    bool initialized = false;
    NVSDK_NGX_Parameter* ngxParams = nullptr;
    unsigned int activeDenoisers = 0;

    DlssRRContext() = default;
    DlssRRContext(const DlssRRContext&) = delete;
    DlssRRContext& operator=(const DlssRRContext&) = delete;
    ~DlssRRContext() {
        if (ngxParams)
            requireCore().DestroyParameters(ngxParams);
        if (initialized)
            releaseSdk();
    }

    // NGX CUDA state is process-wide: contexts share the first initialization
    // and the last context shuts it down.
    static unsigned int activeContexts;
    static void releaseSdk() {
        if (--activeContexts == 0)
            requireCore().Shutdown();
    }
};
unsigned int DlssRRContext::activeContexts = 0;

struct DlssRRDenoiser {
    NVSDK_NGX_Handle* handle = nullptr;
    NVSDK_NGX_Parameter* ngxParams = nullptr;
    DlssRRContext* context = nullptr;
    std::array<CUtexObject, RESOURCE_NUM> textureResources{};
    CUsurfObject outputSurface = 0;
    unsigned int inputWidth = 0;
    unsigned int inputHeight = 0;
    unsigned int outputWidth = 0;
    unsigned int outputHeight = 0;

    DlssRRDenoiser() = default;
    DlssRRDenoiser(const DlssRRDenoiser&) = delete;
    DlssRRDenoiser& operator=(const DlssRRDenoiser&) = delete;
    DlssRRDenoiser(DlssRRDenoiser&& other) noexcept
        : handle(std::exchange(other.handle, nullptr)), ngxParams(std::exchange(other.ngxParams, nullptr)),
          context(std::exchange(other.context, nullptr)), textureResources(other.textureResources),
          outputSurface(other.outputSurface), inputWidth(other.inputWidth), inputHeight(other.inputHeight),
          outputWidth(other.outputWidth), outputHeight(other.outputHeight) {}
    ~DlssRRDenoiser() {
        if (handle) {
            requireCore().ReleaseFeature(handle);
            if (context)
                --context->activeDenoisers;
        }
    }
};

struct DlssRRInitInfo {
    unsigned int inputWidth = 0;
    unsigned int inputHeight = 0;
    unsigned int outputWidth = 0;
    unsigned int outputHeight = 0;
    NVSDK_NGX_PerfQuality_Value quality = NVSDK_NGX_PerfQuality_Value_MaxQuality;
    NVSDK_NGX_RayReconstruction_Hint_Render_Preset preset = NVSDK_NGX_RayReconstruction_Hint_Render_Preset_F;
    bool mvJittered = false;
    bool lowResolutionMotionVectors = true;
    bool isContentHDR = true;
    bool depthInverted = false;
    bool autoExposure = false;
    bool useHWDepth = false;
};

struct DlssRRSupportedSizes {
    unsigned int minWidth = 0;
    unsigned int minHeight = 0;
    unsigned int maxWidth = 0;
    unsigned int maxHeight = 0;
    unsigned int optimalWidth = 0;
    unsigned int optimalHeight = 0;
};

NVSDK_NGX_Application_Identifier projectIdentifier(const std::string& projectId, const std::string& engineVersion) {
    NVSDK_NGX_Application_Identifier identifier = {};
    identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Project_Id;
    identifier.v.ProjectDesc.ProjectId = projectId.c_str();
    identifier.v.ProjectDesc.EngineType = NVSDK_NGX_ENGINE_TYPE_CUSTOM;
    identifier.v.ProjectDesc.EngineVersion = engineVersion.c_str();
    return identifier;
}

// Asks the driver whether Ray Reconstruction is usable on a CUDA device. Works
// before initialization, so callers can fall back cleanly on drivers older than
// R620 (no built-in runtime) or without NGX at all.
py::dict queryRequirements(int cudaDevice, const std::string& applicationDataPath, const std::string& projectId,
                           const std::string& engineVersion, const std::string& featureSearchPath) {
    const std::wstring data = dataPath(applicationDataPath);
    const std::wstring feature = featurePath(featureSearchPath);
    const wchar_t* featureValue = feature.c_str();
    NVSDK_NGX_FeatureCommonInfo featureInfo = {};
    featureInfo.PathListInfo.Path = feature.empty() ? nullptr : &featureValue;
    featureInfo.PathListInfo.Length = feature.empty() ? 0 : 1;

    NVSDK_NGX_FeatureDiscoveryInfo discovery = {};
    discovery.SDKVersion = NVSDK_NGX_Version_API;
    discovery.FeatureID = NVSDK_NGX_Feature_RayReconstruction;
    discovery.Identifier = projectIdentifier(projectId, engineVersion);
    discovery.ApplicationDataPath = data.c_str();
    discovery.FeatureInfo = &featureInfo;

    py::dict out;
    std::string error;
    const NgxCore* core = ngxCore(&error);
    if (!core) {
        out["supported"] = false;
        out["result_string"] = error;
        return out;
    }
    NVSDK_NGX_FeatureRequirement requirement = {};
    NVSDK_NGX_Result result;
    {
        py::gil_scoped_release release;
        result = core->GetFeatureRequirements(cudaDevice, &discovery, &requirement);
    }
    out["result"] = static_cast<unsigned int>(result);
    out["result_string"] = resultString(result);
    out["supported"] =
        NVSDK_NGX_SUCCEED(result) && requirement.FeatureSupported == NVSDK_NGX_FeatureSupportResult_Supported;
    out["feature_supported"] = static_cast<int>(requirement.FeatureSupported);
    out["min_hw_architecture"] = requirement.MinHWArchitecture;
    out["min_os_version"] = std::string(requirement.MinOSVersion);
    return out;
}

void contextInit(DlssRRContext& context, const std::string& applicationDataPath, const std::string& projectId,
                 const std::string& engineVersion, const std::string& featureSearchPath, bool enableLogging) {
    if (context.initialized)
        return;

    const std::wstring data = dataPath(applicationDataPath);
    const std::wstring feature = featurePath(featureSearchPath);
    const wchar_t* featureValue = feature.c_str();
    NVSDK_NGX_FeatureCommonInfo featureInfo = {};
    featureInfo.PathListInfo.Path = feature.empty() ? nullptr : &featureValue;
    featureInfo.PathListInfo.Length = feature.empty() ? 0 : 1;
    if (enableLogging) {
        featureInfo.LoggingInfo.LoggingCallback = logCallback;
        featureInfo.LoggingInfo.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_VERBOSE;
    }

    const bool startSdk = DlssRRContext::activeContexts == 0;
    if (startSdk)
        // The core's argument order differs from the SDK's NVSDK_NGX_CUDA_Init_with_ProjectID wrapper.
        check(requireCore().Init_ProjectID(projectId.c_str(), NVSDK_NGX_ENGINE_TYPE_CUSTOM, engineVersion.c_str(),
                                           data.c_str(), NVSDK_NGX_Version_API, &featureInfo),
              "NGX initialization");
    try {
        check(requireCore().GetCapabilityParameters(&context.ngxParams), "NGX capability query");
    } catch (...) {
        context.ngxParams = nullptr;
        if (startSdk)
            requireCore().Shutdown();
        throw;
    }
    context.initialized = true;
    ++DlssRRContext::activeContexts;
}

void contextDeinit(DlssRRContext& context) {
    if (context.activeDenoisers != 0)
        throw std::runtime_error("Deinitialize all DLSS RR denoisers before the context.");
    if (context.ngxParams) {
        requireCore().DestroyParameters(context.ngxParams);
        context.ngxParams = nullptr;
    }
    if (context.initialized) {
        DlssRRContext::releaseSdk();
        context.initialized = false;
    }
}

void requireInitialized(const DlssRRContext& context) {
    if (!context.ngxParams)
        throw std::runtime_error("DlssRRContext is not initialized.");
}

bool contextIsAvailable(const DlssRRContext& context) {
    requireInitialized(context);
    int needsUpdatedDriver = 0;
    int available = 0;
    int initResult = 0;
    check(context.ngxParams->Get(NVSDK_NGX_Parameter_SuperSamplingDenoising_NeedsUpdatedDriver, &needsUpdatedDriver),
          "NGX driver query");
    if (needsUpdatedDriver)
        return false;
    check(context.ngxParams->Get(NVSDK_NGX_Parameter_SuperSamplingDenoising_Available, &available),
          "NGX availability query");
    if (!available)
        return false;
    check(context.ngxParams->Get(NVSDK_NGX_Parameter_SuperSamplingDenoising_FeatureInitResult, &initResult),
          "NGX feature query");
    return NVSDK_NGX_SUCCEED(static_cast<NVSDK_NGX_Result>(initResult));
}

py::tuple contextMinDriverVersion(const DlssRRContext& context) {
    requireInitialized(context);
    unsigned int major = 0;
    unsigned int minor = 0;
    check(context.ngxParams->Get(NVSDK_NGX_Parameter_SuperSamplingDenoising_MinDriverVersionMajor, &major),
          "NGX driver version query");
    check(context.ngxParams->Get(NVSDK_NGX_Parameter_SuperSamplingDenoising_MinDriverVersionMinor, &minor),
          "NGX driver version query");
    return py::make_tuple(major, minor);
}

DlssRRSupportedSizes querySupportedSizes(DlssRRContext& context, unsigned int outputWidth, unsigned int outputHeight,
                                         NVSDK_NGX_PerfQuality_Value quality) {
    requireInitialized(context);
    if (quality == NVSDK_NGX_PerfQuality_Value_UltraQuality)
        throw std::runtime_error("UltraQuality is not supported for DLSS Ray Reconstruction.");
    // The driver publishes the query as a callback on the capability parameters.
    NVSDK_NGX_Parameter* params = context.ngxParams;
    void* callback = nullptr;
    params->Get(NVSDK_NGX_Parameter_DLSSDOptimalSettingsCallback, &callback);
    if (!callback)
        throw std::runtime_error("The installed DLSS RR runtime does not provide optimal settings (driver too old?)");
    params->Set(NVSDK_NGX_Parameter_Width, outputWidth);
    params->Set(NVSDK_NGX_Parameter_Height, outputHeight);
    params->Set(NVSDK_NGX_Parameter_PerfQualityValue, static_cast<int>(quality));
    params->Set(NVSDK_NGX_Parameter_RTXValue, 0); // Some older runtimes still read this.
    check(reinterpret_cast<PFN_NVSDK_NGX_DLSS_GetOptimalSettingsCallback>(callback)(params),
          "DLSS RR optimal settings query");

    DlssRRSupportedSizes sizes;
    params->Get(NVSDK_NGX_Parameter_OutWidth, &sizes.optimalWidth);
    params->Get(NVSDK_NGX_Parameter_OutHeight, &sizes.optimalHeight);
    // Older runtimes only report the optimal size; default the range to it.
    sizes.minWidth = sizes.maxWidth = sizes.optimalWidth;
    sizes.minHeight = sizes.maxHeight = sizes.optimalHeight;
    params->Get(NVSDK_NGX_Parameter_DLSS_Get_Dynamic_Max_Render_Width, &sizes.maxWidth);
    params->Get(NVSDK_NGX_Parameter_DLSS_Get_Dynamic_Max_Render_Height, &sizes.maxHeight);
    params->Get(NVSDK_NGX_Parameter_DLSS_Get_Dynamic_Min_Render_Width, &sizes.minWidth);
    params->Get(NVSDK_NGX_Parameter_DLSS_Get_Dynamic_Min_Render_Height, &sizes.minHeight);
    return sizes;
}

DlssRRDenoiser create(DlssRRContext& context, const DlssRRInitInfo& initInfo, uintptr_t stream) {
    requireInitialized(context);
    if (initInfo.inputWidth == 0 || initInfo.inputHeight == 0 || initInfo.outputWidth < initInfo.inputWidth ||
        initInfo.outputHeight < initInfo.inputHeight)
        throw py::value_error("DLSS RR needs a non-empty input no larger than the output");

    DlssRRDenoiser denoiser;
    denoiser.ngxParams = context.ngxParams;
    denoiser.inputWidth = initInfo.inputWidth;
    denoiser.inputHeight = initInfo.inputHeight;
    denoiser.outputWidth = initInfo.outputWidth;
    denoiser.outputHeight = initInfo.outputHeight;

    int createFlags = NVSDK_NGX_DLSS_Feature_Flags_None;
    createFlags |= initInfo.lowResolutionMotionVectors ? NVSDK_NGX_DLSS_Feature_Flags_MVLowRes : 0;
    createFlags |= initInfo.isContentHDR ? NVSDK_NGX_DLSS_Feature_Flags_IsHDR : 0;
    createFlags |= initInfo.depthInverted ? NVSDK_NGX_DLSS_Feature_Flags_DepthInverted : 0;
    createFlags |= initInfo.mvJittered ? NVSDK_NGX_DLSS_Feature_Flags_MVJittered : 0;
    createFlags |= initInfo.autoExposure ? NVSDK_NGX_DLSS_Feature_Flags_AutoExposure : 0;

    CUcontext cuContext = nullptr;
    if (cuCtxGetCurrent(&cuContext) != CUDA_SUCCESS || cuContext == nullptr)
        throw std::runtime_error("DLSS RR requires a current CUDA context.");

    Parameters params;
    // The CUDA context and stream the feature runs on.
    params->Set(NVSDK_NGX_Parameter_Input1, static_cast<void*>(cuContext));
    params->Set(NVSDK_NGX_Parameter_Input2, reinterpret_cast<void*>(stream));
    params->Set(NVSDK_NGX_Parameter_Width, initInfo.inputWidth);
    params->Set(NVSDK_NGX_Parameter_Height, initInfo.inputHeight);
    params->Set(NVSDK_NGX_Parameter_OutWidth, initInfo.outputWidth);
    params->Set(NVSDK_NGX_Parameter_OutHeight, initInfo.outputHeight);
    params->Set(NVSDK_NGX_Parameter_PerfQualityValue, static_cast<int>(initInfo.quality));
    params->Set(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags, createFlags);
    params->Set(NVSDK_NGX_Parameter_DLSS_Enable_Output_Subrects, 0);
    params->Set(NVSDK_NGX_Parameter_DLSS_Denoise_Mode, static_cast<int>(NVSDK_NGX_DLSS_Denoise_Mode_DLUnified));
    // Normals and roughness arrive packed in one texture (roughness in w).
    params->Set(NVSDK_NGX_Parameter_DLSS_Roughness_Mode,
                static_cast<unsigned int>(NVSDK_NGX_DLSS_Roughness_Mode_Packed));
    params->Set(NVSDK_NGX_Parameter_Use_HW_Depth,
                static_cast<unsigned int>(initInfo.useHWDepth ? NVSDK_NGX_DLSS_Depth_Type_HW
                                                              : NVSDK_NGX_DLSS_Depth_Type_Linear));
    for (auto name : {NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_DLAA,
                      NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Quality,
                      NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_UltraQuality,
                      NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Balanced,
                      NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Performance,
                      NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_UltraPerformance})
        params->Set(name, static_cast<unsigned int>(initInfo.preset));
    // Release internal resources at once instead of deferring; callers order the
    // release after the feature's stream work.
    params->Set(NVSDK_NGX_Parameter_FreeMemOnReleaseFeature, 1);
    check(requireCore().CreateFeature(NVSDK_NGX_Feature_RayReconstruction, params.params, &denoiser.handle),
          "DLSS RR feature creation");
    denoiser.context = &context;
    ++context.activeDenoisers;
    return denoiser;
}

void destroy(DlssRRDenoiser& denoiser) {
    if (!denoiser.handle)
        return;
    check(requireCore().ReleaseFeature(denoiser.handle), "DLSS RR release");
    denoiser.handle = nullptr;
    --denoiser.context->activeDenoisers;
    denoiser.context = nullptr;
    denoiser.ngxParams = nullptr;
}

void setResource(DlssRRDenoiser& denoiser, DlssRRResource resource, uint64_t handle) {
    if (resource < 0 || resource >= RESOURCE_NUM)
        throw std::invalid_argument("Invalid DLSS RR resource.");
    if (resource == RESOURCE_COLOR_OUT)
        denoiser.outputSurface = static_cast<CUsurfObject>(handle);
    else
        denoiser.textureResources[resource] = static_cast<CUtexObject>(handle);
}

void denoise(DlssRRDenoiser& denoiser, unsigned int renderWidth, unsigned int renderHeight, float jitterX,
             float jitterY, const std::vector<float>& worldToViewMatrix, const std::vector<float>& viewToClipMatrix,
             bool reset, int indicatorInvertXAxis, int indicatorInvertYAxis, float mvScaleX, float mvScaleY) {
    if (!denoiser.handle || !denoiser.ngxParams)
        throw std::runtime_error("DlssRRDenoiser is not initialized.");
    if (worldToViewMatrix.size() != 16 || viewToClipMatrix.size() != 16)
        throw py::value_error("Matrices must contain exactly 16 floats each.");
    const auto& tex = denoiser.textureResources;
    if (tex[RESOURCE_COLOR_IN] == 0 || denoiser.outputSurface == 0 || tex[RESOURCE_DIFFUSE_ALBEDO] == 0 ||
        tex[RESOURCE_SPECULAR_ALBEDO] == 0 || tex[RESOURCE_NORMALROUGHNESS] == 0 || tex[RESOURCE_LINEARDEPTH] == 0 ||
        tex[RESOURCE_MOTIONVECTOR] == 0)
        throw std::runtime_error("Missing required DLSS RR resources.");
    if (renderWidth == 0 || renderHeight == 0 || renderWidth > denoiser.inputWidth ||
        renderHeight > denoiser.inputHeight)
        throw py::value_error("Invalid render dimensions for DLSS RR.");

    // NGX reads CUDA resources through pointers to their texture/surface object handles.
    auto& textures = denoiser.textureResources;
    Parameters params;
    params->Set(NVSDK_NGX_Parameter_Color, static_cast<void*>(&textures[RESOURCE_COLOR_IN]));
    params->Set(NVSDK_NGX_Parameter_Output, static_cast<void*>(&denoiser.outputSurface));
    params->Set(NVSDK_NGX_Parameter_Depth, static_cast<void*>(&textures[RESOURCE_LINEARDEPTH]));
    params->Set(NVSDK_NGX_Parameter_MotionVectors, static_cast<void*>(&textures[RESOURCE_MOTIONVECTOR]));
    params->Set(NVSDK_NGX_Parameter_DiffuseAlbedo, static_cast<void*>(&textures[RESOURCE_DIFFUSE_ALBEDO]));
    params->Set(NVSDK_NGX_Parameter_SpecularAlbedo, static_cast<void*>(&textures[RESOURCE_SPECULAR_ALBEDO]));
    params->Set(NVSDK_NGX_Parameter_GBuffer_Normals, static_cast<void*>(&textures[RESOURCE_NORMALROUGHNESS]));
    params->Set(NVSDK_NGX_Parameter_GBuffer_Roughness, static_cast<void*>(&textures[RESOURCE_NORMALROUGHNESS]));
    if (textures[RESOURCE_SPECULAR_HITDISTANCE] != 0)
        params->Set(NVSDK_NGX_Parameter_DLSSD_SpecularHitDistance,
                    static_cast<void*>(&textures[RESOURCE_SPECULAR_HITDISTANCE]));
    params->Set(NVSDK_NGX_Parameter_DLSS_WORLD_TO_VIEW_MATRIX,
                static_cast<void*>(const_cast<float*>(worldToViewMatrix.data())));
    params->Set(NVSDK_NGX_Parameter_DLSS_VIEW_TO_CLIP_MATRIX,
                static_cast<void*>(const_cast<float*>(viewToClipMatrix.data())));
    params->Set(NVSDK_NGX_Parameter_Jitter_Offset_X, jitterX);
    params->Set(NVSDK_NGX_Parameter_Jitter_Offset_Y, jitterY);
    params->Set(NVSDK_NGX_Parameter_Reset, reset ? 1 : 0);
    params->Set(NVSDK_NGX_Parameter_MV_Scale_X, mvScaleX == 0.0f ? 1.0f : mvScaleX);
    params->Set(NVSDK_NGX_Parameter_MV_Scale_Y, mvScaleY == 0.0f ? 1.0f : mvScaleY);
    params->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width, renderWidth);
    params->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height, renderHeight);
    params->Set(NVSDK_NGX_Parameter_DLSS_Pre_Exposure, 1.0f);
    params->Set(NVSDK_NGX_Parameter_DLSS_Exposure_Scale, 1.0f);
    params->Set(NVSDK_NGX_Parameter_DLSS_Indicator_Invert_X_Axis, indicatorInvertXAxis);
    params->Set(NVSDK_NGX_Parameter_DLSS_Indicator_Invert_Y_Axis, indicatorInvertYAxis);
    check(requireCore().EvaluateFeature(denoiser.handle, params.params, nullptr), "DLSS RR evaluation");
}

} // namespace

void register_dlss_bindings(py::module_& m) {
    m.attr("DLSS_SDK_VERSION") = static_cast<unsigned int>(NVSDK_NGX_Version_API);
    m.attr("DLSS_DEFAULT_PROJECT_ID") = kDefaultProjectId;

    py::enum_<NVSDK_NGX_PerfQuality_Value>(m, "DlssPerfQuality", py::arithmetic())
        .value("MAX_PERF", NVSDK_NGX_PerfQuality_Value_MaxPerf)
        .value("BALANCED", NVSDK_NGX_PerfQuality_Value_Balanced)
        .value("MAX_QUALITY", NVSDK_NGX_PerfQuality_Value_MaxQuality)
        .value("ULTRA_PERFORMANCE", NVSDK_NGX_PerfQuality_Value_UltraPerformance)
        .value("ULTRA_QUALITY", NVSDK_NGX_PerfQuality_Value_UltraQuality)
        .value("DLAA", NVSDK_NGX_PerfQuality_Value_DLAA)
        .export_values();

    py::enum_<NVSDK_NGX_RayReconstruction_Hint_Render_Preset>(m, "RayReconstructionHintRenderPreset", py::arithmetic())
        .value("DEFAULT", NVSDK_NGX_RayReconstruction_Hint_Render_Preset_Default)
        .value("D", NVSDK_NGX_RayReconstruction_Hint_Render_Preset_D)
        .value("E", NVSDK_NGX_RayReconstruction_Hint_Render_Preset_E)
        .value("F", NVSDK_NGX_RayReconstruction_Hint_Render_Preset_F)
        .export_values();

    py::enum_<DlssRRResource>(m, "DlssRRResource", py::arithmetic())
        .value("RESOURCE_COLOR_IN", RESOURCE_COLOR_IN)
        .value("RESOURCE_COLOR_OUT", RESOURCE_COLOR_OUT)
        .value("RESOURCE_DIFFUSE_ALBEDO", RESOURCE_DIFFUSE_ALBEDO)
        .value("RESOURCE_SPECULAR_ALBEDO", RESOURCE_SPECULAR_ALBEDO)
        .value("RESOURCE_NORMALROUGHNESS", RESOURCE_NORMALROUGHNESS)
        .value("RESOURCE_MOTIONVECTOR", RESOURCE_MOTIONVECTOR)
        .value("RESOURCE_LINEARDEPTH", RESOURCE_LINEARDEPTH)
        .value("RESOURCE_SPECULAR_HITDISTANCE", RESOURCE_SPECULAR_HITDISTANCE)
        .export_values();

    m.def("dlssRRQueryRequirements", &queryRequirements, py::arg("cudaDevice") = 0, py::arg("applicationDataPath") = "",
          py::arg("projectId") = kDefaultProjectId, py::arg("engineVersion") = "1.0.0",
          py::arg("featureSearchPath") = "",
          "Ask the driver whether DLSS Ray Reconstruction is supported on a CUDA device. "
          "Works without initializing NGX; returns a dict with 'supported' and diagnostics.");

    py::class_<DlssRRContext>(m, "DlssRRContext")
        .def(py::init<>())
        .def("init", &contextInit, py::arg("applicationDataPath") = "", py::arg("projectId") = kDefaultProjectId,
             py::arg("engineVersion") = "1.0.0", py::arg("featureSearchPath") = "", py::arg("enableLogging") = false)
        .def("deinit", &contextDeinit)
        .def("isDlssRRAvailable", &contextIsAvailable)
        .def("getMinDriverVersion", &contextMinDriverVersion)
        .def("querySupportedDlssInputSizes", &querySupportedSizes, py::arg("outputWidth"), py::arg("outputHeight"),
             py::arg("quality") = NVSDK_NGX_PerfQuality_Value_MaxQuality)
        .def("initDlssRR", &create, py::arg("initInfo"), py::arg("stream") = 0u, py::keep_alive<0, 1>());

    py::class_<DlssRRDenoiser>(m, "DlssRRDenoiser")
        .def("setResource", &setResource)
        .def("denoise", &denoise, py::arg("renderWidth"), py::arg("renderHeight"), py::arg("jitterX"),
             py::arg("jitterY"), py::arg("worldToViewMatrix"), py::arg("viewToClipMatrix"), py::arg("reset") = false,
             py::arg("indicatorInvertXAxis") = 0, py::arg("indicatorInvertYAxis") = 0, py::arg("mvScaleX") = 1.0f,
             py::arg("mvScaleY") = 1.0f)
        .def("deinit", &destroy);

    py::class_<DlssRRInitInfo>(m, "DlssRRInitInfo")
        .def(py::init<>())
        .def_readwrite("inputWidth", &DlssRRInitInfo::inputWidth)
        .def_readwrite("inputHeight", &DlssRRInitInfo::inputHeight)
        .def_readwrite("outputWidth", &DlssRRInitInfo::outputWidth)
        .def_readwrite("outputHeight", &DlssRRInitInfo::outputHeight)
        .def_readwrite("quality", &DlssRRInitInfo::quality)
        .def_readwrite("preset", &DlssRRInitInfo::preset)
        .def_readwrite("mvJittered", &DlssRRInitInfo::mvJittered)
        .def_readwrite("lowResolutionMotionVectors", &DlssRRInitInfo::lowResolutionMotionVectors)
        .def_readwrite("isContentHDR", &DlssRRInitInfo::isContentHDR)
        .def_readwrite("depthInverted", &DlssRRInitInfo::depthInverted)
        .def_readwrite("autoExposure", &DlssRRInitInfo::autoExposure)
        .def_readwrite("useHWDepth", &DlssRRInitInfo::useHWDepth);

    py::class_<DlssRRSupportedSizes>(m, "DlssRRSupportedSizes")
        .def(py::init<>())
        .def_readwrite("minWidth", &DlssRRSupportedSizes::minWidth)
        .def_readwrite("minHeight", &DlssRRSupportedSizes::minHeight)
        .def_readwrite("maxWidth", &DlssRRSupportedSizes::maxWidth)
        .def_readwrite("maxHeight", &DlssRRSupportedSizes::maxHeight)
        .def_readwrite("optimalWidth", &DlssRRSupportedSizes::optimalWidth)
        .def_readwrite("optimalHeight", &DlssRRSupportedSizes::optimalHeight);
}

#endif // WARP_OPTIX_ENABLE_DLSS
