// Copyright (c) 2022 NVIDIA CORPORATION All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// Supplemental bindings only. Standard OptiX bindings come from pyoptix.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>
#define NOMINMAX
#include <optix.h>
static_assert(OPTIX_VERSION == 90100,
    "OptiX SDK version mismatch: the PyOptiX wrapper bridge requires OptiX 9.1.0");
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <climits>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef _WIN32
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#ifdef PYOPTIX_ENABLE_DLSS
#include <nvsdk_ngx.h>
#include <nvsdk_ngx_defs_dlssd.h>
#include <nvsdk_ngx_helpers_dlssd_cuda.h>
#include <nvsdk_ngx_params_dlssd.h>
#endif

namespace py = pybind11;
#define IF_OPTIX72(code) code
#define IF_OPTIX74(code) code
#define COMMA ,
#define PYOPTIX_CHECK(call) do { OptixResult result = (call); \
    if (result != OPTIX_SUCCESS) throw std::runtime_error(optixGetErrorString(result)); } while (0)
#ifdef PYOPTIX_ENABLE_DLSS
#define PYDLSS_CHECK(call) do { NVSDK_NGX_Result result = (call); \
    if (result != NVSDK_NGX_Result_Success) throw std::runtime_error( \
        "DLSS call failed: " + warp_optix_ext::dlssRRGetResultString(result)); } while (0)
#endif
#include "pyoptix_compat.h"

namespace {
py::tuple optix_version() {
    return py::make_tuple(OPTIX_VERSION / 10000,
                          (OPTIX_VERSION % 10000) / 100,
                          OPTIX_VERSION % 100);
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
size_t network_extent(const std::vector<OptixCoopVecMatrixDescription>& layers,
                      size_t stride, unsigned int numNetworks, const char* what) {
    size_t layerEnd = 0;
    for (const OptixCoopVecMatrixDescription& layer : layers)
        layerEnd = std::max(layerEnd, checked_add(layer.offsetInBytes, layer.sizeInBytes, what));
    return checked_add(checked_mul(numNetworks - 1, stride, what), layerEnd, what);
}

// Raw device pointers arrive from Python as integers, so verify that they lie in a
// live CUDA allocation that covers `requiredBytes` before OptiX reads or writes them.
void require_device_range(uintptr_t ptr, size_t requiredBytes, const char* name) {
    CUdeviceptr rangeStart = 0;
    size_t rangeSize = 0;
    const CUdeviceptr devicePtr = static_cast<CUdeviceptr>(ptr);
    if (cuPointerGetAttribute(&rangeStart, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, devicePtr) != CUDA_SUCCESS ||
        cuPointerGetAttribute(&rangeSize, CU_POINTER_ATTRIBUTE_RANGE_SIZE, devicePtr) != CUDA_SUCCESS)
        throw py::value_error(std::string(name) + " is not a CUDA device allocation");
    const size_t available = static_cast<size_t>(rangeStart + rangeSize - devicePtr);
    if (requiredBytes > available)
        throw py::value_error(std::string(name) + " requires " + std::to_string(requiredBytes) +
                              " bytes but its allocation has " + std::to_string(available) +
                              " bytes after the pointer");
}
}

namespace warp_optix_ext {
#ifdef PYOPTIX_ENABLE_DLSS
enum DlssRRResource
{
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

struct DlssRRContext
{
    bool initialized = false;
    NVSDK_NGX_Parameter* ngxParams = nullptr;
    unsigned int activeDenoisers = 0;
    unsigned int minDriverVersionMajor = 0;
    unsigned int minDriverVersionMinor = 0;

    DlssRRContext() = default;
    DlssRRContext( const DlssRRContext& ) = delete;
    DlssRRContext& operator=( const DlssRRContext& ) = delete;
    ~DlssRRContext()
    {
        if( ngxParams )
            NVSDK_NGX_CUDA_DestroyParameters( ngxParams );
        if( initialized )
            releaseSdk();
    }

    static unsigned int activeContexts;
    static void releaseSdk()
    {
        if( --activeContexts == 0 )
            NVSDK_NGX_CUDA_Shutdown();
    }
};
unsigned int DlssRRContext::activeContexts = 0;
bool operator==( const DlssRRContext& a, const DlssRRContext& b) { return a.ngxParams == b.ngxParams; }

struct DlssRRDenoiser
{
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
    DlssRRDenoiser( const DlssRRDenoiser& ) = delete;
    DlssRRDenoiser& operator=( const DlssRRDenoiser& ) = delete;
    DlssRRDenoiser( DlssRRDenoiser&& other ) noexcept
        : handle( std::exchange( other.handle, nullptr ) )
        , ngxParams( std::exchange( other.ngxParams, nullptr ) )
        , context( std::exchange( other.context, nullptr ) )
        , textureResources( std::move( other.textureResources ) )
        , outputSurface( other.outputSurface )
        , inputWidth( other.inputWidth )
        , inputHeight( other.inputHeight )
        , outputWidth( other.outputWidth )
        , outputHeight( other.outputHeight )
    {
    }
    ~DlssRRDenoiser()
    {
        if( handle )
        {
            NVSDK_NGX_CUDA_ReleaseFeature( handle );
            if( context )
                --context->activeDenoisers;
        }
    }
};
bool operator==( const DlssRRDenoiser& a, const DlssRRDenoiser& b) { return a.handle == b.handle; }

struct DlssRRInitInfo
{
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

struct DlssRRSupportedSizes
{
    unsigned int minWidth = 0;
    unsigned int minHeight = 0;
    unsigned int maxWidth = 0;
    unsigned int maxHeight = 0;
    unsigned int optimalWidth = 0;
    unsigned int optimalHeight = 0;
};

std::string dlssRRGetResultString( int ngxResultCode )
{
    const wchar_t* text = GetNGXResultAsString( static_cast<NVSDK_NGX_Result>( ngxResultCode ) );
    return text ? std::filesystem::path( text ).u8string() : "NGX result " + std::to_string( ngxResultCode );
}
#endif

size_t coopVecMatrixComputeSize(
    pyoptix::DeviceContext context,
    unsigned int N,
    unsigned int K,
    OptixCoopVecElemType elementType,
    OptixCoopVecMatrixLayout layout,
    size_t rowColumnStrideInBytes
    )
{
    ensure_optix_initialized();
    if( N == 0 || K == 0 )
        throw py::value_error( "Cooperative-vector matrix dimensions must be non-zero" );

    size_t sizeInBytes = 0;
    PYOPTIX_CHECK(
        optixCoopVecMatrixComputeSize(
            context.deviceContext,
            N,
            K,
            elementType,
            layout,
            rowColumnStrideInBytes,
            &sizeInBytes
        )
    );
    return sizeInBytes;
}

void coopVecMatrixConvert(
    pyoptix::DeviceContext context,
    uintptr_t stream,
    const std::vector<OptixCoopVecMatrixDescription>& inputLayers,
    uintptr_t inputNetworks,
    const std::vector<OptixCoopVecMatrixDescription>& outputLayers,
    uintptr_t outputNetworks,
    size_t inputNetworkStrideInBytes,
    size_t outputNetworkStrideInBytes,
    unsigned int numNetworks
    )
{
    ensure_optix_initialized();
    if( numNetworks == 0 )
        throw py::value_error( "numNetworks must be non-zero" );
    if( inputLayers.empty() )
        throw py::value_error( "inputLayers must contain at least one matrix description" );
    if( inputLayers.size() != outputLayers.size() )
        throw py::value_error( "inputLayers and outputLayers must have the same length" );
    if( inputNetworks == 0 || outputNetworks == 0 )
        throw py::value_error( "inputNetworks and outputNetworks must be non-zero CUDA device pointers" );
    if( numNetworks > 1 && ( inputNetworkStrideInBytes == 0 || outputNetworkStrideInBytes == 0 ) )
        throw py::value_error( "network strides must be non-zero when numNetworks is greater than one" );
    if( inputNetworkStrideInBytes % 64 != 0 || outputNetworkStrideInBytes % 64 != 0 )
        throw py::value_error( "cooperative-vector network strides must be multiples of 64 bytes" );
    if( inputLayers.size() > UINT_MAX )
        throw py::value_error( "too many cooperative-vector matrix descriptions" );

    for( size_t i = 0; i < inputLayers.size(); ++i )
    {
        const OptixCoopVecMatrixDescription& input = inputLayers[i];
        const OptixCoopVecMatrixDescription& output = outputLayers[i];
        if( input.N == 0 || input.K == 0 || output.N == 0 || output.K == 0 )
            throw py::value_error( "cooperative-vector matrix dimensions must be non-zero" );
        if( input.N != output.N || input.K != output.K )
            throw py::value_error( "input and output matrix dimensions must match" );
        if( input.sizeInBytes == 0 || output.sizeInBytes == 0 )
            throw py::value_error( "cooperative-vector matrix sizes must be non-zero" );
        if( ( inputNetworks % 64 + input.offsetInBytes % 64 ) % 64 != 0 )
            throw py::value_error( "each input matrix address must be 64-byte aligned" );
        if( ( outputNetworks % 64 + output.offsetInBytes % 64 ) % 64 != 0 )
            throw py::value_error( "each output matrix address must be 64-byte aligned" );
    }

    require_device_range( inputNetworks,
        network_extent( inputLayers, inputNetworkStrideInBytes, numNetworks, "input network extent" ),
        "inputNetworks" );
    require_device_range( outputNetworks,
        network_extent( outputLayers, outputNetworkStrideInBytes, numNetworks, "output network extent" ),
        "outputNetworks" );

    OptixNetworkDescription inputDescription{
        const_cast<OptixCoopVecMatrixDescription*>( inputLayers.data() ),
        static_cast<unsigned int>( inputLayers.size() )
    };
    OptixNetworkDescription outputDescription{
        const_cast<OptixCoopVecMatrixDescription*>( outputLayers.data() ),
        static_cast<unsigned int>( outputLayers.size() )
    };

    OptixResult result;
    {
        py::gil_scoped_release release;
        result = optixCoopVecMatrixConvert(
            context.deviceContext,
            reinterpret_cast<CUstream>( stream ),
            numNetworks,
            &inputDescription,
            static_cast<CUdeviceptr>( inputNetworks ),
            inputNetworkStrideInBytes,
            &outputDescription,
            static_cast<CUdeviceptr>( outputNetworks ),
            outputNetworkStrideInBytes
        );
    }
    if( result != OPTIX_SUCCESS )
        throw std::runtime_error( optixGetErrorString( result ) );
}

#ifdef PYOPTIX_ENABLE_DLSS
static std::wstring toWideString( const std::string& value )
{
    return std::filesystem::u8path( value ).wstring();
}

// NGX writes logs and other runtime files to the application data path. Default
// to a per-user cache directory rather than the current working directory,
// which may be shared or attacker-writable.
static std::filesystem::path defaultNgxDataPath()
{
    std::filesystem::path root;
#ifdef _WIN32
    if( const wchar_t* localAppData = _wgetenv( L"LOCALAPPDATA" ); localAppData && *localAppData )
        root = localAppData;
#else
    if( const char* cache = std::getenv( "XDG_CACHE_HOME" ); cache && *cache )
        root = cache;
    else if( const char* home = std::getenv( "HOME" ); home && *home )
        root = std::filesystem::path( home ) / ".cache";
#endif
    if( root.empty() || !root.is_absolute() )
        throw std::runtime_error( "Cannot determine a per-user cache directory for DLSS; pass applicationDataPath" );
    const std::filesystem::path path = root / "warp_optix" / "ngx";
#ifdef _WIN32
    std::filesystem::create_directories( path );
#else
    std::filesystem::create_directories( path.parent_path() );
    if( mkdir( path.c_str(), 0700 ) != 0 && errno != EEXIST )
        throw std::runtime_error( "Cannot create DLSS data directory " + path.string() );
    // Check and restrict the directory through one descriptor opened without following
    // symlinks, so it cannot be swapped between the ownership check and chmod.
    const int fd = open( path.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC );
    struct stat info {};
    const bool owned = fd >= 0 && fstat( fd, &info ) == 0 && S_ISDIR( info.st_mode ) && info.st_uid == geteuid();
    const bool restricted = owned && fchmod( fd, 0700 ) == 0;
    if( fd >= 0 )
        close( fd );
    if( !restricted )
        throw std::runtime_error( "DLSS data directory " + path.string() +
                                  " must be a real directory owned by the current user" );
#endif
    return path;
}

// NGX loads the DLSS feature library from this path, so only accept an absolute
// location and default to the directory of this extension, where the wheel
// installs the DLSS runtime.
static std::filesystem::path dlssFeatureSearchPath( const std::string& featureSearchPath )
{
    std::filesystem::path path = featureSearchPath.empty()
        ? std::filesystem::u8path( py::module_::import( "warp_optix._ext" ).attr( "__file__" ).cast<std::string>() ).parent_path()
        : std::filesystem::u8path( featureSearchPath );
    if( !path.is_absolute() )
        throw py::value_error( "featureSearchPath must be an absolute path" );
    return std::filesystem::weakly_canonical( path );
}

// Forwards NGX messages to the Python logger "warp_optix.dlss"; NGX may call
// this from its own threads.
static void NVSDK_CONV dlssLogCallback( const char* message, NVSDK_NGX_Logging_Level level, NVSDK_NGX_Feature )
{
    if( !message || !Py_IsInitialized() )
        return;
    std::string text( message );
    while( !text.empty() && ( text.back() == '\n' || text.back() == '\r' ) )
        text.pop_back();
    py::gil_scoped_acquire gil;
    try
    {
        const int pythonLevel = level == NVSDK_NGX_LOGGING_LEVEL_VERBOSE ? 10 /* DEBUG */ : 20 /* INFO */;
        // NGX messages are not guaranteed to be valid UTF-8.
        py::object decoded = py::reinterpret_steal<py::object>( PyUnicode_DecodeUTF8( text.data(), text.size(), "replace" ) );
        py::module_::import( "logging" ).attr( "getLogger" )( "warp_optix.dlss" ).attr( "log" )( pythonLevel, decoded );
    }
    catch( py::error_already_set& error )
    {
        error.discard_as_unraisable( "warp_optix.dlss log callback" );
    }
}

void dlssRRContextInit(
    warp_optix_ext::DlssRRContext& context,
    const std::string& projectId,
    const std::string& applicationDataPath,
    const std::string& engineVersion,
    const std::string& featureSearchPath,
    bool enableLogging
)
{
    if( context.initialized )
        return;

    const std::wstring appPath = applicationDataPath.empty() ? defaultNgxDataPath().wstring() : toWideString( applicationDataPath );
    const std::wstring featurePath = dlssFeatureSearchPath( featureSearchPath ).wstring();
    const wchar_t* featurePathValue = featurePath.c_str();
    NVSDK_NGX_FeatureCommonInfo featureInfo = {};
    featureInfo.PathListInfo.Path = &featurePathValue;
    featureInfo.PathListInfo.Length = 1;
    if( enableLogging )
    {
        featureInfo.LoggingInfo.LoggingCallback = dlssLogCallback;
        featureInfo.LoggingInfo.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_VERBOSE;
    }
    // NGX CUDA state is process-wide. Contexts share the first initialization
    // and the last context shuts it down.
    const bool startSdk = warp_optix_ext::DlssRRContext::activeContexts == 0;
    if( startSdk )
    {
        PYDLSS_CHECK(
            NVSDK_NGX_CUDA_Init_with_ProjectID(
                projectId.c_str(),
                NVSDK_NGX_ENGINE_TYPE_CUSTOM,
                engineVersion.c_str(),
                appPath.c_str(),
                &featureInfo
            )
        );
    }
    try
    {
        PYDLSS_CHECK( NVSDK_NGX_CUDA_GetCapabilityParameters( &context.ngxParams ) );
    }
    catch( ... )
    {
        context.ngxParams = nullptr;
        if( startSdk )
            NVSDK_NGX_CUDA_Shutdown();
        throw;
    }
    context.initialized = true;
    ++warp_optix_ext::DlssRRContext::activeContexts;
}

void dlssRRContextDeinit( warp_optix_ext::DlssRRContext& context )
{
    if( context.activeDenoisers != 0 )
        throw std::runtime_error( "Deinitialize all DLSS RR denoisers before the context." );
    if( context.ngxParams )
    {
        NVSDK_NGX_Parameter* params = std::exchange( context.ngxParams, nullptr );
        PYDLSS_CHECK( NVSDK_NGX_CUDA_DestroyParameters( params ) );
    }
    if( context.initialized )
    {
        warp_optix_ext::DlssRRContext::releaseSdk();
        context.initialized = false;
    }
}

bool dlssRRContextIsAvailable( const warp_optix_ext::DlssRRContext& context )
{
    if( !context.ngxParams )
        throw std::runtime_error( "DlssRRContext is not initialized." );

    int available = 0;
    int initResult = 0;
    int needsUpdatedDriver = 0;
    PYDLSS_CHECK( context.ngxParams->Get( NVSDK_NGX_Parameter_SuperSamplingDenoising_NeedsUpdatedDriver, &needsUpdatedDriver ) );
    if( needsUpdatedDriver )
        return false;

    PYDLSS_CHECK( context.ngxParams->Get( NVSDK_NGX_Parameter_SuperSamplingDenoising_Available, &available ) );
    if( !available )
        return false;

    PYDLSS_CHECK( context.ngxParams->Get( NVSDK_NGX_Parameter_SuperSamplingDenoising_FeatureInitResult, &initResult ) );
    return NVSDK_NGX_SUCCEED( static_cast<NVSDK_NGX_Result>( initResult ) );
}

py::tuple dlssRRContextGetMinDriverVersion( const warp_optix_ext::DlssRRContext& context )
{
    if( !context.ngxParams )
        throw std::runtime_error( "DlssRRContext is not initialized." );
    unsigned int major = 0;
    unsigned int minor = 0;
    PYDLSS_CHECK( context.ngxParams->Get( NVSDK_NGX_Parameter_SuperSamplingDenoising_MinDriverVersionMajor, &major ) );
    PYDLSS_CHECK( context.ngxParams->Get( NVSDK_NGX_Parameter_SuperSamplingDenoising_MinDriverVersionMinor, &minor ) );
    return py::make_tuple( major, minor );
}

warp_optix_ext::DlssRRSupportedSizes dlssRRQuerySupportedDlssInputSizes(
    warp_optix_ext::DlssRRContext& context,
    unsigned int outputWidth,
    unsigned int outputHeight,
    NVSDK_NGX_PerfQuality_Value quality
)
{
    if( !context.ngxParams )
        throw std::runtime_error( "DlssRRContext is not initialized." );
    if( quality == NVSDK_NGX_PerfQuality_Value_UltraQuality )
        throw std::runtime_error( "UltraQuality is not supported for DLSS Ray Reconstruction query." );

    warp_optix_ext::DlssRRSupportedSizes sizes;
    float sharpness = 0.0f;
    PYDLSS_CHECK(
        NGX_DLSSD_GET_OPTIMAL_SETTINGS(
            context.ngxParams,
            outputWidth,
            outputHeight,
            quality,
            &sizes.optimalWidth,
            &sizes.optimalHeight,
            &sizes.maxWidth,
            &sizes.maxHeight,
            &sizes.minWidth,
            &sizes.minHeight,
            &sharpness
        )
    );
    return sizes;
}

warp_optix_ext::DlssRRDenoiser dlssRRCreate(
    warp_optix_ext::DlssRRContext& context,
    const warp_optix_ext::DlssRRInitInfo& initInfo,
    uintptr_t stream
)
{
    if( !context.ngxParams )
        throw std::runtime_error( "DlssRRContext is not initialized." );

    warp_optix_ext::DlssRRDenoiser denoiser;
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

    NVSDK_NGX_DLSSD_Create_Params dlssParams = {};
    dlssParams.InDenoiseMode = NVSDK_NGX_DLSS_Denoise_Mode_DLUnified;
    // DlssRRCpp exposes a combined NORMALROUGHNESS resource.
    dlssParams.InRoughnessMode = NVSDK_NGX_DLSS_Roughness_Mode_Packed;
    dlssParams.InUseHWDepth = initInfo.useHWDepth ? NVSDK_NGX_DLSS_Depth_Type_HW : NVSDK_NGX_DLSS_Depth_Type_Linear;
    dlssParams.InWidth = initInfo.inputWidth;
    dlssParams.InHeight = initInfo.inputHeight;
    dlssParams.InTargetWidth = initInfo.outputWidth;
    dlssParams.InTargetHeight = initInfo.outputHeight;
    dlssParams.InPerfQualityValue = initInfo.quality;
    dlssParams.InFeatureCreateFlags = createFlags;
    dlssParams.InEnableOutputSubrects = false;

    context.ngxParams->Set( NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_DLAA, initInfo.preset );
    context.ngxParams->Set( NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Quality, initInfo.preset );
    context.ngxParams->Set( NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_UltraQuality, initInfo.preset );
    context.ngxParams->Set( NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Balanced, initInfo.preset );
    context.ngxParams->Set( NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Performance, initInfo.preset );
    context.ngxParams->Set( NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_UltraPerformance, initInfo.preset );

    CUcontext cuContext = nullptr;
    if( cuCtxGetCurrent( &cuContext ) != CUDA_SUCCESS || cuContext == nullptr )
        throw std::runtime_error( "DLSS RR requires a current CUDA context." );

    NVSDK_NGX_CUDA_DLSSD_Create_Params cudaParams = {};
    cudaParams.Feature = dlssParams;
    cudaParams.InCUContext = cuContext;
    cudaParams.InCUStream = reinterpret_cast<CUstream>( stream );
    {
        // Feature creation loads the model and can take hundreds of milliseconds.
        py::gil_scoped_release release;
        PYDLSS_CHECK( NGX_CUDA_CREATE_DLSSD_EXT( &denoiser.handle, context.ngxParams, &cudaParams ) );
    }
    denoiser.context = &context;
    ++context.activeDenoisers;
    return denoiser;
}

void dlssRRDestroy( warp_optix_ext::DlssRRDenoiser& denoiser )
{
    if( denoiser.handle )
    {
        PYDLSS_CHECK( NVSDK_NGX_CUDA_ReleaseFeature( denoiser.handle ) );
        denoiser.handle = nullptr;
        --denoiser.context->activeDenoisers;
        denoiser.context = nullptr;
        denoiser.ngxParams = nullptr;
    }
}

void dlssRRSetResource( warp_optix_ext::DlssRRDenoiser& denoiser, warp_optix_ext::DlssRRResource resourceId, uint64_t resourceHandle )
{
    if( resourceId < 0 || resourceId >= warp_optix_ext::RESOURCE_NUM )
        throw std::invalid_argument( "Invalid DLSS RR resource." );
    if( resourceId == warp_optix_ext::RESOURCE_COLOR_OUT )
        denoiser.outputSurface = static_cast<CUsurfObject>( resourceHandle );
    else
        denoiser.textureResources[resourceId] = static_cast<CUtexObject>( resourceHandle );
}

void dlssRRResetResource( warp_optix_ext::DlssRRDenoiser& denoiser, warp_optix_ext::DlssRRResource resourceId )
{
    if( resourceId < 0 || resourceId >= warp_optix_ext::RESOURCE_NUM )
        throw std::invalid_argument( "Invalid DLSS RR resource." );
    if( resourceId == warp_optix_ext::RESOURCE_COLOR_OUT )
        denoiser.outputSurface = 0;
    else
        denoiser.textureResources[resourceId] = 0;
}

void dlssRRDenoise(
    warp_optix_ext::DlssRRDenoiser& denoiser,
    unsigned int renderWidth,
    unsigned int renderHeight,
    float jitterX,
    float jitterY,
    const std::vector<float>& worldToViewMatrix,
    const std::vector<float>& viewToClipMatrix,
    bool reset,
    int indicatorInvertXAxis,
    int indicatorInvertYAxis,
    float mvScaleX,
    float mvScaleY
)
{
    if( !denoiser.handle || !denoiser.ngxParams )
        throw std::runtime_error( "DlssRRDenoiser is not initialized." );
    if( worldToViewMatrix.size() != 16 || viewToClipMatrix.size() != 16 )
        throw std::runtime_error( "Matrices must contain exactly 16 floats each." );
    if( denoiser.textureResources[warp_optix_ext::RESOURCE_COLOR_IN] == 0 || denoiser.outputSurface == 0
        || denoiser.textureResources[warp_optix_ext::RESOURCE_DIFFUSE_ALBEDO] == 0
        || denoiser.textureResources[warp_optix_ext::RESOURCE_SPECULAR_ALBEDO] == 0
        || denoiser.textureResources[warp_optix_ext::RESOURCE_NORMALROUGHNESS] == 0
        || denoiser.textureResources[warp_optix_ext::RESOURCE_LINEARDEPTH] == 0
        || denoiser.textureResources[warp_optix_ext::RESOURCE_MOTIONVECTOR] == 0 )
    {
        throw std::runtime_error( "Missing required DLSS RR resources." );
    }
    if( renderWidth == 0 || renderHeight == 0 || renderWidth > denoiser.inputWidth || renderHeight > denoiser.inputHeight )
        throw std::runtime_error( "Invalid render dimensions for DLSS RR." );

    NVSDK_NGX_CUDA_DLSSD_Eval_Params evalParams = {};
    evalParams.InReset = reset ? 1 : 0;
    evalParams.pInColor = &denoiser.textureResources[warp_optix_ext::RESOURCE_COLOR_IN];
    evalParams.pInOutput = &denoiser.outputSurface;
    evalParams.pInDiffuseAlbedo = &denoiser.textureResources[warp_optix_ext::RESOURCE_DIFFUSE_ALBEDO];
    evalParams.pInSpecularAlbedo = &denoiser.textureResources[warp_optix_ext::RESOURCE_SPECULAR_ALBEDO];
    evalParams.pInNormals = &denoiser.textureResources[warp_optix_ext::RESOURCE_NORMALROUGHNESS];
    evalParams.pInRoughness = &denoiser.textureResources[warp_optix_ext::RESOURCE_NORMALROUGHNESS];
    evalParams.pInDepth = &denoiser.textureResources[warp_optix_ext::RESOURCE_LINEARDEPTH];
    evalParams.pInMotionVectors = &denoiser.textureResources[warp_optix_ext::RESOURCE_MOTIONVECTOR];
    evalParams.pInSpecularHitDistance = &denoiser.textureResources[warp_optix_ext::RESOURCE_SPECULAR_HITDISTANCE];
    evalParams.pInWorldToViewMatrix = const_cast<float*>( worldToViewMatrix.data() );
    evalParams.pInViewToClipMatrix = const_cast<float*>( viewToClipMatrix.data() );
    evalParams.InJitterOffsetX = jitterX;
    evalParams.InJitterOffsetY = jitterY;
    evalParams.InMVScaleX = mvScaleX;
    evalParams.InMVScaleY = mvScaleY;
    evalParams.InIndicatorInvertXAxis = indicatorInvertXAxis;
    evalParams.InIndicatorInvertYAxis = indicatorInvertYAxis;
    evalParams.InPreExposure = 1.0f;
    evalParams.InExposureScale = 1.0f;
    evalParams.InRenderSubrectDimensions = { renderWidth, renderHeight };

    PYDLSS_CHECK( NGX_CUDA_EVALUATE_DLSSD_EXT( denoiser.handle, denoiser.ngxParams, &evalParams ) );
}
#endif

} // namespace warp_optix_ext

namespace {

template <typename T>
void require_upstream_type(py::handle cls) {
    const auto* registered = py::detail::get_type_info(typeid(T), false);
    const auto* actual = py::detail::get_type_info(reinterpret_cast<PyTypeObject*>(cls.ptr()));
    if (!registered || registered != actual || registered->type_size != sizeof(T)) {
        throw py::import_error(
            "warp-optix requires ABI-compatible pyoptix 9.1.0 bindings. "
            "Rebuild pyoptix and warp-optix using the same pybind11 major version and C++ ABI.");
    }
}

template <typename T, typename Field>
void add_field(py::handle cls, const char* name, Field T::* member) {
    if (py::hasattr(cls, name)) return;
    cls.attr(name) = py::module_::import("builtins").attr("property")(
        py::cpp_function([member](const T& self) { return self.*member; }, py::is_method(cls)),
        py::cpp_function([member](T& self, const Field& value) { self.*member = value; }, py::is_method(cls))
    );
}

void sync_module_options(pyoptix::ModuleCompileOptions& options) {
    // Upstream 9.1.0's sync() returns before forwarding this metadata.
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
        throw py::import_error(
            "OptiX SDK version mismatch: pyoptix was compiled with "
            + py::str(upstream_version).cast<std::string>()
            + ", but warp-optix was compiled with "
            + py::str(optix_version()).cast<std::string>()
            + ". Rebuild both packages with matching OptiX headers.");
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
    context.attr("coopVecMatrixComputeSize") = py::cpp_function(
        &warp_optix_ext::coopVecMatrixComputeSize, py::is_method(context),
        py::arg("N"), py::arg("K"), py::arg("elementType"), py::arg("layout"),
        py::arg("rowColumnStrideInBytes") = 0u);
    context.attr("coopVecMatrixConvert") = py::cpp_function(
        &warp_optix_ext::coopVecMatrixConvert, py::is_method(context),
        py::arg("stream"), py::arg("inputLayers"), py::arg("inputNetworks"),
        py::arg("outputLayers"), py::arg("outputNetworks"),
        py::arg("inputNetworkStrideInBytes") = 0u,
        py::arg("outputNetworkStrideInBytes") = 0u, py::arg("numNetworks") = 1u);

    auto guides = upstream.attr("DenoiserGuideLayer");
    add_field(guides, "previousOutputInternalGuideLayer", &OptixDenoiserGuideLayer::previousOutputInternalGuideLayer);
    add_field(guides, "outputInternalGuideLayer", &OptixDenoiserGuideLayer::outputInternalGuideLayer);
    add_field(upstream.attr("DenoiserParams"), "temporalModeUsePreviousLayers", &OptixDenoiserParams::temporalModeUsePreviousLayers);
    auto sizes = upstream.attr("DenoiserSizes");
    add_field(sizes, "computeIntensitySizeInBytes", &OptixDenoiserSizes::computeIntensitySizeInBytes);
    add_field(sizes, "computeAverageColorSizeInBytes", &OptixDenoiserSizes::computeAverageColorSizeInBytes);
    add_field(sizes, "internalGuideLayerPixelSizeInBytes", &OptixDenoiserSizes::internalGuideLayerPixelSizeInBytes);
    upstream.attr("_warp_optix_native_extensions_installed") = true;
}

} // namespace

PYBIND11_MODULE(_ext, m) {
    m.doc() = "Additional OptiX and DLSS bindings for the pyoptix dependency";
    m.def("optix_version", &optix_version,
          "OptiX SDK version used to compile this supplemental extension.");
    m.def("install_pyoptix_extensions", &install_pyoptix_extensions);
    m.def("sync_module_options", &sync_module_options);
    m.def("coop_vec_property", [](const pyoptix::DeviceContext& context) {
        ensure_optix_initialized();
        uint32_t flags = 0;
        PYOPTIX_CHECK(optixDeviceContextGetProperty(context.deviceContext,
            OPTIX_DEVICE_PROPERTY_COOP_VEC, &flags, sizeof(flags)));
        return flags;
    });
    // Our private OptiX dispatch table is initialized lazily: importing the
    // package must also work on machines without an NVIDIA driver.
    py::enum_<OptixDevicePropertyCoopVecFlags>(m, "DevicePropertyCoopVecFlags", py::arithmetic())
        .value( "DEVICE_PROPERTY_COOP_VEC_FLAG_NONE", OPTIX_DEVICE_PROPERTY_COOP_VEC_FLAG_NONE )
        .value( "DEVICE_PROPERTY_COOP_VEC_FLAG_STANDARD", OPTIX_DEVICE_PROPERTY_COOP_VEC_FLAG_STANDARD )
        .export_values();

    py::enum_<OptixCoopVecElemType>(m, "CoopVecElemType", py::arithmetic())
        .value( "COOP_VEC_ELEM_TYPE_UNKNOWN", OPTIX_COOP_VEC_ELEM_TYPE_UNKNOWN )
        .value( "COOP_VEC_ELEM_TYPE_FLOAT16", OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16 )
        .value( "COOP_VEC_ELEM_TYPE_FLOAT32", OPTIX_COOP_VEC_ELEM_TYPE_FLOAT32 )
        .value( "COOP_VEC_ELEM_TYPE_UINT8", OPTIX_COOP_VEC_ELEM_TYPE_UINT8 )
        .value( "COOP_VEC_ELEM_TYPE_INT8", OPTIX_COOP_VEC_ELEM_TYPE_INT8 )
        .value( "COOP_VEC_ELEM_TYPE_UINT32", OPTIX_COOP_VEC_ELEM_TYPE_UINT32 )
        .value( "COOP_VEC_ELEM_TYPE_INT32", OPTIX_COOP_VEC_ELEM_TYPE_INT32 )
        .value( "COOP_VEC_ELEM_TYPE_FLOAT8_E4M3", OPTIX_COOP_VEC_ELEM_TYPE_FLOAT8_E4M3 )
        .value( "COOP_VEC_ELEM_TYPE_FLOAT8_E5M2", OPTIX_COOP_VEC_ELEM_TYPE_FLOAT8_E5M2 )
        .export_values();

    py::enum_<OptixCoopVecMatrixLayout>(m, "CoopVecMatrixLayout", py::arithmetic())
        .value( "COOP_VEC_MATRIX_LAYOUT_ROW_MAJOR", OPTIX_COOP_VEC_MATRIX_LAYOUT_ROW_MAJOR )
        .value( "COOP_VEC_MATRIX_LAYOUT_COLUMN_MAJOR", OPTIX_COOP_VEC_MATRIX_LAYOUT_COLUMN_MAJOR )
        .value( "COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL", OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL )
        .value( "COOP_VEC_MATRIX_LAYOUT_TRAINING_OPTIMAL", OPTIX_COOP_VEC_MATRIX_LAYOUT_TRAINING_OPTIMAL )
        .export_values();

    py::class_<OptixCoopVecMatrixDescription>(m, "CoopVecMatrixDescription")
        .def( py::init([]() { return std::unique_ptr<OptixCoopVecMatrixDescription>(new OptixCoopVecMatrixDescription{}); } ) )
        .def_readwrite( "N", &OptixCoopVecMatrixDescription::N )
        .def_readwrite( "K", &OptixCoopVecMatrixDescription::K )
        .def_readwrite( "offsetInBytes", &OptixCoopVecMatrixDescription::offsetInBytes )
        .def_readwrite( "elementType", &OptixCoopVecMatrixDescription::elementType )
        .def_readwrite( "layout", &OptixCoopVecMatrixDescription::layout )
        .def_readwrite( "rowColumnStrideInBytes", &OptixCoopVecMatrixDescription::rowColumnStrideInBytes )
        .def_readwrite( "sizeInBytes", &OptixCoopVecMatrixDescription::sizeInBytes )
        ;

    m.attr("DEVICE_PROPERTY_COOP_VEC") = py::int_(static_cast<unsigned int>(OPTIX_DEVICE_PROPERTY_COOP_VEC));
    m.attr("PRIMITIVE_TYPE_ROUND_CATMULLROM") = py::int_(static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM));
    m.attr("PRIMITIVE_TYPE_FLAT_QUADRATIC_BSPLINE") = py::int_(static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_FLAT_QUADRATIC_BSPLINE));
    m.attr("PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER") = py::int_(static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER));
    m.attr("PRIMITIVE_TYPE_FLAGS_ROUND_CATMULLROM") = py::int_(static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CATMULLROM));
    m.attr("PRIMITIVE_TYPE_FLAGS_FLAT_QUADRATIC_BSPLINE") = py::int_(static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_FLAGS_FLAT_QUADRATIC_BSPLINE));
    m.attr("PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BEZIER") = py::int_(static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BEZIER));
    m.attr("PIXEL_FORMAT_FLOAT2") = py::int_(static_cast<unsigned int>(OPTIX_PIXEL_FORMAT_FLOAT2));
    m.attr("PIXEL_FORMAT_INTERNAL_GUIDE_LAYER") = py::int_(static_cast<unsigned int>(OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER));
    m.attr("DENOISER_MODEL_KIND_TEMPORAL") = py::int_(static_cast<unsigned int>(OPTIX_DENOISER_MODEL_KIND_TEMPORAL));
    m.attr("DENOISER_MODEL_KIND_TEMPORAL_AOV") = py::int_(static_cast<unsigned int>(OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV));
    m.attr("DENOISER_MODEL_KIND_UPSCALE2X") = py::int_(static_cast<unsigned int>(OPTIX_DENOISER_MODEL_KIND_UPSCALE2X));
    m.attr("DENOISER_MODEL_KIND_TEMPORAL_UPSCALE2X") = py::int_(static_cast<unsigned int>(OPTIX_DENOISER_MODEL_KIND_TEMPORAL_UPSCALE2X));
#ifdef PYOPTIX_ENABLE_DLSS
    py::enum_<NVSDK_NGX_PerfQuality_Value>(m, "DlssPerfQuality", py::arithmetic())
        .value( "MAX_PERF", NVSDK_NGX_PerfQuality_Value_MaxPerf )
        .value( "BALANCED", NVSDK_NGX_PerfQuality_Value_Balanced )
        .value( "MAX_QUALITY", NVSDK_NGX_PerfQuality_Value_MaxQuality )
        .value( "ULTRA_PERFORMANCE", NVSDK_NGX_PerfQuality_Value_UltraPerformance )
        .value( "ULTRA_QUALITY", NVSDK_NGX_PerfQuality_Value_UltraQuality )
        .value( "DLAA", NVSDK_NGX_PerfQuality_Value_DLAA );

    py::enum_<NVSDK_NGX_RayReconstruction_Hint_Render_Preset>(m, "RayReconstructionHintRenderPreset", py::arithmetic())
        .value( "DEFAULT", NVSDK_NGX_RayReconstruction_Hint_Render_Preset_Default )
        .value( "D", NVSDK_NGX_RayReconstruction_Hint_Render_Preset_D )
        .value( "E", NVSDK_NGX_RayReconstruction_Hint_Render_Preset_E )
        .value( "F", NVSDK_NGX_RayReconstruction_Hint_Render_Preset_F );

    py::enum_<warp_optix_ext::DlssRRResource>(m, "DlssRRResource", py::arithmetic())
        .value( "RESOURCE_COLOR_IN", warp_optix_ext::RESOURCE_COLOR_IN )
        .value( "RESOURCE_COLOR_OUT", warp_optix_ext::RESOURCE_COLOR_OUT )
        .value( "RESOURCE_DIFFUSE_ALBEDO", warp_optix_ext::RESOURCE_DIFFUSE_ALBEDO )
        .value( "RESOURCE_SPECULAR_ALBEDO", warp_optix_ext::RESOURCE_SPECULAR_ALBEDO )
        .value( "RESOURCE_NORMALROUGHNESS", warp_optix_ext::RESOURCE_NORMALROUGHNESS )
        .value( "RESOURCE_MOTIONVECTOR", warp_optix_ext::RESOURCE_MOTIONVECTOR )
        .value( "RESOURCE_LINEARDEPTH", warp_optix_ext::RESOURCE_LINEARDEPTH )
        .value( "RESOURCE_SPECULAR_HITDISTANCE", warp_optix_ext::RESOURCE_SPECULAR_HITDISTANCE );

    py::class_<warp_optix_ext::DlssRRContext>( m, "DlssRRContext" )
        .def( py::init<>() )
        .def(
            "init",
            &warp_optix_ext::dlssRRContextInit,
            py::arg( "projectId" ),
            py::arg( "applicationDataPath" ) = "",
            py::arg( "engineVersion" ) = "1.0.0",
            py::arg( "featureSearchPath" ) = "",
            py::arg( "enableLogging" ) = false
        )
        .def( "deinit", &warp_optix_ext::dlssRRContextDeinit )
        .def( "isDlssRRAvailable", &warp_optix_ext::dlssRRContextIsAvailable )
        .def( "getMinDriverVersion", &warp_optix_ext::dlssRRContextGetMinDriverVersion )
        .def( "querySupportedDlssInputSizes", &warp_optix_ext::dlssRRQuerySupportedDlssInputSizes,
              py::arg( "outputWidth" ), py::arg( "outputHeight" ),
              py::arg( "quality" ) = NVSDK_NGX_PerfQuality_Value_MaxQuality )
        .def( "initDlssRR", &warp_optix_ext::dlssRRCreate, py::arg( "initInfo" ), py::arg( "stream" ) = 0u,
              py::keep_alive<0, 1>() )
        .def(py::self == py::self)
        ;

    py::class_<warp_optix_ext::DlssRRDenoiser>( m, "DlssRRDenoiser" )
        .def( "setResource", &warp_optix_ext::dlssRRSetResource )
        .def( "resetResource", &warp_optix_ext::dlssRRResetResource )
        .def( "denoise", &warp_optix_ext::dlssRRDenoise,
              py::arg( "renderWidth" ),
              py::arg( "renderHeight" ),
              py::arg( "jitterX" ),
              py::arg( "jitterY" ),
              py::arg( "worldToViewMatrix" ),
              py::arg( "viewToClipMatrix" ),
              py::arg( "reset" ) = false,
              py::arg( "indicatorInvertXAxis" ) = 0,
              py::arg( "indicatorInvertYAxis" ) = 0,
              py::arg( "mvScaleX" ) = 1.0f,
              py::arg( "mvScaleY" ) = 1.0f )
        .def( "deinit", &warp_optix_ext::dlssRRDestroy )
        .def(py::self == py::self)
        ;

    m.def( "dlssRRGetResultString", &warp_optix_ext::dlssRRGetResultString, py::arg( "ngxResultCode" ) );

    py::class_<warp_optix_ext::DlssRRInitInfo>(m, "DlssRRInitInfo")
        .def( py::init([]() { return std::unique_ptr<warp_optix_ext::DlssRRInitInfo>(new warp_optix_ext::DlssRRInitInfo{} ); } ) )
        .def_readwrite( "inputWidth", &warp_optix_ext::DlssRRInitInfo::inputWidth )
        .def_readwrite( "inputHeight", &warp_optix_ext::DlssRRInitInfo::inputHeight )
        .def_readwrite( "outputWidth", &warp_optix_ext::DlssRRInitInfo::outputWidth )
        .def_readwrite( "outputHeight", &warp_optix_ext::DlssRRInitInfo::outputHeight )
        .def_readwrite( "quality", &warp_optix_ext::DlssRRInitInfo::quality )
        .def_readwrite( "preset", &warp_optix_ext::DlssRRInitInfo::preset )
        .def_readwrite( "mvJittered", &warp_optix_ext::DlssRRInitInfo::mvJittered )
        .def_readwrite( "lowResolutionMotionVectors", &warp_optix_ext::DlssRRInitInfo::lowResolutionMotionVectors )
        .def_readwrite( "isContentHDR", &warp_optix_ext::DlssRRInitInfo::isContentHDR )
        .def_readwrite( "depthInverted", &warp_optix_ext::DlssRRInitInfo::depthInverted )
        .def_readwrite( "autoExposure", &warp_optix_ext::DlssRRInitInfo::autoExposure )
        .def_readwrite( "useHWDepth", &warp_optix_ext::DlssRRInitInfo::useHWDepth )
        ;

    py::class_<warp_optix_ext::DlssRRSupportedSizes>(m, "DlssRRSupportedSizes")
        .def( py::init([]() { return std::unique_ptr<warp_optix_ext::DlssRRSupportedSizes>(new warp_optix_ext::DlssRRSupportedSizes{} ); } ) )
        .def_readwrite( "minWidth", &warp_optix_ext::DlssRRSupportedSizes::minWidth )
        .def_readwrite( "minHeight", &warp_optix_ext::DlssRRSupportedSizes::minHeight )
        .def_readwrite( "maxWidth", &warp_optix_ext::DlssRRSupportedSizes::maxWidth )
        .def_readwrite( "maxHeight", &warp_optix_ext::DlssRRSupportedSizes::maxHeight )
        .def_readwrite( "optimalWidth", &warp_optix_ext::DlssRRSupportedSizes::optimalWidth )
        .def_readwrite( "optimalHeight", &warp_optix_ext::DlssRRSupportedSizes::optimalHeight )
        ;

#endif
}
