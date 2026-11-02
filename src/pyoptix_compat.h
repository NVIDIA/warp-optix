// Copyright (c) 2022 NVIDIA CORPORATION All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// PyOptiX 9.1.0 does not export native handles or a C API. These four
// upstream wrapper definitions provide a deliberately version-pinned pybind11
// bridge. No PyOptiX types are registered by our extension. At import we verify
// the registered type identities, sizes, and binding ABI before accessing them.
// Replace this bridge with upstream capsules when they become available.
namespace pyoptix {
struct DeviceContext
{
    OptixDeviceContext deviceContext = 0;
    py::object         logCallbackFunction;
};

struct ModuleCompileBoundValueEntry
{
    ModuleCompileBoundValueEntry(
        size_t pipelineParamOffsetInBytes,
        const py::buffer& boundValue,
        const std::string& annotation
        )
    {
        entry.pipelineParamOffsetInBytes = pipelineParamOffsetInBytes;
        setBoundValue( boundValue );
        setAnnotation( annotation );
    }


    ModuleCompileBoundValueEntry(const ModuleCompileBoundValueEntry& other)
    {
        value      = other.value;
        annotation = other.annotation;
        entry      = other.entry;
    }


    ModuleCompileBoundValueEntry( ModuleCompileBoundValueEntry&& other )
    {
        value      = std::move( other.value );
        annotation = std::move( other.annotation );
        entry      = other.entry;
    }


    void setBoundValue( const py::buffer& val )
    {
        py::buffer_info binfo = val.request();
        if( binfo.ndim != 1 )
            throw std::runtime_error(
                "Multi-dimensional array passed as value for "
                "optix.ModuleCompileBoundValueEntry.boundValue" );

        size_t byte_size = binfo.itemsize * binfo.shape[0];
        const std::byte* bytes = reinterpret_cast<const std::byte*>(binfo.ptr);
        value.clear();
        std::copy( bytes, bytes+byte_size, std::back_inserter( value ) );

    }


    void setAnnotation( const std::string& val )
    {
        annotation = val;
    }


    void sync()
    {
        entry.annotation    = annotation.c_str();
        entry.sizeInBytes   = value.size();
        entry.boundValuePtr = value.data();
    }


    OptixModuleCompileBoundValueEntry entry{};
    std::string            annotation;
    std::vector<std::byte> value;
};

struct PayloadType
{
    PayloadType() {}
    PayloadType( const py::list&  payload_semantics )
    {
        setPayloadSemantics( payload_semantics );
    }


    void setPayloadSemantics( const py::list& val )
    {
        payload_semantics = val.cast<std::vector<uint32_t> >();
    }


    void sync()
    {
        payload_type.numPayloadValues = payload_semantics.size();
        if( !payload_semantics.empty() )
            payload_type.payloadSemantics = payload_semantics.data();
        else
            payload_type.payloadSemantics = nullptr;
    }


    OptixPayloadType  payload_type{};
    std::vector<uint32_t> payload_semantics;
};

struct ModuleCompileOptions
{
    ModuleCompileOptions(
        int32_t                       maxRegisterCount,
        OptixCompileOptimizationLevel optLevel,
        OptixCompileDebugLevel        debugLevel
        IF_OPTIX72( COMMA std::vector<pyoptix::ModuleCompileBoundValueEntry>&& bound_values )
        IF_OPTIX74( COMMA std::vector<pyoptix::PayloadType>&& payload_types )
        )
    {
        memset(&options, 0, sizeof(OptixModuleCompileOptions));
        options.maxRegisterCount = maxRegisterCount;
        options.optLevel         = optLevel;
        options.debugLevel       = debugLevel;

        IF_OPTIX72(
        pyboundValues = std::move( bound_values );
        )

        IF_OPTIX74(
        pypayloadTypes = std::move( payload_types );
        )
    }


    void sync()
    {
        return;
#if OPTIX_VERSION >= 70200
        boundValues.clear();
        for( auto& pybve : pyboundValues )
        {
            pybve.sync();
            boundValues.push_back( pybve.entry );
        }

        options.boundValues    = boundValues.empty() ?
                                 nullptr             :
                                 boundValues.data();
        options.numBoundValues = static_cast<uint32_t>( boundValues.size() );
#endif

#if OPTIX_VERSION >= 70400
        payloadTypes.clear();
        for( auto& pypt : pypayloadTypes )
        {
            pypt.sync();
            payloadTypes.push_back( pypt.payload_type);
        }

        options.payloadTypes    = payloadTypes.empty() ?
                                 nullptr             :
                                 payloadTypes.data();
        options.numPayloadTypes= static_cast<uint32_t>( payloadTypes.size() );
#endif
    }

    OptixModuleCompileOptions options{};

#if OPTIX_VERSION >= 70200
    std::vector<pyoptix::ModuleCompileBoundValueEntry> pyboundValues;
    std::vector<OptixModuleCompileBoundValueEntry>     boundValues;
#endif
#if OPTIX_VERSION >= 70400
    std::vector<pyoptix::PayloadType> pypayloadTypes;
    std::vector<OptixPayloadType>     payloadTypes;
#endif
};

} // namespace pyoptix
