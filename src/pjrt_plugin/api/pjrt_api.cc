// Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "pjrt_plugin/core/logging.h"
#include "pjrt_plugin/core/pjrt_types.h"

const char* const kPlatformName = "mps";
const char* const kPlatformVersion = "0.1.0";

static PJRT_Client* g_default_client = nullptr;

PJRT_Client* GetOrCreateDefaultClient() {
    if (!g_default_client)
        g_default_client = CreatePjrtClient();
    return g_default_client;
}

PJRT_Client* GetClient(PJRT_Client* client) {
    return client ? client : GetOrCreateDefaultClient();
}

void MPS_Error_Destroy(PJRT_Error_Destroy_Args* args);
void MPS_Error_Message(PJRT_Error_Message_Args* args);
PJRT_Error* MPS_Error_GetCode(PJRT_Error_GetCode_Args* args);
#if PJRT_API_MINOR >= 99
PJRT_Error* MPS_Error_ForEachPayload(PJRT_Error_ForEachPayload_Args* args);
#endif

PJRT_Error* MPS_Plugin_Initialize(PJRT_Plugin_Initialize_Args* args);
PJRT_Error* MPS_Plugin_Attributes(PJRT_Plugin_Attributes_Args* args);

PJRT_Error* MPS_Event_Destroy(PJRT_Event_Destroy_Args* args);
PJRT_Error* MPS_Event_IsReady(PJRT_Event_IsReady_Args* args);
PJRT_Error* MPS_Event_Error(PJRT_Event_Error_Args* args);
PJRT_Error* MPS_Event_Await(PJRT_Event_Await_Args* args);
PJRT_Error* MPS_Event_Create(PJRT_Event_Create_Args* args);
PJRT_Error* MPS_Event_Set(PJRT_Event_Set_Args* args);
PJRT_Error* MPS_Event_OnReady(PJRT_Event_OnReady_Args* args);

PJRT_Error* MPS_Client_Create(PJRT_Client_Create_Args* args);
PJRT_Error* MPS_Client_Destroy(PJRT_Client_Destroy_Args* args);
PJRT_Error* MPS_Client_PlatformName(PJRT_Client_PlatformName_Args* args);
PJRT_Error* MPS_Client_ProcessIndex(PJRT_Client_ProcessIndex_Args* args);
PJRT_Error* MPS_Client_PlatformVersion(PJRT_Client_PlatformVersion_Args* args);
PJRT_Error* MPS_Client_Devices(PJRT_Client_Devices_Args* args);
PJRT_Error* MPS_Client_AddressableDevices(PJRT_Client_AddressableDevices_Args* args);
PJRT_Error* MPS_Client_LookupDevice(PJRT_Client_LookupDevice_Args* args);
PJRT_Error* MPS_Client_LookupAddressableDevice(PJRT_Client_LookupAddressableDevice_Args* args);
PJRT_Error* MPS_Client_AddressableMemories(PJRT_Client_AddressableMemories_Args* args);
PJRT_Error* MPS_Client_Compile(PJRT_Client_Compile_Args* args);
PJRT_Error* MPS_Client_DefaultDeviceAssignment(PJRT_Client_DefaultDeviceAssignment_Args* args);
PJRT_Error* MPS_Client_BufferFromHostBuffer(PJRT_Client_BufferFromHostBuffer_Args* args);
PJRT_Error* MPS_Client_CreateViewOfDeviceBuffer(PJRT_Client_CreateViewOfDeviceBuffer_Args* args);
PJRT_Error* MPS_Client_CreateUninitializedBuffer(PJRT_Client_CreateUninitializedBuffer_Args* args);
PJRT_Error* MPS_Client_CreateErrorBuffer(PJRT_Client_CreateErrorBuffer_Args* args);

PJRT_Error* MPS_DeviceDescription_Id(PJRT_DeviceDescription_Id_Args* args);
PJRT_Error* MPS_DeviceDescription_ProcessIndex(PJRT_DeviceDescription_ProcessIndex_Args* args);
PJRT_Error* MPS_DeviceDescription_Attributes(PJRT_DeviceDescription_Attributes_Args* args);
PJRT_Error* MPS_DeviceDescription_Kind(PJRT_DeviceDescription_Kind_Args* args);
PJRT_Error* MPS_DeviceDescription_DebugString(PJRT_DeviceDescription_DebugString_Args* args);
PJRT_Error* MPS_DeviceDescription_ToString(PJRT_DeviceDescription_ToString_Args* args);

PJRT_Error* MPS_Device_GetDescription(PJRT_Device_GetDescription_Args* args);
PJRT_Error* MPS_Device_IsAddressable(PJRT_Device_IsAddressable_Args* args);
PJRT_Error* MPS_Device_LocalHardwareId(PJRT_Device_LocalHardwareId_Args* args);
PJRT_Error* MPS_Device_AddressableMemories(PJRT_Device_AddressableMemories_Args* args);
PJRT_Error* MPS_Device_DefaultMemory(PJRT_Device_DefaultMemory_Args* args);
PJRT_Error* MPS_Device_MemoryStats(PJRT_Device_MemoryStats_Args* args);

PJRT_Error* MPS_Memory_Id(PJRT_Memory_Id_Args* args);
PJRT_Error* MPS_Memory_Kind(PJRT_Memory_Kind_Args* args);
PJRT_Error* MPS_Memory_DebugString(PJRT_Memory_DebugString_Args* args);
PJRT_Error* MPS_Memory_ToString(PJRT_Memory_ToString_Args* args);
PJRT_Error* MPS_Memory_Kind_Id(PJRT_Memory_Kind_Id_Args* args);
PJRT_Error* MPS_Memory_AddressableByDevices(PJRT_Memory_AddressableByDevices_Args* args);

PJRT_Error* MPS_Executable_Destroy(PJRT_Executable_Destroy_Args* args);
PJRT_Error* MPS_Executable_Name(PJRT_Executable_Name_Args* args);
PJRT_Error* MPS_Executable_NumReplicas(PJRT_Executable_NumReplicas_Args* args);
PJRT_Error* MPS_Executable_NumPartitions(PJRT_Executable_NumPartitions_Args* args);
PJRT_Error* MPS_Executable_NumOutputs(PJRT_Executable_NumOutputs_Args* args);
PJRT_Error* MPS_Executable_SizeOfGeneratedCodeInBytes(PJRT_Executable_SizeOfGeneratedCodeInBytes_Args* args);
PJRT_Error* MPS_Executable_GetCostAnalysis(PJRT_Executable_GetCostAnalysis_Args* args);
PJRT_Error* MPS_Executable_OutputMemoryKinds(PJRT_Executable_OutputMemoryKinds_Args* args);
PJRT_Error* MPS_Executable_OutputElementTypes(PJRT_Executable_OutputElementTypes_Args* args);
PJRT_Error* MPS_Executable_OutputDimensions(PJRT_Executable_OutputDimensions_Args* args);
PJRT_Error* MPS_Executable_OptimizedProgram(PJRT_Executable_OptimizedProgram_Args* args);
PJRT_Error* MPS_Executable_Serialize(PJRT_Executable_Serialize_Args* args);
PJRT_Error* MPS_Executable_Fingerprint(PJRT_Executable_Fingerprint_Args* args);

PJRT_Error* MPS_LoadedExecutable_Destroy(PJRT_LoadedExecutable_Destroy_Args* args);
PJRT_Error* MPS_LoadedExecutable_GetExecutable(PJRT_LoadedExecutable_GetExecutable_Args* args);
PJRT_Error* MPS_LoadedExecutable_AddressableDevices(PJRT_LoadedExecutable_AddressableDevices_Args* args);
PJRT_Error* MPS_LoadedExecutable_Delete(PJRT_LoadedExecutable_Delete_Args* args);
PJRT_Error* MPS_LoadedExecutable_IsDeleted(PJRT_LoadedExecutable_IsDeleted_Args* args);
PJRT_Error* MPS_LoadedExecutable_Execute(PJRT_LoadedExecutable_Execute_Args* args);
PJRT_Error* MPS_Executable_DeserializeAndLoad(PJRT_Executable_DeserializeAndLoad_Args* args);
PJRT_Error* MPS_LoadedExecutable_Fingerprint(PJRT_LoadedExecutable_Fingerprint_Args* args);
PJRT_Error* MPS_LoadedExecutable_GetDeviceAssignment(PJRT_LoadedExecutable_GetDeviceAssignment_Args* args);

PJRT_Error* MPS_Buffer_Destroy(PJRT_Buffer_Destroy_Args* args);
PJRT_Error* MPS_Buffer_ElementType(PJRT_Buffer_ElementType_Args* args);
PJRT_Error* MPS_Buffer_Dimensions(PJRT_Buffer_Dimensions_Args* args);
PJRT_Error* MPS_Buffer_UnpaddedDimensions(PJRT_Buffer_UnpaddedDimensions_Args* args);
PJRT_Error* MPS_Buffer_DynamicDimensionIndices(PJRT_Buffer_DynamicDimensionIndices_Args* args);
PJRT_Error* MPS_Buffer_GetMemoryLayout(PJRT_Buffer_GetMemoryLayout_Args* args);
PJRT_Error* MPS_Buffer_OnDeviceSizeInBytes(PJRT_Buffer_OnDeviceSizeInBytes_Args* args);
PJRT_Error* MPS_Buffer_Device(PJRT_Buffer_Device_Args* args);
PJRT_Error* MPS_Buffer_Memory(PJRT_Buffer_Memory_Args* args);
PJRT_Error* MPS_Buffer_Delete(PJRT_Buffer_Delete_Args* args);
PJRT_Error* MPS_Buffer_IsDeleted(PJRT_Buffer_IsDeleted_Args* args);
PJRT_Error* MPS_Buffer_CopyToDevice(PJRT_Buffer_CopyToDevice_Args* args);
PJRT_Error* MPS_Buffer_CopyToMemory(PJRT_Buffer_CopyToMemory_Args* args);
PJRT_Error* MPS_Buffer_CopyRawToHost(PJRT_Buffer_CopyRawToHost_Args* args);
PJRT_Error* MPS_Buffer_CopyRawToHostFuture(PJRT_Buffer_CopyRawToHostFuture_Args* args);
PJRT_Error* MPS_Buffer_ToHostBuffer(PJRT_Buffer_ToHostBuffer_Args* args);
PJRT_Error* MPS_Buffer_IsOnCpu(PJRT_Buffer_IsOnCpu_Args* args);
PJRT_Error* MPS_Buffer_ReadyEvent(PJRT_Buffer_ReadyEvent_Args* args);
PJRT_Error* MPS_Buffer_UnsafePointer(PJRT_Buffer_UnsafePointer_Args* args);
PJRT_Error* MPS_Buffer_IncreaseExternalReferenceCount(PJRT_Buffer_IncreaseExternalReferenceCount_Args* args);
PJRT_Error* MPS_Buffer_DecreaseExternalReferenceCount(PJRT_Buffer_DecreaseExternalReferenceCount_Args* args);
PJRT_Error* MPS_Buffer_OpaqueDeviceMemoryDataPointer(PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args* args);

PJRT_Error* MPS_CopyToDeviceStream_Destroy(PJRT_CopyToDeviceStream_Destroy_Args* args);
PJRT_Error* MPS_CopyToDeviceStream_AddChunk(PJRT_CopyToDeviceStream_AddChunk_Args* args);
PJRT_Error* MPS_CopyToDeviceStream_TotalBytes(PJRT_CopyToDeviceStream_TotalBytes_Args* args);
PJRT_Error* MPS_CopyToDeviceStream_GranuleSize(PJRT_CopyToDeviceStream_GranuleSize_Args* args);
PJRT_Error* MPS_CopyToDeviceStream_CurrentBytes(PJRT_CopyToDeviceStream_CurrentBytes_Args* args);

PJRT_Error* MPS_Client_TopologyDescription(PJRT_Client_TopologyDescription_Args* args);
PJRT_Error* MPS_TopologyDescription_Create(PJRT_TopologyDescription_Create_Args* args);
PJRT_Error* MPS_TopologyDescription_Destroy(PJRT_TopologyDescription_Destroy_Args* args);
PJRT_Error* MPS_TopologyDescription_PlatformName(PJRT_TopologyDescription_PlatformName_Args* args);
PJRT_Error* MPS_TopologyDescription_PlatformVersion(PJRT_TopologyDescription_PlatformVersion_Args* args);
PJRT_Error*
MPS_TopologyDescription_GetDeviceDescriptions(PJRT_TopologyDescription_GetDeviceDescriptions_Args* args);
PJRT_Error* MPS_TopologyDescription_Serialize(PJRT_TopologyDescription_Serialize_Args* args);
PJRT_Error* MPS_TopologyDescription_Attributes(PJRT_TopologyDescription_Attributes_Args* args);
#if PJRT_API_MINOR >= 101
PJRT_Error* MPS_TopologyDescription_Fingerprint(PJRT_TopologyDescription_Fingerprint_Args* args);
#endif
#if PJRT_API_MINOR >= 102
PJRT_Error* MPS_Executable_ParameterMemoryKinds(PJRT_Executable_ParameterMemoryKinds_Args* args);
#endif

PJRT_Error* MPS_Compile(PJRT_Compile_Args* args);

static const PJRT_Api pjrt_api = {
    .struct_size = PJRT_Api_STRUCT_SIZE,
    .extension_start = nullptr,

    .pjrt_api_version =
        {
            .struct_size = PJRT_Api_Version_STRUCT_SIZE,
            .extension_start = nullptr,
            .major_version = PJRT_API_MAJOR,
            .minor_version = PJRT_API_MINOR,
        },

    .PJRT_Error_Destroy = MPS_Error_Destroy,
    .PJRT_Error_Message = MPS_Error_Message,
    .PJRT_Error_GetCode = MPS_Error_GetCode,

    .PJRT_Plugin_Initialize = MPS_Plugin_Initialize,
    .PJRT_Plugin_Attributes = MPS_Plugin_Attributes,

    .PJRT_Event_Destroy = MPS_Event_Destroy,
    .PJRT_Event_IsReady = MPS_Event_IsReady,
    .PJRT_Event_Error = MPS_Event_Error,
    .PJRT_Event_Await = MPS_Event_Await,
    .PJRT_Event_OnReady = MPS_Event_OnReady,

    .PJRT_Client_Create = MPS_Client_Create,
    .PJRT_Client_Destroy = MPS_Client_Destroy,
    .PJRT_Client_PlatformName = MPS_Client_PlatformName,
    .PJRT_Client_ProcessIndex = MPS_Client_ProcessIndex,
    .PJRT_Client_PlatformVersion = MPS_Client_PlatformVersion,
    .PJRT_Client_Devices = MPS_Client_Devices,
    .PJRT_Client_AddressableDevices = MPS_Client_AddressableDevices,
    .PJRT_Client_LookupDevice = MPS_Client_LookupDevice,
    .PJRT_Client_LookupAddressableDevice = MPS_Client_LookupAddressableDevice,
    .PJRT_Client_AddressableMemories = MPS_Client_AddressableMemories,
    .PJRT_Client_Compile = MPS_Client_Compile,
    .PJRT_Client_DefaultDeviceAssignment = MPS_Client_DefaultDeviceAssignment,
    .PJRT_Client_BufferFromHostBuffer = MPS_Client_BufferFromHostBuffer,

    .PJRT_DeviceDescription_Id = MPS_DeviceDescription_Id,
    .PJRT_DeviceDescription_ProcessIndex = MPS_DeviceDescription_ProcessIndex,
    .PJRT_DeviceDescription_Attributes = MPS_DeviceDescription_Attributes,
    .PJRT_DeviceDescription_Kind = MPS_DeviceDescription_Kind,
    .PJRT_DeviceDescription_DebugString = MPS_DeviceDescription_DebugString,
    .PJRT_DeviceDescription_ToString = MPS_DeviceDescription_ToString,

    .PJRT_Device_GetDescription = MPS_Device_GetDescription,
    .PJRT_Device_IsAddressable = MPS_Device_IsAddressable,
    .PJRT_Device_LocalHardwareId = MPS_Device_LocalHardwareId,
    .PJRT_Device_AddressableMemories = MPS_Device_AddressableMemories,
    .PJRT_Device_DefaultMemory = MPS_Device_DefaultMemory,
    .PJRT_Device_MemoryStats = MPS_Device_MemoryStats,

    .PJRT_Memory_Id = MPS_Memory_Id,
    .PJRT_Memory_Kind = MPS_Memory_Kind,
    .PJRT_Memory_DebugString = MPS_Memory_DebugString,
    .PJRT_Memory_ToString = MPS_Memory_ToString,
    .PJRT_Memory_AddressableByDevices = MPS_Memory_AddressableByDevices,

    .PJRT_Executable_Destroy = MPS_Executable_Destroy,
    .PJRT_Executable_Name = MPS_Executable_Name,
    .PJRT_Executable_NumReplicas = MPS_Executable_NumReplicas,
    .PJRT_Executable_NumPartitions = MPS_Executable_NumPartitions,
    .PJRT_Executable_NumOutputs = MPS_Executable_NumOutputs,
    .PJRT_Executable_SizeOfGeneratedCodeInBytes = MPS_Executable_SizeOfGeneratedCodeInBytes,
    .PJRT_Executable_GetCostAnalysis = MPS_Executable_GetCostAnalysis,
    .PJRT_Executable_OutputMemoryKinds = MPS_Executable_OutputMemoryKinds,
    .PJRT_Executable_OptimizedProgram = MPS_Executable_OptimizedProgram,
    .PJRT_Executable_Serialize = MPS_Executable_Serialize,

    .PJRT_LoadedExecutable_Destroy = MPS_LoadedExecutable_Destroy,
    .PJRT_LoadedExecutable_GetExecutable = MPS_LoadedExecutable_GetExecutable,
    .PJRT_LoadedExecutable_AddressableDevices = MPS_LoadedExecutable_AddressableDevices,
    .PJRT_LoadedExecutable_Delete = MPS_LoadedExecutable_Delete,
    .PJRT_LoadedExecutable_IsDeleted = MPS_LoadedExecutable_IsDeleted,
    .PJRT_LoadedExecutable_Execute = MPS_LoadedExecutable_Execute,
    .PJRT_Executable_DeserializeAndLoad = MPS_Executable_DeserializeAndLoad,
    .PJRT_LoadedExecutable_Fingerprint = MPS_LoadedExecutable_Fingerprint,

    .PJRT_Buffer_Destroy = MPS_Buffer_Destroy,
    .PJRT_Buffer_ElementType = MPS_Buffer_ElementType,
    .PJRT_Buffer_Dimensions = MPS_Buffer_Dimensions,
    .PJRT_Buffer_UnpaddedDimensions = MPS_Buffer_UnpaddedDimensions,
    .PJRT_Buffer_DynamicDimensionIndices = MPS_Buffer_DynamicDimensionIndices,
    .PJRT_Buffer_GetMemoryLayout = MPS_Buffer_GetMemoryLayout,
    .PJRT_Buffer_OnDeviceSizeInBytes = MPS_Buffer_OnDeviceSizeInBytes,
    .PJRT_Buffer_Device = MPS_Buffer_Device,
    .PJRT_Buffer_Memory = MPS_Buffer_Memory,
    .PJRT_Buffer_Delete = MPS_Buffer_Delete,
    .PJRT_Buffer_IsDeleted = MPS_Buffer_IsDeleted,
    .PJRT_Buffer_CopyToDevice = MPS_Buffer_CopyToDevice,
    .PJRT_Buffer_ToHostBuffer = MPS_Buffer_ToHostBuffer,
    .PJRT_Buffer_IsOnCpu = MPS_Buffer_IsOnCpu,
    .PJRT_Buffer_ReadyEvent = MPS_Buffer_ReadyEvent,
    .PJRT_Buffer_UnsafePointer = MPS_Buffer_UnsafePointer,
    .PJRT_Buffer_IncreaseExternalReferenceCount = MPS_Buffer_IncreaseExternalReferenceCount,
    .PJRT_Buffer_DecreaseExternalReferenceCount = MPS_Buffer_DecreaseExternalReferenceCount,
    .PJRT_Buffer_OpaqueDeviceMemoryDataPointer = MPS_Buffer_OpaqueDeviceMemoryDataPointer,

    .PJRT_CopyToDeviceStream_Destroy = MPS_CopyToDeviceStream_Destroy,
    .PJRT_CopyToDeviceStream_AddChunk = MPS_CopyToDeviceStream_AddChunk,
    .PJRT_CopyToDeviceStream_TotalBytes = MPS_CopyToDeviceStream_TotalBytes,
    .PJRT_CopyToDeviceStream_GranuleSize = MPS_CopyToDeviceStream_GranuleSize,
    .PJRT_CopyToDeviceStream_CurrentBytes = MPS_CopyToDeviceStream_CurrentBytes,

    .PJRT_TopologyDescription_Create = MPS_TopologyDescription_Create,
    .PJRT_TopologyDescription_Destroy = MPS_TopologyDescription_Destroy,
    .PJRT_TopologyDescription_PlatformName = MPS_TopologyDescription_PlatformName,
    .PJRT_TopologyDescription_PlatformVersion = MPS_TopologyDescription_PlatformVersion,
    .PJRT_TopologyDescription_GetDeviceDescriptions = MPS_TopologyDescription_GetDeviceDescriptions,
    .PJRT_TopologyDescription_Serialize = MPS_TopologyDescription_Serialize,
    .PJRT_TopologyDescription_Attributes = MPS_TopologyDescription_Attributes,

    .PJRT_Compile = MPS_Compile,

    .PJRT_Executable_OutputElementTypes = MPS_Executable_OutputElementTypes,
    .PJRT_Executable_OutputDimensions = MPS_Executable_OutputDimensions,
    .PJRT_Buffer_CopyToMemory = MPS_Buffer_CopyToMemory,
    .PJRT_Client_CreateViewOfDeviceBuffer = MPS_Client_CreateViewOfDeviceBuffer,
    .PJRT_Executable_Fingerprint = MPS_Executable_Fingerprint,
    .PJRT_Client_TopologyDescription = MPS_Client_TopologyDescription,
    .PJRT_Executable_GetCompiledMemoryStats = nullptr,
    .PJRT_Memory_Kind_Id = MPS_Memory_Kind_Id,
    .PJRT_ExecuteContext_Create = nullptr,
    .PJRT_ExecuteContext_Destroy = nullptr,
    .PJRT_Buffer_CopyRawToHost = MPS_Buffer_CopyRawToHost,
    .PJRT_AsyncHostToDeviceTransferManager_Destroy = nullptr,
    .PJRT_AsyncHostToDeviceTransferManager_TransferData = nullptr,
    .PJRT_Client_CreateBuffersForAsyncHostToDevice = nullptr,
    .PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer = nullptr,
    .PJRT_AsyncHostToDeviceTransferManager_Device = nullptr,
    .PJRT_AsyncHostToDeviceTransferManager_BufferCount = nullptr,
    .PJRT_AsyncHostToDeviceTransferManager_BufferSize = nullptr,
    .PJRT_AsyncHostToDeviceTransferManager_SetBufferError = nullptr,
    .PJRT_AsyncHostToDeviceTransferManager_AddMetadata = nullptr,
    .PJRT_Client_DmaMap = nullptr,
    .PJRT_Client_DmaUnmap = nullptr,
    .PJRT_Client_CreateUninitializedBuffer = MPS_Client_CreateUninitializedBuffer,
    .PJRT_Client_UpdateGlobalProcessInfo = nullptr,
    .PJRT_TopologyDescription_Deserialize = nullptr,
    .PJRT_Client_CreateAliasBuffer = nullptr,
    .PJRT_Client_FulfillAliasBuffer = nullptr,
    .PJRT_LoadedExecutable_GetDeviceAssignment = MPS_LoadedExecutable_GetDeviceAssignment,
    .PJRT_Client_CreateErrorBuffer = MPS_Client_CreateErrorBuffer,
    .PJRT_AsyncHostToDeviceTransferManager_TransferLiteral = nullptr,
    .PJRT_Buffer_CopyRawToHostFuture = MPS_Buffer_CopyRawToHostFuture,
    .PJRT_Device_PoisonExecution = nullptr,
    .PJRT_Device_CreateAsyncTrackingEvent = nullptr,
    .PJRT_AsyncTrackingEvent_Destroy = nullptr,
    .PJRT_Executable_GetCompileOptions = nullptr,
    .PJRT_Buffer_DonateWithControlDependency = nullptr,
    .PJRT_Event_Create = MPS_Event_Create,
    .PJRT_Event_Set = MPS_Event_Set,
#if PJRT_API_MINOR >= 99
    .PJRT_Error_ForEachPayload = MPS_Error_ForEachPayload,
#endif
#if PJRT_API_MINOR >= 101
    .PJRT_TopologyDescription_Fingerprint = MPS_TopologyDescription_Fingerprint,
#endif
#if PJRT_API_MINOR >= 102
    .PJRT_Executable_ParameterMemoryKinds = MPS_Executable_ParameterMemoryKinds,
#endif
};

extern "C" {

__attribute__((visibility("default"))) const PJRT_Api* GetPjrtApi() {
    return &pjrt_api;
}
}
