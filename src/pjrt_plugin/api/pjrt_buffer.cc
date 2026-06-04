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

PJRT_Error* MPS_Buffer_Destroy(PJRT_Buffer_Destroy_Args* args) {
    ReleasePjrtBuffer(args->buffer);
    return nullptr;
}

PJRT_Error* MPS_Buffer_ElementType(PJRT_Buffer_ElementType_Args* args) {
    args->type = args->buffer && args->buffer->buffer
                     ? static_cast<PJRT_Buffer_Type>(args->buffer->buffer->dtype())
                     : PJRT_Buffer_Type_F32;
    return nullptr;
}

PJRT_Error* MPS_Buffer_Dimensions(PJRT_Buffer_Dimensions_Args* args) {
    if (args->buffer && args->buffer->buffer) {
        const auto& dims = args->buffer->buffer->dimensions();
        args->dims = dims.data();
        args->num_dims = dims.size();
    } else {
        args->dims = nullptr;
        args->num_dims = 0;
    }
    return nullptr;
}

PJRT_Error* MPS_Buffer_UnpaddedDimensions(PJRT_Buffer_UnpaddedDimensions_Args* args) {
    if (args->buffer && args->buffer->buffer) {
        const auto& dims = args->buffer->buffer->dimensions();
        args->unpadded_dims = dims.data();
        args->num_dims = dims.size();
    } else {
        args->unpadded_dims = nullptr;
        args->num_dims = 0;
    }
    return nullptr;
}

PJRT_Error* MPS_Buffer_DynamicDimensionIndices(PJRT_Buffer_DynamicDimensionIndices_Args* args) {
    args->dynamic_dim_indices = nullptr;
    args->num_dynamic_dims = 0;
    return nullptr;
}

PJRT_Error* MPS_Buffer_GetMemoryLayout(PJRT_Buffer_GetMemoryLayout_Args* args) {
    args->layout.type = PJRT_Buffer_MemoryLayout_Type_Strides;
    if (args->buffer && args->buffer->buffer) {
        const auto& strides = args->buffer->buffer->byte_strides();
        args->layout.strides.byte_strides = strides.data();
        args->layout.strides.num_byte_strides = strides.size();
    }
    return nullptr;
}

PJRT_Error* MPS_Buffer_OnDeviceSizeInBytes(PJRT_Buffer_OnDeviceSizeInBytes_Args* args) {
    args->on_device_size_in_bytes =
        args->buffer && args->buffer->buffer ? args->buffer->buffer->storage_byte_size() : 0;
    return nullptr;
}

PJRT_Error* MPS_Buffer_Device(PJRT_Buffer_Device_Args* args) {
    MPS_LOG_DEBUG(" PJRT_Buffer_Device called, buffer=%p\n", (void*)args->buffer);
    if (args->buffer && args->buffer->client && !args->buffer->client->devices.empty()) {
        args->device = args->buffer->client->devices[0];
        MPS_LOG_DEBUG(" PJRT_Buffer_Device: returning device=%p from client=%p\n", (void*)args->device,
                      (void*)args->buffer->client);
    } else {
        args->device = nullptr;
        MPS_LOG_DEBUG(" PJRT_Buffer_Device: returning nullptr (no client or devices)\n");
    }
    return nullptr;
}

PJRT_Error* MPS_Buffer_Memory(PJRT_Buffer_Memory_Args* args) {
    MPS_LOG_DEBUG(" PJRT_Buffer_Memory called\n");

    if (args->buffer && args->buffer->client && !args->buffer->client->memories.empty()) {
        args->memory = args->buffer->client->memories[0];
    } else {
        args->memory = nullptr;
    }
    return nullptr;
}

PJRT_Error* MPS_Buffer_Delete(PJRT_Buffer_Delete_Args* args) {
    if (args->buffer && args->buffer->buffer) {
        args->buffer->buffer->Delete();
    }
    return nullptr;
}

PJRT_Error* MPS_Buffer_IsDeleted(PJRT_Buffer_IsDeleted_Args* args) {
    args->is_deleted = args->buffer && args->buffer->buffer ? args->buffer->buffer->IsDeleted() : true;
    return nullptr;
}

PJRT_Error* MPS_Buffer_CopyToDevice(PJRT_Buffer_CopyToDevice_Args* args) {
    if (!args->buffer || !args->buffer->buffer) {
        return MakeError("CopyToDevice: null buffer", PJRT_Error_Code_INVALID_ARGUMENT);
    }

    std::string error;
    auto cloned = args->buffer->buffer->Clone(&error);
    if (!cloned) {
        return MakeError("CopyToDevice: " + (error.empty() ? "clone failed" : error));
    }

    auto* dst = new PJRT_Buffer();
    dst->buffer = std::move(cloned);
    dst->client = args->buffer->client;
    args->dst_buffer = dst;
    return nullptr;
}

PJRT_Error* MPS_Buffer_CopyToMemory(PJRT_Buffer_CopyToMemory_Args* args) {
    if (!args->buffer || !args->buffer->buffer) {
        return MakeError("CopyToMemory: null buffer", PJRT_Error_Code_INVALID_ARGUMENT);
    }

    std::string error;
    auto cloned = args->buffer->buffer->Clone(&error);
    if (!cloned) {
        return MakeError("CopyToMemory: " + (error.empty() ? "clone failed" : error));
    }

    auto* dst = new PJRT_Buffer();
    dst->buffer = std::move(cloned);
    dst->client = args->buffer->client;
    args->dst_buffer = dst;
    return nullptr;
}

PJRT_Error* MPS_Buffer_CopyRawToHost(PJRT_Buffer_CopyRawToHost_Args* args) {
    if (!args->buffer || !args->buffer->buffer) {
        return MakeError("CopyRawToHost: null buffer", PJRT_Error_Code_INVALID_ARGUMENT);
    }
    if (!args->dst) {
        return MakeError("CopyRawToHost: null destination", PJRT_Error_Code_INVALID_ARGUMENT);
    }

    std::string error;
    bool ok = args->buffer->buffer->CopyRawToHost(args->dst, args->offset, args->transfer_size, &error);

    auto* event = new PJRT_Event();
    if (ok) {
        event->completion = jax_metallib::MakeReadyCompletionEvent();
    } else {
        event->completion = jax_metallib::MakePendingCompletionEvent();
        jax_metallib::MarkCompletionEventError(event->completion,
                                               error.empty() ? "CopyRawToHost failed" : error);
    }
    args->event = event;
    return nullptr;
}

PJRT_Error* MPS_Buffer_CopyRawToHostFuture(PJRT_Buffer_CopyRawToHostFuture_Args* args) {
    if (!args->buffer || !args->buffer->buffer) {
        return MakeError("CopyRawToHostFuture: null buffer", PJRT_Error_Code_INVALID_ARGUMENT);
    }

    auto* event = new PJRT_Event();
    event->completion = jax_metallib::MakePendingCompletionEvent();
    args->event = event;

    struct CallbackData {
        PJRT_Buffer* buffer;
        int64_t offset;
        int64_t transfer_size;
        jax_metallib::CompletionEvent completion;
    };
    auto* cb_data = new CallbackData{args->buffer, args->offset, args->transfer_size, event->completion};
    args->callback_data = cb_data;

    args->future_ready_callback = [](PJRT_Buffer_CopyRawToHostFuture_Callback_Args* cb_args) {
        auto* data = static_cast<CallbackData*>(cb_args->callback_data);
        if (!data)
            return;

        if (cb_args->error_code != PJRT_Error_Code_OK) {
            std::string msg(cb_args->error_message, cb_args->error_message_size);
            jax_metallib::MarkCompletionEventError(data->completion, std::move(msg));
        } else if (cb_args->dst) {
            std::string error;
            if (data->buffer && data->buffer->buffer &&
                data->buffer->buffer->CopyRawToHost(cb_args->dst, data->offset, data->transfer_size,
                                                    &error)) {
                jax_metallib::MarkCompletionEventReady(data->completion);
            } else {
                jax_metallib::MarkCompletionEventError(
                    data->completion, error.empty() ? "CopyRawToHostFuture: copy failed" : error);
            }
        } else {
            jax_metallib::MarkCompletionEventError(data->completion,
                                                   "CopyRawToHostFuture: null dst in callback");
        }
        delete data;
    };

    return nullptr;
}

PJRT_Error* MPS_Buffer_ToHostBuffer(PJRT_Buffer_ToHostBuffer_Args* args) {
    auto* event = new PJRT_Event();
    event->completion = jax_metallib::MakePendingCompletionEvent();
    if (args->src && args->src->buffer && args->dst) {
        std::string error;
        if (args->src->buffer->ToHostBuffer(args->dst, nullptr, &error)) {
            jax_metallib::MarkCompletionEventReady(event->completion);
        } else {
            jax_metallib::MarkCompletionEventError(event->completion,
                                                   error.empty() ? "ToHostBuffer failed" : error);
        }
    } else {
        jax_metallib::MarkCompletionEventReady(event->completion);
    }
    args->event = event;

    return nullptr;
}

PJRT_Error* MPS_Buffer_IsOnCpu(PJRT_Buffer_IsOnCpu_Args* args) {
    args->is_on_cpu = false;
    return nullptr;
}

PJRT_Error* MPS_Buffer_ReadyEvent(PJRT_Buffer_ReadyEvent_Args* args) {
    auto* event = AcquirePjrtEvent();
    event->completion = args->buffer && args->buffer->buffer ? args->buffer->buffer->producer_completion()
                                                             : jax_metallib::MakeReadyCompletionEvent();
    args->event = event;
    return nullptr;
}

PJRT_Error* MPS_Buffer_UnsafePointer(PJRT_Buffer_UnsafePointer_Args* args) {
    args->buffer_pointer = 0;
    return nullptr;
}

PJRT_Error* MPS_Buffer_IncreaseExternalReferenceCount(PJRT_Buffer_IncreaseExternalReferenceCount_Args* args) {
    return nullptr;
}

PJRT_Error* MPS_Buffer_DecreaseExternalReferenceCount(PJRT_Buffer_DecreaseExternalReferenceCount_Args* args) {
    return nullptr;
}

PJRT_Error* MPS_Buffer_OpaqueDeviceMemoryDataPointer(PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args* args) {
    args->device_memory_ptr = nullptr;
    return nullptr;
}

PJRT_Error* MPS_CopyToDeviceStream_Destroy(PJRT_CopyToDeviceStream_Destroy_Args* args) {
    return nullptr;
}

PJRT_Error* MPS_CopyToDeviceStream_AddChunk(PJRT_CopyToDeviceStream_AddChunk_Args* args) {
    return MakeError("CopyToDeviceStream not implemented", PJRT_Error_Code_UNIMPLEMENTED);
}

PJRT_Error* MPS_CopyToDeviceStream_TotalBytes(PJRT_CopyToDeviceStream_TotalBytes_Args* args) {
    args->total_bytes = 0;
    return nullptr;
}

PJRT_Error* MPS_CopyToDeviceStream_GranuleSize(PJRT_CopyToDeviceStream_GranuleSize_Args* args) {
    args->granule_size_in_bytes = 0;
    return nullptr;
}

PJRT_Error* MPS_CopyToDeviceStream_CurrentBytes(PJRT_CopyToDeviceStream_CurrentBytes_Args* args) {
    args->current_bytes = 0;
    return nullptr;
}
