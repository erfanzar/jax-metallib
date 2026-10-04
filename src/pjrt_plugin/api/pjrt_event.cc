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

PJRT_Error* MPS_Event_Destroy(PJRT_Event_Destroy_Args* args) {
    ReleasePjrtEvent(args->event);
    return nullptr;
}

PJRT_Error* MPS_Event_IsReady(PJRT_Event_IsReady_Args* args) {
    if (!args->event || !args->event->completion) {
        args->is_ready = true;
        return nullptr;
    }
    args->is_ready = args->event->completion->IsReady();
    return nullptr;
}

PJRT_Error* MPS_Event_Error(PJRT_Event_Error_Args* args) {
    if (!args->event || !args->event->completion) {
        return nullptr;
    }
    if (!args->event->completion->IsReady()) {
        return MakeError("Event is not ready", PJRT_Error_Code_FAILED_PRECONDITION);
    }
    const std::string error = args->event->completion->Error();
    if (!error.empty()) {
        return MakeError(error);
    }
    return nullptr;
}

PJRT_Error* MPS_Event_Await(PJRT_Event_Await_Args* args) {
    if (!args->event || !args->event->completion) {
        return nullptr;
    }
    std::string wait_error;
    if (!args->event->completion->AwaitWithTimeout(&wait_error)) {
        return MakeError(wait_error, PJRT_Error_Code_DEADLINE_EXCEEDED);
    }
    const std::string error = args->event->completion->Error();
    if (!error.empty()) {
        return MakeError(error);
    }
    return nullptr;
}

PJRT_Error* MPS_Event_Create(PJRT_Event_Create_Args* args) {
    auto* event = new PJRT_Event();
    event->completion = jax_metallib::MakePendingCompletionEvent();
    args->event = event;
    return nullptr;
}

PJRT_Error* MPS_Event_Set(PJRT_Event_Set_Args* args) {
    if (!args->event || !args->event->completion) {
        return MakeError("Event_Set: null event", PJRT_Error_Code_INVALID_ARGUMENT);
    }
    if (args->error_code == PJRT_Error_Code_OK) {
        jax_metallib::MarkCompletionEventReady(args->event->completion);
    } else {
        std::string msg(args->error_message, args->error_message_size);
        jax_metallib::MarkCompletionEventError(args->event->completion, std::move(msg));
    }
    return nullptr;
}

PJRT_Error* MPS_Event_OnReady(PJRT_Event_OnReady_Args* args) {
    if (!args->callback) {
        return nullptr;
    }
    if (!args->event || !args->event->completion) {
        args->callback(nullptr, args->user_arg);
        return nullptr;
    }

    auto completion = args->event->completion;
    auto callback = args->callback;
    void* user_arg = args->user_arg;
    completion->AddReadyCallback([completion, callback, user_arg] {
        const std::string error = completion->Error();
        callback(error.empty() ? nullptr : MakeError(error), user_arg);
    });
    return nullptr;
}
