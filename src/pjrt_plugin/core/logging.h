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

#pragma once
#include <cstdio>

#ifndef MPS_LOG_LEVEL
#define MPS_LOG_LEVEL 1
#endif

#if MPS_LOG_LEVEL >= 0
#define MPS_LOG_ERROR(...)                                                                                   \
    do {                                                                                                     \
        fprintf(stderr, "[MPS ERROR] " __VA_ARGS__);                                                         \
    } while (0)
#else
#define MPS_LOG_ERROR(...)                                                                                   \
    do {                                                                                                     \
    } while (0)
#endif

#if MPS_LOG_LEVEL >= 1
#define MPS_LOG_WARN(...)                                                                                    \
    do {                                                                                                     \
        fprintf(stderr, "[MPS WARN] " __VA_ARGS__);                                                          \
    } while (0)
#else
#define MPS_LOG_WARN(...)                                                                                    \
    do {                                                                                                     \
    } while (0)
#endif

#if MPS_LOG_LEVEL >= 2
#define MPS_LOG_INFO(...)                                                                                    \
    do {                                                                                                     \
        fprintf(stderr, "[MPS INFO] " __VA_ARGS__);                                                          \
    } while (0)
#else
#define MPS_LOG_INFO(...)                                                                                    \
    do {                                                                                                     \
    } while (0)
#endif

#if MPS_LOG_LEVEL >= 3
#define MPS_LOG_DEBUG(...)                                                                                   \
    do {                                                                                                     \
        fprintf(stderr, "[MPS DEBUG] " __VA_ARGS__);                                                         \
    } while (0)
#else
#define MPS_LOG_DEBUG(...)                                                                                   \
    do {                                                                                                     \
    } while (0)
#endif
