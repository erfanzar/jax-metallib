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

#import "pjrt_plugin/runtime/custom_metal_kernel.h"

#import <Foundation/Foundation.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <mutex>
#include <tuple>
#include <utility>

namespace jax_metallib {
namespace {

bool Fail(std::string* error, const std::string& message) {
    if (error)
        *error = "Custom Metal kernel: " + message;
    return false;
}

std::string Description(NSString* text) {
    const char* utf8 = [text UTF8String];
    return utf8 ? utf8 : "no diagnostic available";
}

// Adopt +1 objects under MRC; ARC provides the same lifetime automatically.
template <typename T> struct Owned {
    T value = nil;
    Owned() = default;
    explicit Owned(T object) : value(object) {}
    Owned(const Owned&) = delete;
    Owned& operator=(const Owned&) = delete;
    ~Owned() {
#if !__has_feature(objc_arc)
        [value release];
#endif
    }
};

template <typename T> T Retain(T object) {
#if __has_feature(objc_arc)
    return object;
#else
    return [object retain];
#endif
}

struct Constant {
    NSUInteger index;
    std::vector<unsigned char> bytes;
};

struct Config {
    std::string source, entry;
    MTLSize threads{}, group{};
    std::vector<Constant> constants;
};

bool Keys(id object, NSArray<NSString*>* keys, std::string* error, const char* name) {
    if (![object isKindOfClass:[NSDictionary class]] || [object count] != keys.count)
        return Fail(error, std::string(name) + " has unknown or missing fields");
    for (NSString* key in keys)
        if (![object objectForKey:key])
            return Fail(error, std::string(name) + " has unknown or missing fields");
    return true;
}

bool Integer(id object, uint32_t* value, std::string* error, const char* name) {
    if (![object isKindOfClass:[NSNumber class]] ||
        CFGetTypeID((__bridge CFTypeRef)object) == CFBooleanGetTypeID())
        return Fail(error, std::string(name) + " must be an unsigned integer");
    const char* type = [object objCType];
    if (std::strlen(type) != 1 || !std::strchr("cCsSiIlLqQ", type[0]) ||
        [object compare:@0] == NSOrderedAscending ||
        [object compare:@(std::numeric_limits<uint32_t>::max())] == NSOrderedDescending)
        return Fail(error, std::string(name) + " must fit uint32 without fractional values");
    *value = [object unsignedIntValue];
    return true;
}

bool Dimensions(id object, bool positive, MTLSize* size, std::string* error, const char* name) {
    if (![object isKindOfClass:[NSArray class]] || [object count] != 3)
        return Fail(error, std::string(name) + " must contain exactly three integer dimensions");
    uint32_t d[3];
    for (size_t i = 0; i < 3; ++i) {
        if (!Integer([object objectAtIndex:i], &d[i], error, name))
            return false;
        if (positive && d[i] == 0)
            return Fail(error, std::string(name) + " dimensions must be positive");
    }
    size_t product = 1;
    if (d[0] && d[1] && d[2]) {
        for (uint32_t n : d) {
            if (product > std::numeric_limits<size_t>::max() / n)
                return Fail(error, std::string(name) + " dimension product overflows");
            product *= n;
        }
    }
    *size = MTLSizeMake(d[0], d[1], d[2]);
    return true;
}

int HexDigit(unichar c) {
    if (c >= '0' && c <= '9')
        return c - '0';
    if (c >= 'a' && c <= 'f')
        return c - 'a' + 10;
    if (c >= 'A' && c <= 'F')
        return c - 'A' + 10;
    return -1;
}

bool Parse(const std::string& json, size_t buffer_count, Config* config, std::string* error) {
    NSError* diagnostic = nil;
    NSData* data = [NSData dataWithBytes:json.data() length:json.size()];
    id object = [NSJSONSerialization JSONObjectWithData:data options:0 error:&diagnostic];
    if (!object)
        return Fail(error, "invalid backend_config JSON: " + Description(diagnostic.localizedDescription));
    if (!Keys(object, @[@"version", @"source", @"entry_point", @"threads", @"threadgroup", @"constants"],
              error, "backend_config"))
        return false;
    NSDictionary* root = object;
    uint32_t version;
    if (!Integer(root[@"version"], &version, error, "version"))
        return false;
    if (version != 1)
        return Fail(error, "unsupported backend_config version (expected 1)");
    for (NSString* key in @[@"source", @"entry_point"]) {
        id text = root[key];
        if (![text isKindOfClass:[NSString class]] || [text length] == 0)
            return Fail(error, Description(key) + " must be a nonempty string");
        NSData* utf8 = [text dataUsingEncoding:NSUTF8StringEncoding allowLossyConversion:NO];
        if (!utf8 || std::memchr(utf8.bytes, 0, utf8.length))
            return Fail(error, Description(key) + " must be valid UTF-8 without NUL bytes");
        ([key isEqualToString:@"source"] ? config->source : config->entry)
            .assign(static_cast<const char*>(utf8.bytes), utf8.length);
    }
    auto letter = [](unsigned char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
    };
    if (!letter(config->entry[0]))
        return Fail(error, "entry_point must be an identifier");
    for (unsigned char c : config->entry)
        if (!letter(c) && !(c >= '0' && c <= '9'))
            return Fail(error, "entry_point must be an identifier");
    if (!Dimensions(root[@"threads"], false, &config->threads, error, "threads") ||
        !Dimensions(root[@"threadgroup"], true, &config->group, error, "threadgroup"))
        return false;
    id constants = root[@"constants"];
    if (![constants isKindOfClass:[NSArray class]] || [constants count] > 31 - buffer_count)
        return Fail(error, "constants must be an array of unique static buffer bindings below 31");
    std::array<bool, 31> seen{};
    for (id item in constants) {
        if (!Keys(item, @[@"index", @"hex"], error, "constant"))
            return false;
        uint32_t index;
        if (!Integer(item[@"index"], &index, error, "constant index"))
            return false;
        if (index < buffer_count || index >= 31 || seen[index])
            return Fail(error, "constant binding must be unique, below 31, and not overlap array buffers");
        seen[index] = true;
        id hex = item[@"hex"];
        if (![hex isKindOfClass:[NSString class]] || [hex length] == 0 || [hex length] > 8192 ||
            [hex length] % 2)
            return Fail(error, "constant hex must encode 1 to 4096 bytes");
        Constant constant{index, {}};
        constant.bytes.reserve([hex length] / 2);
        for (NSUInteger i = 0; i < [hex length]; i += 2) {
            int hi = HexDigit([hex characterAtIndex:i]), lo = HexDigit([hex characterAtIndex:i + 1]);
            if (hi < 0 || lo < 0)
                return Fail(error, "constant hex contains a non-hexadecimal character");
            constant.bytes.push_back(static_cast<unsigned char>((hi << 4) | lo));
        }
        config->constants.push_back(std::move(constant));
    }
    return true;
}

struct Pipeline {
    Owned<id<MTLDevice>> device;
    Owned<id<MTLLibrary>> library;
    Owned<id<MTLFunction>> function;
    Owned<id<MTLComputePipelineState>> state;
    Owned<MTLComputePipelineReflection*> reflection;
};

std::shared_ptr<Pipeline> GetPipeline(id<MTLDevice> device, const Config& config, std::string* error) {
    using Key = std::tuple<uintptr_t, std::string, std::string>;
    static std::mutex mutex;
    static std::map<Key, std::weak_ptr<Pipeline>> cache;
    // Serializing misses also prevents concurrent duplicate Metal compilations.
    std::lock_guard<std::mutex> lock(mutex);
    Key key{reinterpret_cast<uintptr_t>((__bridge void*)device), config.source, config.entry};
    auto found = cache.find(key);
    if (found != cache.end())
        if (auto pipeline = found->second.lock())
            return pipeline;
    auto pipeline = std::make_shared<Pipeline>();
    pipeline->device.value = Retain(device);
    Owned<NSString*> source([[NSString alloc] initWithBytes:config.source.data()
                                                     length:config.source.size()
                                                   encoding:NSUTF8StringEncoding]);
    Owned<NSString*> entry([[NSString alloc] initWithBytes:config.entry.data()
                                                    length:config.entry.size()
                                                  encoding:NSUTF8StringEncoding]);
    Owned<MTLCompileOptions*> options([[MTLCompileOptions alloc] init]);
    options.value.mathMode = MTLMathModeSafe;
    NSError* diagnostic = nil;
    pipeline->library.value = [device newLibraryWithSource:source.value
                                                   options:options.value
                                                     error:&diagnostic];
    if (!pipeline->library.value) {
        Fail(error, "MSL compilation failed: " + Description(diagnostic.localizedDescription));
        return nullptr;
    }
    pipeline->function.value = [pipeline->library.value newFunctionWithName:entry.value];
    if (!pipeline->function.value || pipeline->function.value.functionType != MTLFunctionTypeKernel) {
        Fail(error, "entry_point '" + config.entry + "' is not a Metal kernel function");
        return nullptr;
    }
    MTLComputePipelineReflection* reflection = nil;
    pipeline->state.value = [device
        newComputePipelineStateWithFunction:pipeline->function.value
                                    options:MTLPipelineOptionBindingInfo | MTLPipelineOptionBufferTypeInfo
                                 reflection:&reflection
                                      error:&diagnostic];
    pipeline->reflection.value = Retain(reflection);
    if (!pipeline->state.value || !pipeline->reflection.value) {
        Fail(error,
             "Metal pipeline compilation/reflection failed: " + Description(diagnostic.localizedDescription));
        return nullptr;
    }
    for (auto it = cache.begin(); it != cache.end();)
        if (it->second.expired())
            it = cache.erase(it);
        else
            ++it;
    if (cache.size() >= 64)
        cache.erase(cache.begin());
    cache[std::move(key)] = pipeline;
    return pipeline;
}

bool Validate(const Pipeline& pipeline, const Config& config, size_t input_count, size_t buffer_count,
              std::array<NSUInteger, 31>* minimum_sizes, std::string* error) {
    MTLSize limit = pipeline.device.value.maxThreadsPerThreadgroup;
    if (config.group.width > limit.width || config.group.height > limit.height ||
        config.group.depth > limit.depth ||
        config.group.width * config.group.height * config.group.depth >
            pipeline.state.value.maxTotalThreadsPerThreadgroup)
        return Fail(error, "threadgroup exceeds device or pipeline thread limits");
    if (pipeline.state.value.staticThreadgroupMemoryLength > pipeline.device.value.maxThreadgroupMemoryLength)
        return Fail(error, "kernel exceeds device threadgroup memory limit");
    for (id<MTLBinding> binding in pipeline.reflection.value.bindings) {
        if (!binding.isUsed)
            continue;
        if (binding.type != MTLBindingTypeBuffer)
            return Fail(error, "active textures, samplers, threadgroup memory, or other non-buffer bindings "
                               "are unsupported");
        id<MTLBufferBinding> argument = (id<MTLBufferBinding>)binding;
        if (argument.bufferDataType == MTLDataTypePointer &&
            argument.bufferPointerType.elementIsArgumentBuffer)
            return Fail(error, "argument buffers with indirect resource bindings are unsupported");
        NSUInteger index = argument.index;
        if (index >= 31)
            return Fail(error, "active buffer binding must be below 31");
        if (index < buffer_count) {
            bool input = index < input_count;
            if (input && argument.access != MTLBindingAccessReadOnly)
                return Fail(error, "input buffer binding " + std::to_string(index) + " must be read-only");
            if (!input && argument.access == MTLBindingAccessReadOnly)
                return Fail(error, "output buffer binding " + std::to_string(index) + " must be writable");
            (*minimum_sizes)[index] = argument.bufferDataSize;
        } else {
            const Constant* constant = nullptr;
            for (const auto& candidate : config.constants)
                if (candidate.index == index)
                    constant = &candidate;
            if (!constant)
                return Fail(error, "active buffer binding " + std::to_string(index) + " is unbound");
            if (argument.access != MTLBindingAccessReadOnly)
                return Fail(error, "constant buffer binding " + std::to_string(index) + " must be read-only");
            if (constant->bytes.size() < argument.bufferDataSize)
                return Fail(error,
                            "constant buffer binding " + std::to_string(index) + " has insufficient bytes");
        }
    }
    return true;
}

} // namespace

struct CustomMetalKernel::Impl {
    Config config;
    std::shared_ptr<Pipeline> pipeline;
    size_t input_count = 0, buffer_count = 0;
    std::array<NSUInteger, 31> minimum_sizes{};
};

CustomMetalKernel::CustomMetalKernel(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
CustomMetalKernel::~CustomMetalKernel() = default;

std::shared_ptr<CustomMetalKernel> CustomMetalKernel::Compile(id<MTLDevice> device,
                                                              const std::string& backend_config,
                                                              size_t input_count, size_t output_count,
                                                              std::string* error) {
    if (error)
        error->clear();
    @autoreleasepool {
        try {
            @try {
                if (!device || !output_count || input_count > 31 || output_count > 31 - input_count) {
                    Fail(
                        error,
                        "requires a Metal device, at least one output, and at most 31 array buffer bindings");
                    return nullptr;
                }
                auto impl = std::make_unique<Impl>();
                impl->input_count = input_count;
                impl->buffer_count = input_count + output_count;
                if (!Parse(backend_config, impl->buffer_count, &impl->config, error))
                    return nullptr;
                impl->pipeline = GetPipeline(device, impl->config, error);
                if (!impl->pipeline || !Validate(*impl->pipeline, impl->config, input_count,
                                                 impl->buffer_count, &impl->minimum_sizes, error))
                    return nullptr;
                return std::shared_ptr<CustomMetalKernel>(new CustomMetalKernel(std::move(impl)));
            } @catch (NSException* exception) {
                Fail(error, "Metal compilation exception: " + Description(exception.reason));
            }
        } catch (const std::exception& exception) {
            Fail(error, "compilation failed: " + std::string(exception.what()));
        } catch (...) {
            Fail(error, "compilation failed with an unknown exception");
        }
    }
    return nullptr;
}

bool CustomMetalKernel::Encode(id<MTLComputeCommandEncoder> encoder,
                               const std::vector<id<MTLBuffer>>& buffers, std::string* error) const {
    if (error)
        error->clear();
    @autoreleasepool {
        try {
            @try {
                if (!encoder || encoder.device != impl_->pipeline->device.value)
                    return Fail(error, "encoder must belong to the compiled Metal device");
                if (buffers.size() != impl_->buffer_count)
                    return Fail(error, "array buffer count does not match the compiled signature");
                const auto& config = impl_->config;
                if (!config.threads.width || !config.threads.height || !config.threads.depth)
                    return true;
                for (size_t i = 0; i < buffers.size(); ++i) {
                    if (!buffers[i] || buffers[i].device != impl_->pipeline->device.value)
                        return Fail(error, "array buffer must belong to the compiled Metal device");
                    if (buffers[i].length < impl_->minimum_sizes[i])
                        return Fail(error,
                                    "array buffer binding " + std::to_string(i) + " has insufficient bytes");
                    if (i >= impl_->input_count)
                        for (size_t j = 0; j < i; ++j)
                            if (buffers[i] == buffers[j])
                                return Fail(error, "output buffers must not alias inputs or other outputs");
                }
                [encoder setComputePipelineState:impl_->pipeline->state.value];
                for (size_t i = 0; i < buffers.size(); ++i)
                    [encoder setBuffer:buffers[i] offset:0 atIndex:i];
                for (const auto& constant : config.constants)
                    [encoder setBytes:constant.bytes.data()
                               length:constant.bytes.size()
                              atIndex:constant.index];
                [encoder dispatchThreads:config.threads threadsPerThreadgroup:config.group];
                return true;
            } @catch (NSException* exception) {
                return Fail(error, "Metal encoding exception: " + Description(exception.reason));
            }
        } catch (const std::exception& exception) {
            return Fail(error, "encoding failed: " + std::string(exception.what()));
        } catch (...) {
            return Fail(error, "encoding failed with an unknown exception");
        }
    }
}

} // namespace jax_metallib
