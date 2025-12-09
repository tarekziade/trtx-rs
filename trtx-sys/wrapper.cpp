#include "wrapper.hpp"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cstring>
#include <exception>
#include <memory>

// Helper to copy error messages
static void copy_error(const char* msg, char* error_msg, size_t error_msg_len) {
    if (error_msg && error_msg_len > 0) {
        strncpy(error_msg, msg, error_msg_len - 1);
        error_msg[error_msg_len - 1] = '\0';
    }
}

// Helper macro for exception handling
#define TRTX_TRY_CATCH_BEGIN try {
#define TRTX_TRY_CATCH_END(error_msg, error_msg_len) \
    } catch (const std::bad_alloc& e) { \
        copy_error(e.what(), error_msg, error_msg_len); \
        return TRTX_ERROR_OUT_OF_MEMORY; \
    } catch (const std::exception& e) { \
        copy_error(e.what(), error_msg, error_msg_len); \
        return TRTX_ERROR_RUNTIME_ERROR; \
    } catch (...) { \
        copy_error("Unknown error", error_msg, error_msg_len); \
        return TRTX_ERROR_UNKNOWN; \
    }

// Logger wrapper that calls back into Rust
class LoggerImpl : public nvinfer1::ILogger {
public:
    LoggerImpl(TrtxLoggerCallback callback, void* user_data)
        : callback_(callback), user_data_(user_data) {}

    void log(Severity severity, const char* msg) noexcept override {
        if (callback_) {
            callback_(user_data_, static_cast<TrtxLoggerSeverity>(severity), msg);
        }
    }

private:
    TrtxLoggerCallback callback_;
    void* user_data_;
};

// Logger functions
int32_t trtx_logger_create(
    TrtxLoggerCallback callback,
    void* user_data,
    TrtxLogger** out_logger,
    char* error_msg,
    size_t error_msg_len
) {
    if (!callback || !out_logger) {
        copy_error("Invalid arguments", error_msg, error_msg_len);
        return TRTX_ERROR_INVALID_ARGUMENT;
    }

    TRTX_TRY_CATCH_BEGIN
        auto logger = new LoggerImpl(callback, user_data);
        *out_logger = reinterpret_cast<TrtxLogger*>(logger);
        return TRTX_SUCCESS;
    TRTX_TRY_CATCH_END(error_msg, error_msg_len)
}

void trtx_logger_destroy(TrtxLogger* logger) {
    if (logger) {
        delete reinterpret_cast<LoggerImpl*>(logger);
    }
}

// Builder functions
int32_t trtx_builder_create(
    TrtxLogger* logger,
    TrtxBuilder** out_builder,
    char* error_msg,
    size_t error_msg_len
) {
    if (!logger || !out_builder) {
        copy_error("Invalid arguments", error_msg, error_msg_len);
        return TRTX_ERROR_INVALID_ARGUMENT;
    }

    TRTX_TRY_CATCH_BEGIN
        auto* logger_impl = reinterpret_cast<LoggerImpl*>(logger);
        auto* builder = nvinfer1::createInferBuilder(*logger_impl);
        if (!builder) {
            copy_error("Failed to create builder", error_msg, error_msg_len);
            return TRTX_ERROR_RUNTIME_ERROR;
        }
        *out_builder = reinterpret_cast<TrtxBuilder*>(builder);
        return TRTX_SUCCESS;
    TRTX_TRY_CATCH_END(error_msg, error_msg_len)
}

void trtx_builder_destroy(TrtxBuilder* builder) {
    if (builder) {
        auto* impl = reinterpret_cast<nvinfer1::IBuilder*>(builder);
        delete impl;
    }
}

int32_t trtx_builder_create_network(
    TrtxBuilder* builder,
    uint32_t flags,
    TrtxNetworkDefinition** out_network,
    char* error_msg,
    size_t error_msg_len
) {
    if (!builder || !out_network) {
        copy_error("Invalid arguments", error_msg, error_msg_len);
        return TRTX_ERROR_INVALID_ARGUMENT;
    }

    TRTX_TRY_CATCH_BEGIN
        auto* builder_impl = reinterpret_cast<nvinfer1::IBuilder*>(builder);
        auto* network = builder_impl->createNetworkV2(flags);
        if (!network) {
            copy_error("Failed to create network", error_msg, error_msg_len);
            return TRTX_ERROR_RUNTIME_ERROR;
        }
        *out_network = reinterpret_cast<TrtxNetworkDefinition*>(network);
        return TRTX_SUCCESS;
    TRTX_TRY_CATCH_END(error_msg, error_msg_len)
}

int32_t trtx_builder_create_builder_config(
    TrtxBuilder* builder,
    TrtxBuilderConfig** out_config,
    char* error_msg,
    size_t error_msg_len
) {
    if (!builder || !out_config) {
        copy_error("Invalid arguments", error_msg, error_msg_len);
        return TRTX_ERROR_INVALID_ARGUMENT;
    }

    TRTX_TRY_CATCH_BEGIN
        auto* builder_impl = reinterpret_cast<nvinfer1::IBuilder*>(builder);
        auto* config = builder_impl->createBuilderConfig();
        if (!config) {
            copy_error("Failed to create builder config", error_msg, error_msg_len);
            return TRTX_ERROR_RUNTIME_ERROR;
        }
        *out_config = reinterpret_cast<TrtxBuilderConfig*>(config);
        return TRTX_SUCCESS;
    TRTX_TRY_CATCH_END(error_msg, error_msg_len)
}

int32_t trtx_builder_build_serialized_network(
    TrtxBuilder* builder,
    TrtxNetworkDefinition* network,
    TrtxBuilderConfig* config,
    void** out_data,
    size_t* out_size,
    char* error_msg,
    size_t error_msg_len
) {
    if (!builder || !network || !config || !out_data || !out_size) {
        copy_error("Invalid arguments", error_msg, error_msg_len);
        return TRTX_ERROR_INVALID_ARGUMENT;
    }

    TRTX_TRY_CATCH_BEGIN
        auto* builder_impl = reinterpret_cast<nvinfer1::IBuilder*>(builder);
        auto* network_impl = reinterpret_cast<nvinfer1::INetworkDefinition*>(network);
        auto* config_impl = reinterpret_cast<nvinfer1::IBuilderConfig*>(config);

        auto* serialized = builder_impl->buildSerializedNetwork(*network_impl, *config_impl);
        if (!serialized) {
            copy_error("Failed to build serialized network", error_msg, error_msg_len);
            return TRTX_ERROR_RUNTIME_ERROR;
        }

        // Copy data to heap so Rust can manage it
        size_t size = serialized->size();
        void* data = malloc(size);
        if (!data) {
            delete serialized;
            copy_error("Failed to allocate memory", error_msg, error_msg_len);
            return TRTX_ERROR_OUT_OF_MEMORY;
        }

        memcpy(data, serialized->data(), size);
        delete serialized;

        *out_data = data;
        *out_size = size;
        return TRTX_SUCCESS;
    TRTX_TRY_CATCH_END(error_msg, error_msg_len)
}

// BuilderConfig functions
void trtx_builder_config_destroy(TrtxBuilderConfig* config) {
    if (config) {
        auto* impl = reinterpret_cast<nvinfer1::IBuilderConfig*>(config);
        delete impl;
    }
}

int32_t trtx_builder_config_set_memory_pool_limit(
    TrtxBuilderConfig* config,
    int32_t pool_type,
    size_t pool_size,
    char* error_msg,
    size_t error_msg_len
) {
    if (!config) {
        copy_error("Invalid arguments", error_msg, error_msg_len);
        return TRTX_ERROR_INVALID_ARGUMENT;
    }

    TRTX_TRY_CATCH_BEGIN
        auto* config_impl = reinterpret_cast<nvinfer1::IBuilderConfig*>(config);
        config_impl->setMemoryPoolLimit(
            static_cast<nvinfer1::MemoryPoolType>(pool_type),
            pool_size
        );
        return TRTX_SUCCESS;
    TRTX_TRY_CATCH_END(error_msg, error_msg_len)
}

// NetworkDefinition functions
void trtx_network_destroy(TrtxNetworkDefinition* network) {
    if (network) {
        auto* impl = reinterpret_cast<nvinfer1::INetworkDefinition*>(network);
        delete impl;
    }
}

// Runtime functions
int32_t trtx_runtime_create(
    TrtxLogger* logger,
    TrtxRuntime** out_runtime,
    char* error_msg,
    size_t error_msg_len
) {
    if (!logger || !out_runtime) {
        copy_error("Invalid arguments", error_msg, error_msg_len);
        return TRTX_ERROR_INVALID_ARGUMENT;
    }

    TRTX_TRY_CATCH_BEGIN
        auto* logger_impl = reinterpret_cast<LoggerImpl*>(logger);
        auto* runtime = nvinfer1::createInferRuntime(*logger_impl);
        if (!runtime) {
            copy_error("Failed to create runtime", error_msg, error_msg_len);
            return TRTX_ERROR_RUNTIME_ERROR;
        }
        *out_runtime = reinterpret_cast<TrtxRuntime*>(runtime);
        return TRTX_SUCCESS;
    TRTX_TRY_CATCH_END(error_msg, error_msg_len)
}

void trtx_runtime_destroy(TrtxRuntime* runtime) {
    if (runtime) {
        auto* impl = reinterpret_cast<nvinfer1::IRuntime*>(runtime);
        delete impl;
    }
}

int32_t trtx_runtime_deserialize_cuda_engine(
    TrtxRuntime* runtime,
    const void* data,
    size_t size,
    TrtxCudaEngine** out_engine,
    char* error_msg,
    size_t error_msg_len
) {
    if (!runtime || !data || !out_engine) {
        copy_error("Invalid arguments", error_msg, error_msg_len);
        return TRTX_ERROR_INVALID_ARGUMENT;
    }

    TRTX_TRY_CATCH_BEGIN
        auto* runtime_impl = reinterpret_cast<nvinfer1::IRuntime*>(runtime);
        auto* engine = runtime_impl->deserializeCudaEngine(data, size);
        if (!engine) {
            copy_error("Failed to deserialize engine", error_msg, error_msg_len);
            return TRTX_ERROR_RUNTIME_ERROR;
        }
        *out_engine = reinterpret_cast<TrtxCudaEngine*>(engine);
        return TRTX_SUCCESS;
    TRTX_TRY_CATCH_END(error_msg, error_msg_len)
}

// CudaEngine functions
void trtx_cuda_engine_destroy(TrtxCudaEngine* engine) {
    if (engine) {
        auto* impl = reinterpret_cast<nvinfer1::ICudaEngine*>(engine);
        delete impl;
    }
}

int32_t trtx_cuda_engine_create_execution_context(
    TrtxCudaEngine* engine,
    TrtxExecutionContext** out_context,
    char* error_msg,
    size_t error_msg_len
) {
    if (!engine || !out_context) {
        copy_error("Invalid arguments", error_msg, error_msg_len);
        return TRTX_ERROR_INVALID_ARGUMENT;
    }

    TRTX_TRY_CATCH_BEGIN
        auto* engine_impl = reinterpret_cast<nvinfer1::ICudaEngine*>(engine);
        auto* context = engine_impl->createExecutionContext();
        if (!context) {
            copy_error("Failed to create execution context", error_msg, error_msg_len);
            return TRTX_ERROR_RUNTIME_ERROR;
        }
        *out_context = reinterpret_cast<TrtxExecutionContext*>(context);
        return TRTX_SUCCESS;
    TRTX_TRY_CATCH_END(error_msg, error_msg_len)
}

int32_t trtx_cuda_engine_get_tensor_name(
    TrtxCudaEngine* engine,
    int32_t index,
    const char** out_name,
    char* error_msg,
    size_t error_msg_len
) {
    if (!engine || !out_name) {
        copy_error("Invalid arguments", error_msg, error_msg_len);
        return TRTX_ERROR_INVALID_ARGUMENT;
    }

    TRTX_TRY_CATCH_BEGIN
        auto* engine_impl = reinterpret_cast<nvinfer1::ICudaEngine*>(engine);
        const char* name = engine_impl->getIOTensorName(index);
        if (!name) {
            copy_error("Invalid tensor index", error_msg, error_msg_len);
            return TRTX_ERROR_INVALID_ARGUMENT;
        }
        *out_name = name;
        return TRTX_SUCCESS;
    TRTX_TRY_CATCH_END(error_msg, error_msg_len)
}

int32_t trtx_cuda_engine_get_nb_io_tensors(
    TrtxCudaEngine* engine,
    int32_t* out_count
) {
    if (!engine || !out_count) {
        return TRTX_ERROR_INVALID_ARGUMENT;
    }

    TRTX_TRY_CATCH_BEGIN
        auto* engine_impl = reinterpret_cast<nvinfer1::ICudaEngine*>(engine);
        *out_count = engine_impl->getNbIOTensors();
        return TRTX_SUCCESS;
    TRTX_TRY_CATCH_END(nullptr, 0)
}

// ExecutionContext functions
void trtx_execution_context_destroy(TrtxExecutionContext* context) {
    if (context) {
        auto* impl = reinterpret_cast<nvinfer1::IExecutionContext*>(context);
        delete impl;
    }
}

int32_t trtx_execution_context_set_tensor_address(
    TrtxExecutionContext* context,
    const char* tensor_name,
    void* data,
    char* error_msg,
    size_t error_msg_len
) {
    if (!context || !tensor_name) {
        copy_error("Invalid arguments", error_msg, error_msg_len);
        return TRTX_ERROR_INVALID_ARGUMENT;
    }

    TRTX_TRY_CATCH_BEGIN
        auto* context_impl = reinterpret_cast<nvinfer1::IExecutionContext*>(context);
        bool success = context_impl->setTensorAddress(tensor_name, data);
        if (!success) {
            copy_error("Failed to set tensor address", error_msg, error_msg_len);
            return TRTX_ERROR_RUNTIME_ERROR;
        }
        return TRTX_SUCCESS;
    TRTX_TRY_CATCH_END(error_msg, error_msg_len)
}

int32_t trtx_execution_context_enqueue_v3(
    TrtxExecutionContext* context,
    void* cuda_stream,
    char* error_msg,
    size_t error_msg_len
) {
    if (!context) {
        copy_error("Invalid arguments", error_msg, error_msg_len);
        return TRTX_ERROR_INVALID_ARGUMENT;
    }

    TRTX_TRY_CATCH_BEGIN
        auto* context_impl = reinterpret_cast<nvinfer1::IExecutionContext*>(context);
        bool success = context_impl->enqueueV3(static_cast<cudaStream_t>(cuda_stream));
        if (!success) {
            copy_error("Failed to enqueue inference", error_msg, error_msg_len);
            return TRTX_ERROR_RUNTIME_ERROR;
        }
        return TRTX_SUCCESS;
    TRTX_TRY_CATCH_END(error_msg, error_msg_len)
}

// Utility functions
void trtx_free_buffer(void* buffer) {
    free(buffer);
}
