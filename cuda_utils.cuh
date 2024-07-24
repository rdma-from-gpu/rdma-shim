#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#ifndef CUDA_CALL
#define CUDA_CALL(call)                                                        \
    {                                                                          \
        cudaError_t res = (call);                                              \
        if (res != cudaSuccess) {                                              \
            printf("CUDA Error: %s\n", cudaGetErrorString(res));               \
            exit(3);                                                           \
        }                                                                      \
    }
#endif

#ifndef CUDA_CALL_DEV
#define CUDA_CALL_DEV(call)                                                    \
    {                                                                          \
        cudaError_t res = (call);                                              \
        if (res != cudaSuccess) {                                              \
            printf("CUDA Error: %s\n", cudaGetErrorString(res));               \
            asm("exit;");                                                      \
        }                                                                      \
    }
#endif

#ifndef CUDA_HOST_REGISTER_PRINT
#define CUDA_HOST_REGISTER_PRINT(ptr, size, flags, what)                       \
    {                                                                          \
        printf("cudaHostRegister %li B at %p for %s\n", (size), (ptr),         \
               (what));                                                        \
        CUDA_CALL(cudaHostRegister((ptr), (size), (flags)));                   \
    }
#endif

#ifndef NVTX_PUSH_RANGE
#if CUDA_TRACING

#pragma message "CUDA NVTX tracing enabled!"

#include "nvToolsExt.h"
const uint32_t colors[] = {0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff,
                           0xff00ffff, 0xffff0000, 0xffffffff};
const int num_colors = sizeof(colors) / sizeof(uint32_t);

#define NVTX_PUSH_RANGE(name, cid)                                             \
    {                                                                          \
        DLOG(INFO) << "NVTX PUSH" << name;                                     \
        int color_id = cid;                                                    \
        color_id = color_id % num_colors;                                      \
        nvtxEventAttributes_t eventAttrib = {0};                               \
        eventAttrib.version = NVTX_VERSION;                                    \
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;                      \
        eventAttrib.colorType = NVTX_COLOR_ARGB;                               \
        eventAttrib.color = colors[color_id];                                  \
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;                     \
        eventAttrib.message.ascii = name;                                      \
        nvtxRangePushEx(&eventAttrib);                                         \
    }
#define NVTX_POP_RANGE() nvtxRangePop();

#define NVTX_EVENT(name, cid)                                                  \
    {                                                                          \
        DLOG(INFO) << "NVTX EVENT" << name;                                    \
        int color_id = cid;                                                    \
        color_id = color_id % num_colors;                                      \
        nvtxEventAttributes_t eventAttrib = {0};                               \
        eventAttrib.version = NVTX_VERSION;                                    \
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;                      \
        eventAttrib.colorType = NVTX_COLOR_ARGB;                               \
        eventAttrib.color = colors[color_id];                                  \
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;                     \
        eventAttrib.message.ascii = name;                                      \
        nvtxMarkEx(&eventAttrib);                                              \
    }
#define NVTX_EVENT_UINT64(name, payload_, cid)                                 \
    {                                                                          \
        DLOG(INFO) << "NVTX EVENT" << name << "->" << (payload_);              \
        int color_id = cid;                                                    \
        color_id = color_id % num_colors;                                      \
        nvtxEventAttributes_t eventAttrib = {0};                               \
        eventAttrib.version = NVTX_VERSION;                                    \
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;                      \
        eventAttrib.colorType = NVTX_COLOR_ARGB;                               \
        eventAttrib.color = colors[color_id];                                  \
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;                     \
        eventAttrib.message.ascii = name;                                      \
        eventAttrib.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;            \
        eventAttrib.payload.llValue = (payload_);                              \
        nvtxMarkEx(&eventAttrib);                                              \
    }

// Add both range and event. ranges are tricky when multi-threading!
// Also takes an ID to append to the name
#define NVTX_PUSH_RANGE_EVENT(name, id, cid)                                   \
    {                                                                          \
        std::string label = name;                                              \
        label += std::to_string(id);                                           \
        /*NVTX_PUSH_RANGE(label, cid);*/                                       \
        NVTX_EVENT(label.c_str(), cid);                                        \
    }
#else

#define NVTX_PUSH_RANGE(name, cid)
#define NVTX_PUSH_RANGE_EVENT(name, id, cid)
#define NVTX_POP_RANGE()
#define NVTX_EVENT(name, cid)
#define NVTX_EVENT_UINT64(name, payload_, cid)

#endif //CUDA_TRACING
#endif //ndef NVTX_PUSH_RANGE

__host__ void register_cuda_areas(struct rdma_shim_data *data);
__host__ void register_cuda_driver_data(void * driver_data, size_t driver_data_size);

#endif
