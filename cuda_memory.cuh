#ifndef RDMA_CUDA_MEMORY_H
#define RDMA_CUDA_MEMORY_H
#include <linux/types.h>
#include "rdma_shim.cuh"
// Here we have more help functions/macros to operate over CUDA memory
// Mostly from NVSHSMEM
// #define NVSHMEMI_IBGDA_PTX_OPTIMIZATION_STORE_RELEASE 1
// THE ABOVE IS DISABLED TO COMPILE ON TESLA T4


#ifdef __CUDA_ARCH__
#define HTOBE64(x) BSWAP64(x)
#define HTOBE32(x) BSWAP32(x)
#define HTOBE16(x) BSWAP16(x)

#else
#define HTOBE16(x) htobe16(x)
#define HTOBE32(x) htobe32(x)
#define HTOBE64(x) htobe64(x)

#endif

#define NTOH64(x)                                                              \
    *x = ((*(x)&0xFF00000000000000) >> 56 | (*(x)&0x00FF000000000000) >> 40 |  \
          (*(x)&0x0000FF0000000000) >> 24 | (*(x)&0x000000FF00000000) >> 8 |   \
          (*(x)&0x00000000FF000000) << 8 | (*(x)&0x0000000000FF0000) << 24 |   \
          (*(x)&0x000000000000FF00) << 40 gg| (*(x)&0x00000000000000FF) << 56)

#define NTOH32(x)                                                              \
    *x = ((*(x)&0xFF000000) >> 24 | (*(x)&0x00FF0000) >> 8 |                   \
          (*(x)&0x0000FF00) << 8 | (*(x)&0x000000FF) << 24)


__device__ static inline uint64_t BSWAP64(uint64_t x) {
    uint64_t ret;
    asm volatile("{\n\t"
                 ".reg .b32 mask;\n\t"
                 ".reg .b32 ign;\n\t"
                 ".reg .b32 lo;\n\t"
                 ".reg .b32 hi;\n\t"
                 ".reg .b32 new_lo;\n\t"
                 ".reg .b32 new_hi;\n\t"
                 "mov.b32 mask, 0x0123;\n\t"
                 "mov.b64 {lo,hi}, %1;\n\t"
                 "prmt.b32 new_hi, lo, ign, mask;\n\t"
                 "prmt.b32 new_lo, hi, ign, mask;\n\t"
                 "mov.b64 %0, {new_lo,new_hi};\n\t"
                 "}"
                 : "=l"(ret)
                 : "l"(x));
    return ret;
}

__device__ static inline uint32_t BSWAP32(uint32_t x) {
    uint32_t ret;
    asm volatile("{\n\t"
                 ".reg .b32 mask;\n\t"
                 ".reg .b32 ign;\n\t"
                 "mov.b32 mask, 0x0123;\n\t"
                 "prmt.b32 %0, %1, ign, mask;\n\t"
                 "}"
                 : "=r"(ret)
                 : "r"(x));
    return ret;
}

__device__ static inline uint16_t BSWAP16(uint16_t x) {
    uint16_t ret;

    uint32_t a = (uint32_t)x;
    uint32_t d;
    asm volatile("{\n\t"
                 ".reg .b32 mask;\n\t"
                 ".reg .b32 ign;\n\t"
                 "mov.b32 mask, 0x4401;\n\t"
                 "mov.b32 ign, 0x0;\n\t"
                 "prmt.b32 %0, %1, ign, mask;\n\t"
                 "}"
                 : "=r"(d)
                 : "r"(a));
    ret = (uint16_t)d;
    return ret;
}

#ifndef ACCESS_ONCE
#define ACCESS_ONCE(x) (*(volatile typeof(x) *)&(x))
#endif

#ifndef READ_ONCE
#define READ_ONCE(x) ACCESS_ONCE(x)
#endif

#ifndef WRITE_ONCE
#define WRITE_ONCE(x, v) (ACCESS_ONCE(x) = (v))
#endif


template <typename T>
__device__ static inline void cu_store_relaxed(T *ptr, T val) {
    WRITE_ONCE(*ptr, val);
}

template <>
__device__ inline void cu_store_relaxed(uint8_t *ptr, uint8_t val) {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_STORE_RELEASE
    uint16_t _val = val;
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b8 [%0], %1;" : : "l"(ptr), "h"(_val));
#else
    WRITE_ONCE(*ptr, val);
#endif
}

template <>
__device__ inline void cu_store_relaxed(uint16_t *ptr, uint16_t val) {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_STORE_RELEASE
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b16 [%0], %1;" : : "l"(ptr), "h"(val));
#else
    WRITE_ONCE(*ptr, val);
#endif
}

template <>
__device__ inline void cu_store_relaxed(uint32_t *ptr, uint32_t val) {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_STORE_RELEASE
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
#else
    WRITE_ONCE(*ptr, val);
#endif
}

template <>
__device__ inline void cu_store_relaxed(uint64_t *ptr, uint64_t val) {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_STORE_RELEASE
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b64 [%0], %1;" : : "l"(ptr), "l"(val));
#else
    WRITE_ONCE(*ptr, val);
#endif
}
__device__ static inline void cu_store_release(uint32_t *ptr, uint32_t val) {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_STORE_RELEASE
    asm volatile("st.release.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
#else
    WRITE_ONCE(*ptr, val);
#endif
}

__device__ static inline void cu_store_release(uint64_t *ptr, uint64_t val) {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_STORE_RELEASE
    asm volatile("st.release.gpu.global.L1::no_allocate.b64 [%0], %1;" : : "l"(ptr), "l"(val));
#else
    WRITE_ONCE(*ptr, val);
#endif
}
__device__ static inline void cu_update_dbr(__be32 * dbrec_ptr , uint32_t dbrec_head) {
    // DBREC contains the index of the next empty WQEBB.
    __be32 dbrec_val;

    // This is equivalent to
    WRITE_ONCE(dbrec_ptr, (__be32*) HTOBE32(dbrec_head & 0xffff));
    asm volatile(
        "{\n\t"
        ".reg .b32 mask1;\n\t"
        ".reg .b32 dbrec_head_16b;\n\t"
        ".reg .b32 ign;\n\t"
        ".reg .b32 mask2;\n\t"
        "mov.b32 mask1, 0xffff;\n\t"
        "mov.b32 mask2, 0x123;\n\t"
        "and.b32 dbrec_head_16b, %1, mask1;\n\t"
        "prmt.b32 %0, dbrec_head_16b, ign, mask2;\n\t"
        "}"
        : "=r"(dbrec_val)
        : "r"(dbrec_head));
    cu_store_release(dbrec_ptr, dbrec_val);
}

#define ROUND_UP(N, S) ((((N) + (S) - 1) / (S)) * (S))


#endif
