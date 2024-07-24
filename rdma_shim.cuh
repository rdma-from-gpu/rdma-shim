#ifndef RDMA_SHIM_CUH
#define RDMA_SHIM_CUH
#include <cstdint>

#include "cuda_utils.h"
#include "cuda_runtime.h"
#include "cuda_memory.cuh"
#include "device_launch_parameters.h"



__global__ void rdma_write_with_imm_kernel(struct rdma_shim_data *data,
                                            void * buffer=0, size_t size=0,
                                            uint32_t buffer_lkey = 0,
        uint32_t buffer_rkey=0, void * raddr=0,
                                           int imm=0, bool signaled = false);

__device__ __host__ void rdma_write_with_imm_cu(struct rdma_shim_data *data,
                                                void * buffer, size_t size,
                                                uint32_t buffer_lkey,
        uint32_t buffer_rkey=0, void * raddr=nullptr,
                                                int imm=0, bool signaled = false);

__host__ __device__ void consume_cqe_cu(struct rdma_shim_data *data) ;
__global__ void consume_cqe_kernel(struct rdma_shim_data *data);
__host__ void register_cuda_areas(struct rdma_shim_data *data);
__host__ void register_cuda_driver_data(void * driver_data, size_t driver_data_size);

__global__ void rdma_write_with_imm_kernel_multiple(struct rdma_shim_data *data,
                                           void *buffer, size_t size,
                                           uint32_t buffer_lkey,
                                           uint32_t buffer_rkey, void *raddr,
                                           uint64_t abs_idx,
                                           int n,
                                           int batch);



#ifdef __CUDA_ARCH__
#define HTOBE64(x) BSWAP64(x)
#define HTOBE32(x) BSWAP32(x)
#define HTOBE16(x) BSWAP16(x)

#else
#define HTOBE16(x) htobe16(x)
#define HTOBE32(x) htobe32(x)
#define HTOBE64(x) htobe64(x)

#endif

#ifndef DIV_ROUND_UP
#define DIV_ROUND_UP(n, d) ((n + d - 1) / d)
#endif

#endif
