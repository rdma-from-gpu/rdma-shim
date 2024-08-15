#include "infiniband/mlx5dv.h"
#include "rdma_shim.cuh"
#include <util/udma_barrier.h>

#include "cuda_memory.cuh"
extern "C" {
#include "rdma_shim.h"
}

// This extracts relevant memory areas, which needs to be registered with CUDA
__host__ void register_cuda_areas(struct rdma_shim_data *data) {
    printf("data is %p, data->mqp is %p\n", data, data->mqp);
    // CUDA_HOST_REGISTER_PRINT(data, sizeof(struct rdma_shim_data),
    //                          cudaHostRegisterMapped |
    //                          cudaHostRegisterPortable, "data");
    // CUDA_HOST_REGISTER_PRINT(data->mqp, sizeof_mlx5_qp(),
    //                          cudaHostRegisterMapped |
    //                          cudaHostRegisterPortable, "mlx5_qp");
    // int size_sq = 16 * 512;
    // We are not completely sure (yet) how big should this be
    int size_sq = 1024 * 512;
    CUDA_HOST_REGISTER_PRINT(*(data->sq_start), size_sq,
                             cudaHostRegisterMapped | cudaHostRegisterPortable,
                             "sq");

    // These may be avoided by asking rdma-core to use the custom allocator
    CUDA_HOST_REGISTER_PRINT(*(data->db), 64,
                             cudaHostRegisterMapped | cudaHostRegisterPortable,
                             "doorbell");
    CUDA_HOST_REGISTER_PRINT(*(data->send_dbrec), 64,
                             cudaHostRegisterMapped | cudaHostRegisterPortable,
                             "doorbell 2");
    CUDA_HOST_REGISTER_PRINT(*(data->mbf_reg), // 2 * 1024,
                             mlx5_bf_reg_size(data),
                             cudaHostRegisterMapped | cudaHostRegisterPortable |
                                 cudaHostRegisterIoMemory,
                             "mlx5_bf_reg");

    // TODO: get the correct size from rdma-core
    // TODO: this can be fixed as above
    int sq_buf_size = 32768 * 8; // Always? Maybe not
    struct mlx5_buf *active_buf = (*data->active_buf);
    // Pointer madness! We need the first member of the active buf.
    // And for simplicity we just assume the struct has not been redefined
    // (please don't!)
    void **buf_buf = (void **)active_buf;
    void *sq_buf = *buf_buf;
    CUDA_HOST_REGISTER_PRINT(sq_buf, sq_buf_size,
                             cudaHostRegisterMapped | cudaHostRegisterPortable,
                             "sq_buf");
}

// This is a generic registration for the "big" data area
__host__ void register_cuda_driver_data(void *driver_data,
                                        size_t driver_data_size) {
    CUDA_HOST_REGISTER_PRINT((driver_data), driver_data_size,
                             cudaHostRegisterMapped | cudaHostRegisterPortable,
                             "driver_data");
}
