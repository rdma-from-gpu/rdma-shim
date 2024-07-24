#include <util/udma_barrier.h>
#include "rdma_shim.cuh"
#include "infiniband/mlx5dv.h"


#include "cuda_memory.cuh"
extern "C" {
#include "rdma_shim.h"
//#include "providers/mlx5/mlx5.h"
//#include "rdma_shim.h"
// #include "rdma.h"
// #include "rdma-core/providers/mlx5/mlx5.h"
//#include <providers/mlx5/mlx5dv.h>
// #include <providers/mlx5/mlx5.h>
//#include <providers/mlx5/wqe.h>
}
//__forceinline__
__device__ __host__ struct mlx5_wqe_ctrl_seg *
get_ctrl_seg_ptr_cu(struct rdma_shim_data *data, int idx) {
    return (struct mlx5_wqe_ctrl_seg *)((uintptr_t)(*data->sq_start) +
                                        (idx << MLX5_SEND_WQE_SHIFT));
}

// The following function can likely be optimized (at least on the CUDA side)
// by borrowing implementation from nvshmem
__device__ __host__ static inline void
mlx5_bf_copy_cu(struct rdma_shim_data *data, int bytes, void *ctrl) {
#ifdef __CUDA_ARCH__
    uint64_t *dst = (uint64_t *)((uint64_t)(*data->mbf_reg) +
                                 (uint64_t)(*data->mbf_offset));
    assert(bytes == 64);
    cu_store_release(dst, *(uint64_t *)ctrl);
#else
    // On CPU, rely on default rdma-core implementation
    // Which should take care of the mmio open and flush too
    mlx5_bf_copy3(data, bytes, ctrl);
#endif
}

// Drin! These functions should be replaced with more efficient versions
// Maybe with different implemantation CPU/GPU
// e.g. CPU should come from rdma-core and GPU from nvshmem.
// We are already relying  on this distinction for mlx5_bf_copy_cu
__device__ __host__ __inline__ static void
write_doorbell(struct rdma_shim_data *data) {
    // printf("Write to %p cur_post %i\n",
    //         &(*data->db)[MLX5_SND_DBR],
    //         HTOBE32(*(data->cur_post) & 0xffff));

#ifdef __CUDA_ARCH__
    //(*data->db)[MLX5_SND_DBR] = HTOBE32(*(data->cur_post) & 0xffff);
    cu_update_dbr(&(*data->db)[MLX5_SND_DBR], *(data->cur_post));
#else
    udma_to_device_barrier();
    (*data->db)[MLX5_SND_DBR] = HTOBE32(*(data->cur_post) & 0xffff);
#endif
}
__device__ __host__ __inline__ static void
set_offset(struct rdma_shim_data *data) {
#ifdef __CUDA_ARCH__
    *data->mbf_offset ^= *data->mbf_buf_size;
#else
    *data->mbf_offset ^= *data->mbf_buf_size;
#endif
}
__device__ __host__ __inline__ static void
write_doorbell_cqe(struct rdma_shim_data *data) {
#ifdef __CUDA_ARCH__
    cu_update_dbr(&(*data->send_dbrec)[SHIM_MLX5_CQ_SET_CI],
                  *(data->cons_index));
#else
    udma_from_device_barrier();
    (*data->send_dbrec)[SHIM_MLX5_CQ_SET_CI] =
        htobe32(*data->cons_index & 0xffffff);
#endif
}

//__global__
__device__ __host__ void
rdma_write_with_imm_cu(struct rdma_shim_data *data, void *buffer,
                       size_t buffer_size, uint32_t buffer_lkey,
                       uint32_t buffer_rkey, void *raddr, int imm,
                       bool signaled) {
    // TODO: handle locks
    // mlx5_lock_qp(mqp); // This is unlikely to work on GPU!

    // This will be a single request
    int nreq = 1;

    struct mlx5_wqe_ctrl_seg ctrl_seg = {0};
    struct mlx5_wqe_ctrl_seg *ctrl_seg_ptr;
    struct mlx5_wqe_raddr_seg raddr_seg = {0};
    struct mlx5_wqe_raddr_seg *raddr_seg_ptr;
    struct mlx5_wqe_data_seg data_seg = {0};
    struct mlx5_wqe_data_seg *data_seg_ptr;

    // Get the index accordingly to where we are in the queue
    unsigned int idx = *data->cur_post & (*data->wqe_cnt - 1);
    // Start to fill the ctrl segment
    int seg_size = sizeof(struct mlx5_wqe_ctrl_seg);
    ctrl_seg_ptr = get_ctrl_seg_ptr_cu(data, idx);
    //*(uint32_t *)((void*)(ctrl_seg_ptr + 8)) = 0;
    ctrl_seg_ptr->signature = 0;
    // ctrl_seg.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE | MLX5_WQE_CTRL_SOLICITED;
    ctrl_seg.fm_ce_se =
        (signaled ? MLX5_WQE_CTRL_CQ_UPDATE | MLX5_WQE_CTRL_SOLICITED : 0);

    ctrl_seg.opmod_idx_opcode =
            HTOBE32(((*data->cur_post & 0xffff) << 8) |
                    MLX5_OPCODE_RDMA_WRITE_IMM | (0 << 24));
    ctrl_seg.imm = imm;

    raddr_seg_ptr =
        (struct mlx5_wqe_raddr_seg *)((uintptr_t)ctrl_seg_ptr + seg_size);
    raddr_seg.raddr = HTOBE64((uint64_t)raddr);
    raddr_seg.rkey = HTOBE32(buffer_rkey);
    raddr_seg.reserved = HTOBE32(0);
    seg_size += sizeof(struct mlx5_wqe_raddr_seg);

    if (buffer_size > 0 && buffer != 0) {
        data_seg_ptr =
            (struct mlx5_wqe_data_seg *)((uintptr_t)ctrl_seg_ptr + seg_size);
        data_seg.addr = HTOBE64((uint64_t)buffer);
        data_seg.byte_count = HTOBE32(buffer_size);
        data_seg.lkey = HTOBE32(buffer_lkey);
        seg_size += sizeof(struct mlx5_wqe_data_seg);
    }

    // Then calculcate sizes and advance pointers in the working queue
    ctrl_seg.qpn_ds = HTOBE32((data->qp->qp_num << 8) | (seg_size / 16));
    (*data->wqe_head)[idx] = (*(data->sq_head)) + nreq;
    (*data->wrid)[idx] = imm;
    *(data->cur_post) =
        *(data->cur_post) + DIV_ROUND_UP(seg_size, MLX5_SEND_WQE_BB);

    // Copy the data structures to the working queue
    memcpy(ctrl_seg_ptr, &ctrl_seg, sizeof(struct mlx5_wqe_ctrl_seg));
    memcpy(raddr_seg_ptr, &raddr_seg, sizeof(struct mlx5_wqe_raddr_seg));
    if (buffer_size > 0 && buffer != 0)
        memcpy(data_seg_ptr, &data_seg, sizeof(struct mlx5_wqe_data_seg));

    (*data->sq_head) += nreq;
    write_doorbell(data);
    mlx5_bf_copy_cu(data, 64, ctrl_seg_ptr);
    set_offset(data);

    // TODO: Handle locks!
    // mlx5_unlock_qp(mqp); // Not on CUDA (?)
}
__global__ void rdma_write_with_imm_kernel(struct rdma_shim_data *data,
                                           void *buffer, size_t size,
                                           uint32_t buffer_lkey,
                                           uint32_t buffer_rkey, void *raddr,
                                           int imm, bool signaled) {
    rdma_write_with_imm_cu(data, buffer, size, buffer_lkey, buffer_rkey, raddr,
                           imm, signaled);
}

__host__ __device__ void consume_cqe_cu(struct rdma_shim_data *data) {
    // TODO: This is a "ignorant" method: it consumes anything.
    // Without actually checking validity nor if there is (or not) 
    // something that should be consumed. It's up to the caller to ensure
    // that in the current implementation
    struct mlx5_cqe64 *cqe64;
    uint64_t offset =
        ((uint32_t)(*data->cons_index) & (uint32_t)(*data->send_cqe)) * 64;

    struct mlx5_buf *active_buf = (*data->active_buf);
    void **buf_buf = (void **)active_buf;
    cqe64 = (struct mlx5_cqe64 *)((uint8_t *)(*buf_buf) + offset);

    // mlx5_spin_lock(&mcq->lock);
    // assert(mcq->cqe_sz == 64); // Just to avoid further complexity
    // assert(!mcq->stall_enable); // Just to avoid further complexity
    // uint8_t opcode = cqe64->op_own >> 4;

    // if (((opcode != MLX5_CQE_INVALID) &&
    //       !((cqe64->op_own & MLX5_CQE_OWNER_MASK) ^
    // 	!!(*data->cons_index & ((uint32_t)(*data->send_cqe)) + 1))))
    // printf("ORRORE\n");
    // else {

    ++(*data->cons_index);
    // exit(1);
    // VALGRIND_MAKE_MEM_DEFINED(cqe64, sizeof *cqe64);

    /*
     * Make sure we read CQ entry contents after we've checked the
     * ownership bit.
     */
    // udma_from_device_barrier();

    // uint16_t wqe_ctr = be16toh(cqe64->wqe_counter);
    uint16_t wqe_ctr = HTOBE16(cqe64->wqe_counter);
    int idx = wqe_ctr & (*data->wqe_cnt - 1);
    *data->sq_tail = (*data->wqe_head)[idx] + 1;
    write_doorbell_cqe(data);
}
__global__ void consume_cqe_kernel(struct rdma_shim_data *data) {
    consume_cqe_cu(data);
}

__host__ void register_cuda_areas(struct rdma_shim_data *data) {
    printf("data is %p, data->mqp is %p\n", data, data->mqp);
    // CUDA_HOST_REGISTER_PRINT(data, sizeof(struct rdma_shim_data),
    //                          cudaHostRegisterMapped |
    //                          cudaHostRegisterPortable, "data");
    // CUDA_HOST_REGISTER_PRINT(data->mqp, sizeof_mlx5_qp(),
    //                          cudaHostRegisterMapped |
    //                          cudaHostRegisterPortable, "mlx5_qp");
    //int size_sq = 16 * 512;
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
    int sq_buf_size = 32768*8; // Always? Maybe not
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
__host__ void register_cuda_driver_data(void *driver_data,
                                        size_t driver_data_size) {
    CUDA_HOST_REGISTER_PRINT((driver_data), driver_data_size,
                             cudaHostRegisterMapped | cudaHostRegisterPortable,
                             "driver_data");
}


__global__ void rdma_write_with_imm_kernel_multiple(struct rdma_shim_data *data,
                                                    void *buffer, size_t size,
                                                    uint32_t buffer_lkey,
                                                    uint32_t buffer_rkey, void *raddr,
                                                    uint64_t abs_idx, // This is how many we have already done
                                                    int n,
                                                    int batch) {

    for(uint64_t i =0; i < n; i++)
    {
        int imm = 0xff | ((abs_idx+i) << 16);
        bool signaled = ((abs_idx+i) % batch) == 0;

        if (signaled && (abs_idx+i) != 0)
            consume_cqe_cu(data);
        rdma_write_with_imm_cu(data, buffer, size, buffer_lkey, buffer_rkey, raddr,
                               imm, signaled);
    }
}

