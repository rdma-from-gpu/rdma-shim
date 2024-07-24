#ifndef RDMA_SHIM_H
#define RDMA_SHIM_H



#include <linux/types.h>
#include <stddef.h>
#include <stdint.h>

// From rdma-core providers/mlx5/mlx5.h 
//#define SHIM_MLX5_CQ_SET_CI MLX5_CQ_SET_CI
#define SHIM_MLX5_CQ_SET_CI 0

struct mlx5_qp;
struct mlx5_wq;
//struct mlx5_bf;
struct mlx4_ctx;
struct ibv_qp;
struct ibv_context;
struct mlx5_qp *mqp_from_ibv_qp(struct ibv_context *ctx, struct ibv_qp *qp);
struct mlx5_bf *  mbf_from_mlx5_qp(struct mlx5_qp * qp);
struct mlx5_context * mctx_from_mlx5_qp(struct mlx5_qp * qp);

// This structure is use to keep the status of the network stack
// and it's passed around the different kernels that interact with the NIC
struct rdma_shim_data {
        struct ibv_qp * qp;
        struct mlx5_qp * mqp;
        struct mlx5_cq * send_cq;
        struct mlx5_context * mctx;
        struct mlx5_bf * mbf;
        // uint8_t is_simple_doorbell;
        void ** sq_start;
        void ** wq_end;
        // These are pointers to the actual rdma-core structs
        // We cannot import those on the CUDA side...
        unsigned * cur_post;
        unsigned * wqe_cnt;
        uint64_t ** wrid;
        uint32_t ** wqe_head;
        uint32_t *sq_head;
        uint32_t *sq_tail;
        __be32 ** db;
        __be32 ** send_dbrec;
        void ** mbf_reg;
        unsigned * mbf_offset;
        unsigned * mbf_buf_size;
        uint32_t * cons_index;
        struct mlx5_buf ** active_buf;
        int * send_cqe;


};
void advance_doorbell(struct ibv_qp * qp, int size, void * ctrl);
void advance_doorbell_mqp(struct mlx5_qp * mqp, int size, void * ctrl);
void * sq_buf_buf_from_mlx5_qp(struct mlx5_qp * qp);
void * sq_start_from_mlx5_qp(struct mlx5_qp * qp);
int sq_idx_from_qp(struct mlx5_qp * qp);
void set_wr_stuff(struct mlx5_qp * mqp, int idx, int id, int size, int nreq);
void mlx5_lock_qp(struct mlx5_qp * mqp);
void mlx5_unlock_qp(struct mlx5_qp * mqp);

uint64_t prepare_rdma(
        struct ibv_qp * qp,
        struct rdma_shim_data * data,
        void ** lowest_p, void ** highest_p);
void rdma_write(
        struct rdma_shim_data * data,
        uint64_t laddr, uint64_t raddr, uint32_t lkey, uint32_t rkey, uint32_t bytes);
void rdma_write_with_imm(
        struct rdma_shim_data * data,
        int imm);

void mlx5_bf_copy3(struct rdma_shim_data * data, int bytes, void * ctrl);


// Use these to access the sizes of the structs from the C++ code (we don't want to import everything)
int inline sizeof_mlx5_qp();
int inline sizeof_mlx5_wq();
int inline sizeof_mlx5_bf();
int inline mlx5_bf_reg_size(struct rdma_shim_data * data);
void consume_send_cq(struct rdma_shim_data * data);
int sizeof_wrid(struct mlx5_qp * qp);

void * rdma_shim_malloc(int size);
void * rdma_shim_calloc(size_t n, size_t size);
void setup_custom_allocs(void * buffer, size_t size);



#endif

