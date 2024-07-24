#include "rdma_shim.h"

#include <providers/mlx5/mlx5dv.h>
#include <providers/mlx5/mlx5.h>
#include <providers/mlx5/wqe.h>
#include <stdatomic.h>
#include <infiniband/verbs.h>

typedef struct mlx5_wqe_ctrl_seg __attribute__((__aligned__(8))) gic_ctrl_seg_t;

struct mlx5_qp *mqp_from_ibv_qp(struct ibv_context *ctx, struct ibv_qp *qp) {
    struct mlx5_qp * mdv_qp = mlx5qp_from_ibvqp(qp);
    return mdv_qp;
}

struct mlx5_bf *  mbf_from_mlx5_qp(struct mlx5_qp * qp)
{
    return qp->bf;
}
struct mlx5_context * mctx_from_mlx5_qp(struct mlx5_qp * qp)
{
    return to_mctx(qp->ibv_qp->context);
}

void advance_doorbell_mqp(struct mlx5_qp * mqp, int size, void * ctrl)
{
    advance_doorbell(mqp->ibv_qp, size, ctrl);
}
void advance_doorbell(struct ibv_qp * qp, int size, void * ctrl)
{
    struct mlx5_qp * mqp = mlx5qp_from_ibvqp(qp);
    struct mlx5_bf * bf = mqp->bf;
    struct mlx5_context * ctx =to_mctx( mqp->ibv_qp->context);
    bool is_simple = mlx5_is_simple_doorbell(qp);
    int nreq =1;
    if (is_simple)
    {
        mqp->sq.head += nreq;
        mqp->db[MLX5_SND_DBR] = htobe32(mqp->sq.cur_post & 0xffff);
        mmio_wc_start();
        mlx5_bf_copy2(mqp, size, ctrl);
        mmio_flush_writes();
        bf->offset ^= bf->buf_size;
    }
    else
    {
        printf("I don't know how to advance the doorbell!\n");
    }
    // Original check in the post_send_db
    // if (!ctx->shut_up_bf && nreq == 1 && bf->uuarn &&
    //         (inl || ctx->prefer_bf) && size > 1 &&
    //         size <= bf->buf_size / 16){
    // }
}

void * sq_buf_buf_from_mlx5_qp(struct mlx5_qp * qp)
{
    return qp->sq_buf.buf;
}

void * sq_start_from_mlx5_qp(struct mlx5_qp * qp)
{
    return qp->sq_start;
}
int sq_idx_from_qp(struct mlx5_qp * qp)
{
    return qp->sq.cur_post & (qp->sq.wqe_cnt - 1); 
}
void set_wr_stuff(struct mlx5_qp * mqp, int idx, int id, int size, int nreq)
{
    mqp->sq.wrid[idx] = id;
    mqp->sq.wqe_head[idx] = mqp->sq.head + nreq;
    mqp->sq.cur_post += DIV_ROUND_UP(size, MLX5_SEND_WQE_BB);

}

void mlx5_lock_qp(struct mlx5_qp * mqp)
{
    // mlx5_qp_lock(&mqp->sq.lock);
}
void mlx5_unlock_qp(struct mlx5_qp * mqp)
{
    // mlx5_qp_unlock(&mqp->sq.lock);
}

uint64_t prepare_rdma(
        struct ibv_qp * qp,
        struct rdma_shim_data* data,
        void ** lowest_p, void ** highest_p)
{
    data->qp = qp;
    data->mqp = mlx5qp_from_ibvqp(qp);
    data->mctx =to_mctx( data->mqp->ibv_qp->context);
    data->mbf = data->mqp->bf;
    data->sq_start = &(data->mqp->sq_start);
    data->wq_end = &(data->mqp->sq.qend);
    data->send_cq = to_mcq(qp->send_cq);

    data->cur_post = &(data->mqp->sq.cur_post);
    data->wqe_cnt = &(data->mqp->sq.wqe_cnt);
    data->wrid = &(data->mqp->sq.wrid);
    data->wqe_head = &(data->mqp->sq.wqe_head);
    data->sq_head = &(data->mqp->sq.head);
    data->sq_tail = &(data->mqp->sq.tail);
    data->db = &(data->mqp->db);
    data->send_dbrec = &(data->send_cq->dbrec);
    data->mbf_reg = &(data->mbf->reg);
    data->mbf_buf_size = &(data->mbf->buf_size);
    data->mbf_offset = &(data->mbf->offset);
    data->cons_index = &(data->send_cq->cons_index);
    data->active_buf = &(data->send_cq->active_buf);
    data->send_cqe= &(data->send_cq->verbs_cq.cq.cqe);

    printf("data->qp: %p\n", data->qp);
    printf("data->mqp: %p\n", data->mqp);
    printf("data->sq_start: %p -> %p\n", data->sq_start, *data->sq_start);
    printf("data->mctx: %p\n", data->mctx);
    printf("data->mbf: %p\n", data->mbf);
    printf("data->cur_post: %p\n", data->cur_post);
    printf("data->wqe_cnt: %p\n", data->wqe_cnt);
    printf("data->wrid: %p  -> %p\n", data->wrid, *data->wrid);
    printf("data->wqe_head: %p -> %p\n", data->wqe_head, *(data->wqe_head));
    printf("data->sq_head: %p\n", data->sq_head);
    printf("data->db: %p -> %p\n", data->db, *(data->db));
    printf("data->mbf_reg: %p -> %p\n", data->mbf_reg, *(data->mbf_reg));
    printf("data->mbf_buf_size: %p\n", data->mbf_buf_size);
    printf("data->mbf_offset: %p\n", data->mbf_offset);

    uint64_t lowest = 0xffffffffffff;
    uint64_t highest= 0x0;

    // Use a trick to find out how the pointers are sorted
    int size = sizeof(struct rdma_shim_data) / sizeof (uint64_t);
    uint64_t * data_ptrs = (uint64_t*)data;
    for (int i=0; i< size; i++)
    {
        //printf("%i: %p\n", i, data_ptrs[i]);
        lowest = data_ptrs[i] < lowest? data_ptrs[i] : lowest;
        highest = data_ptrs[i] > highest? data_ptrs[i] : highest;
    }
    if (lowest_p)
        *lowest_p = (void*) lowest;

    if (highest_p)
        *highest_p = (void*)highest;


    printf("Lowest address is %p, highest %p, total %liB\n",
            (void*)lowest, (void*)highest, highest-lowest);
    if(!mlx5_is_simple_doorbell(data->qp))
    {
        printf("Cannot work with not-so-simple doorbells!\n");
        exit(1);
    }
    return highest - lowest;

}



void rdma_write(
        struct rdma_shim_data * data,
        uint64_t laddr, uint64_t raddr, uint32_t lkey, uint32_t rkey, uint32_t bytes){

    mlx5_lock_qp(data->mqp); // This is unlikely to work on GPU!

    int nreq=1;
	struct mlx5_wqe_ctrl_seg ctrl_seg;
	struct mlx5_wqe_ctrl_seg *  ctrl_seg_ptr;
	struct mlx5_wqe_raddr_seg raddr_seg;
	struct mlx5_wqe_raddr_seg *  raddr_seg_ptr;
	struct mlx5_wqe_data_seg data_seg;
	struct mlx5_wqe_data_seg * data_seg_ptr;
	memset(&ctrl_seg, 0, sizeof(struct mlx5_wqe_ctrl_seg));
	memset(&data_seg, 0, sizeof(struct mlx5_wqe_data_seg));
	memset(&raddr_seg, 0, sizeof(struct mlx5_wqe_raddr_seg));

    int idx = sq_idx_from_qp(data->mqp);
    int size = sizeof(struct mlx5_wqe_ctrl_seg);
	ctrl_seg_ptr = (struct mlx5_wqe_ctrl_seg *)((uintptr_t)data->sq_start+ (idx << MLX5_SEND_WQE_SHIFT));

	ctrl_seg.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE | MLX5_WQE_CTRL_SOLICITED;
	ctrl_seg.opmod_idx_opcode = htobe32((idx << 8) | MLX5_OPCODE_RDMA_WRITE);

	raddr_seg_ptr = (struct mlx5_wqe_raddr_seg*) ((uintptr_t)ctrl_seg_ptr + size);
	raddr_seg.raddr = htobe64(raddr);
	raddr_seg.rkey = htobe32(rkey);
	raddr_seg.reserved = htobe32(0);
    size += sizeof(struct mlx5_wqe_raddr_seg);

	data_seg_ptr = (struct mlx5_wqe_data_seg*) ((uintptr_t)ctrl_seg_ptr + size);
    data_seg.addr = htobe64((uint64_t)laddr);
    data_seg.byte_count = htobe32(bytes);
    data_seg.lkey = htobe32(lkey);
    size += sizeof(struct mlx5_wqe_data_seg);

    ctrl_seg.qpn_ds = htobe32((data->qp->qp_num << 8) | size/16); 
    set_wr_stuff(data->mqp, idx, 0x1234, size,1 );

	memcpy(ctrl_seg_ptr, &ctrl_seg, sizeof(struct mlx5_wqe_ctrl_seg));
	memcpy(raddr_seg_ptr, &raddr_seg, sizeof(struct mlx5_wqe_raddr_seg));
	memcpy(data_seg_ptr, &data_seg, sizeof(struct mlx5_wqe_data_seg));

    data->mqp->sq.head += nreq;
    data->mqp->db[MLX5_SND_DBR] = htobe32(data->mqp->sq.cur_post & 0xffff);
    mmio_wc_start();
    mlx5_bf_copy2(data->mqp, size, ctrl_seg_ptr);
    mmio_flush_writes();
    data->mbf->offset ^= data->mbf->buf_size;

    mlx5_unlock_qp(data->mqp); // Not on CUDA (?)
} 

struct mlx5_wqe_ctrl_seg * get_ctrl_seg_ptr(struct rdma_shim_data * data, int idx)
{
	return (struct mlx5_wqe_ctrl_seg *)((uintptr_t)data->sq_start + (idx << MLX5_SEND_WQE_SHIFT));

}
// THis is a write with immediate, but doesnt actually write anything
void rdma_write_with_imm(
        struct rdma_shim_data * data, int imm){

    mlx5_lock_qp(data->mqp); // This is unlikely to work on GPU!

    int nreq=1;
	struct mlx5_wqe_ctrl_seg ctrl_seg;
	struct mlx5_wqe_ctrl_seg *  ctrl_seg_ptr;
	struct mlx5_wqe_raddr_seg raddr_seg;
	struct mlx5_wqe_raddr_seg *  raddr_seg_ptr;
	struct mlx5_wqe_data_seg data_seg;
	struct mlx5_wqe_data_seg * data_seg_ptr;
	memset(&ctrl_seg, 0, sizeof(struct mlx5_wqe_ctrl_seg));
	memset(&data_seg, 0, sizeof(struct mlx5_wqe_data_seg));
	memset(&raddr_seg, 0, sizeof(struct mlx5_wqe_raddr_seg));

    int idx = sq_idx_from_qp(data->mqp);
    int size = sizeof(struct mlx5_wqe_ctrl_seg);
	ctrl_seg_ptr = get_ctrl_seg_ptr(data, idx);

	ctrl_seg.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE | MLX5_WQE_CTRL_SOLICITED;
	ctrl_seg.opmod_idx_opcode = htobe32((idx << 8) | MLX5_OPCODE_RDMA_WRITE_IMM);
    ctrl_seg.imm  = imm;

	raddr_seg_ptr = (struct mlx5_wqe_raddr_seg*) ((uintptr_t)ctrl_seg_ptr + size);
	raddr_seg.raddr = htobe64(0);
	raddr_seg.rkey = htobe32(0);
	raddr_seg.reserved = htobe32(0);
    size += sizeof(struct mlx5_wqe_raddr_seg);

	// data_seg_ptr = (struct mlx5_wqe_data_seg*) ((uintptr_t)ctrl_seg_ptr + size);
    // data_seg.addr = htobe64((uint64_t)laddr);
    // data_seg.byte_count = htobe32(bytes);
    // data_seg.lkey = htobe32(lkey);
    // size += sizeof(struct mlx5_wqe_data_seg);

    ctrl_seg.qpn_ds = htobe32((data->qp->qp_num << 8) | size/16); 
    set_wr_stuff(data->mqp, idx, 0x1234, size, 1);

	memcpy(ctrl_seg_ptr, &ctrl_seg, sizeof(struct mlx5_wqe_ctrl_seg));
	memcpy(raddr_seg_ptr, &raddr_seg, sizeof(struct mlx5_wqe_raddr_seg));
	// memcpy(data_seg_ptr, &data_seg, sizeof(struct mlx5_wqe_data_seg));

    data->mqp->sq.head += nreq;
    data->mqp->db[MLX5_SND_DBR] = htobe32(data->mqp->sq.cur_post & 0xffff);
    // mlx5_bf_copy2(data->mqp, size/16, ctrl_seg_ptr);
    // mlx5_bf_copy2(data, size, ctrl_seg_ptr);
    mlx5_bf_copy3(data, size, ctrl_seg_ptr);
    data->mbf->offset ^= data->mbf->buf_size;

    mlx5_unlock_qp(data->mqp); // Not on CUDA (?)
} 

// Rexport it for cuda (there should be a better way...)
// SIZE IS IN BYTE!
void mlx5_bf_copy3(struct rdma_shim_data * data, int bytes, void * ctrl){
        mmio_wc_start();
        mlx5_bf_copy2(data->mqp, bytes / 16, ctrl);
        mmio_flush_writes();
}
// Use these to access the sizes of the structs from the C++ code (we don't want to import everything)
int sizeof_mlx5_qp(){return sizeof(struct mlx5_qp);}
int sizeof_mlx5_wq(){return sizeof(struct mlx5_wq);}
int sizeof_mlx5_bf(){return sizeof(struct mlx5_bf);}
int mlx5_bf_reg_size(struct rdma_shim_data * data) {return data->mctx->bf_reg_size;}
void consume_send_cq(struct rdma_shim_data * data){
    mlx5_consume_send_cq(data->mqp);
}
int sizeof_wrid(struct mlx5_qp * qp)
{
    return qp->sq.wqe_cnt * sizeof(*qp->sq.wrid);
}

void * custom_malloc_buffer;
size_t custom_malloc_buffer_size;
size_t allocated_malloc_buffer_size;
void * rdma_shim_malloc(int size)
{
    if (size < (custom_malloc_buffer_size - allocated_malloc_buffer_size))
    {
        void * ret = (void*)((uint64_t)custom_malloc_buffer + allocated_malloc_buffer_size);
        allocated_malloc_buffer_size+=size;
        printf("custom_malloc for %i B at %p\n", size, ret);
        return ret;
    }
    return NULL;

}
void * rdma_shim_calloc(size_t num, size_t size)
{
    void * ret = rdma_shim_malloc(num*size);
    if (ret)
        memset(ret, 0, num*size);
        return ret;
}

void setup_custom_allocs(void * buffer, size_t size){
    printf("BUFFER IS AT %p\n", buffer);
    custom_malloc_buffer_size = size;
    custom_malloc_buffer = buffer;
    allocated_malloc_buffer_size = 0;
    mlx5_change_malloc(&rdma_shim_malloc);
    mlx5_change_calloc(&rdma_shim_calloc);
}


