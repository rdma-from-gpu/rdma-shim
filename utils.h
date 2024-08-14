#ifndef RDMA_SHIM_UTILS_H
#define RDMA_SHIM_UTILS_H

#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <stdint.h>
#include <chrono>

using namespace std;

inline uint64_t now()
{
    // TODO: Use a more optimized solution
    auto time = std::chrono::high_resolution_clock::now();
    return std::chrono::time_point_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now())
      .time_since_epoch()
      .count();
}

inline void really_now(volatile uint64_t *ptr)
{
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    *ptr = ts.tv_sec*1'000'000'000 + ts.tv_nsec;
    __asm__ __volatile__("mfence");
}




inline void print_sge(ibv_sge *sge) {
    // We do these casts to avoid warnings in the print
    void *ptr = nullptr;
    void *lkey = nullptr;
    ptr = (void *)sge->addr;
    lkey = (void *)((uint64_t)(sge->lkey));

    printf("SGE: %i B at %p with lkey %p (%li)\n", sge->length, (void *)ptr,
           lkey, (uint64_t)lkey);
}

inline const char *ibv_opcode_str(ibv_wc_opcode opcode) {
    switch (opcode) {
    case IBV_WC_SEND:
        return "IBV_WC_SEND";
    case IBV_WC_RDMA_WRITE:
        return "IBV_WC_RDMA_WRITE";
    case IBV_WC_RDMA_READ:
        return "IBV_WC_RDMA_READ";
    case IBV_WC_COMP_SWAP:
        return "IBV_WC_COMP_SWAP";
    case IBV_WC_FETCH_ADD:
        return "IBV_WC_FETCH_ADD";
    case IBV_WC_BIND_MW:
        return "IBV_WC_BIND_MW";
    case IBV_WC_RECV:
        return "IBV_WC_RECV";
    case IBV_WC_RECV_RDMA_WITH_IMM:
        return "IBV_WC_RECV_RDMA_WITH_IMM";
    default:
        return "UNKNOWN";
    }
}

inline const char *ibv_opcode_str(ibv_wr_opcode opcode) {
    switch (opcode) {
    case IBV_WR_SEND:
        return "IBV_WR_SEND";
    case IBV_WR_RDMA_WRITE:
        return "IBV_WR_RDMA_WRITE";
    case IBV_WR_RDMA_READ:
        return "IBV_WR_RDMA_READ";
        // case IBV_WR_COMP_SWAP: return "IBV_WR_COMP_SWAP";
        // case IBV_WR_FETCH_ADD: return "IBV_WR_FETCH_ADD";
    case IBV_WR_BIND_MW:
        return "IBV_WR_BIND_MW";
    case IBV_WR_RDMA_WRITE_WITH_IMM:
        return "IBV_WR_WRITE_WITH_IMM";
    default:
        return "UNKNOWN";
    }
}

inline void print_rdma_wr(ibv_send_wr *wr) {
    ibv_send_wr *w = wr;
    while (w != nullptr) {
        printf("WR: %p with id %li, %s. Remote %p with rkey %p.\n", w, w->wr_id,
               ibv_opcode_str(w->opcode), (void *)w->wr.rdma.remote_addr,
               (void *)(uint64_t)w->wr.rdma.rkey);
        for (int i = 0; i < w->num_sge; i++)
            print_sge(&w->sg_list[i]);
        w = w->next;
    }
}

inline const char *qp_state_str(ibv_qp_state s) {
    switch (s) {
    case IBV_QPS_RESET:
        return "IBV_QPS_RESET";
    case IBV_QPS_SQD:
        return "IBV_QPS_SQD";
    case IBV_QPS_SQE:
        return "IBV_QPS_SQE";
    case IBV_QPS_ERR:
        return "IBV_QPS_ERR";
    case IBV_QPS_RTR:
        return "IBV_QPS_RTR";
    case IBV_QPS_RTS:
        return "IBV_QPS_RTS";
    case IBV_QPS_INIT:
        return "IBV_QPS_INIT";
    case IBV_QPS_UNKNOWN:
    default:
        return "IBV_QPS_UNKNOWN";
    }
}

inline int mtu_to_int(enum ibv_mtu mtu) {
    switch (mtu) {
    case IBV_MTU_256:
        return 256;
    case IBV_MTU_512:
        return 512;
    case IBV_MTU_1024:
        return 1024;
    case IBV_MTU_2048:
        return 2048;
    case IBV_MTU_4096:
        return 4096;
    }
    return 0;
}

inline ibv_mtu mtu_to_int(int mtu) {
    switch (mtu) {
    case 256:
        return IBV_MTU_256;
    case 512:
        return IBV_MTU_512;
    case 1024:
        return IBV_MTU_1024;
    case 2048:
        return IBV_MTU_2048;
    default:
        return IBV_MTU_4096;
    }
}

#endif
