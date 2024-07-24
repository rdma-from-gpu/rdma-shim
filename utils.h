#ifndef RDMA_SHIM_UTILS_H
#define RDMA_SHIM_UTILS_H

#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>

using namespace std;


uint64_t now();

inline void really_now(volatile uint64_t *ptr)
{
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    *ptr = ts.tv_sec*1'000'000'000 + ts.tv_nsec;
    __asm__ __volatile__("mfence");
}

#endif
