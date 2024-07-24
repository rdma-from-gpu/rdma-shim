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

#endif
