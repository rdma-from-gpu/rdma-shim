#include "utils.h"

uint64_t now()
{
    // TODO: Use a more optimized solution
    auto time = std::chrono::high_resolution_clock::now();
    return std::chrono::time_point_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now())
      .time_since_epoch()
      .count();
}
