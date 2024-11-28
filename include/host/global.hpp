#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
namespace oltpim {
// Constants
#define CACHE_LINE 64

// Macros
#ifdef NDEBUG
#define OLTPIM_ASSERT(stmt) (void)(stmt)
#else
#define OLTPIM_ASSERT(stmt) \
do { \
  bool _status = stmt; \
  if (!_status) { \
    fprintf(stderr, "Assertion failed: %s (%s:%d)\n", \
      #stmt, __FILE__, __LINE__); \
    exit(1); \
  } \
} while (0)
#endif

#define __STRINGIFY(x) #x
#define TOSTRING(x) __STRINGIFY(x)

#define ALIGN8(x) (((x) + 7) & (~7))

// Functions
static inline uint64_t now_us() {
  struct timeval tv;
  gettimeofday(&tv, 0);
  return ((uint64_t)tv.tv_sec) * 1000000 + tv.tv_usec;
}

}
