#include <stdio.h>
#include <defs.h>
#include <stdint.h>
#include <mram.h>
#include <seqread.h>
#include <assert.h>
#include <mutex.h>
#include <barrier.h>
#include <stdbool.h>
#include <string.h>
#include "common.h"
#include "global.h"
#include "requests.h"

#define ARGS_BUF DPU_ARGS_SYMBOL
#define RETS_BUF DPU_RETS_SYMBOL
__mram_noinit uint8_t ARGS_BUF[DPU_BUFFER_SIZE];
__mram_noinit uint8_t RETS_BUF[DPU_BUFFER_SIZE];

bool initialized = false;
BARRIER_INIT(main_barrier, NR_TASKLETS);

// Args reader
MUTEX_INIT(args_mutex);
seqreader_buffer_t args_reader_cache;
seqreader_t args_reader;
uint8_t *args_ptr;
uint32_t args_offset, args_total_offset;
__mram_ptr uint8_t *rets_ptr;

int main() {
  if (me() == 0) {
    // Initialize
    if (!initialized) {
      args_reader_cache = seqread_alloc();
      initialized = true;
    }

    // Setup args reader
    args_ptr = seqread_init(args_reader_cache, ARGS_BUF, &args_reader);
    args_offset = 0;
    args_total_offset = *(uint32_t*)args_ptr;
    args_ptr = seqread_get(args_ptr, sizeof(uint32_t), &args_reader);
    rets_ptr = (__mram_ptr uint8_t*)RETS_BUF;
  }
  barrier_wait(&main_barrier);

  // Process requests
  args_any_t arg;
  __dma_aligned rets_any_t ret;
  request_type_t request_type;
  uint32_t request_arg_size = 0;
  uint32_t request_ret_size = 0;
  while (true) {
    __mram_ptr uint8_t *retp;
    // Copy arg into tasklet-local buffer
    mutex_lock(args_mutex);
    if (args_offset >= args_total_offset) {
      mutex_unlock(args_mutex);
      break;
    }
    request_type = (request_type_t)(*(uint8_t*)args_ptr);
    args_ptr = seqread_get(args_ptr, sizeof(uint8_t), &args_reader);

    #define case_get_arg_and_size(name) \
    memcpy(&arg.name, args_ptr, sizeof(args_##name##_t)); \
    request_arg_size = sizeof(args_##name##_t); \
    request_ret_size = sizeof(rets_##name##_t);

    REQUEST_SWITCH_CASE(request_type, case_get_arg_and_size)

    #undef case_get_arg_and_size

    args_offset += (sizeof(uint8_t) + request_arg_size);
    args_ptr = seqread_get(args_ptr, request_arg_size, &args_reader);

    retp = rets_ptr;
    rets_ptr += request_ret_size;
    mutex_unlock(args_mutex);

    // Process request
    process_request(request_type, &arg, &ret);

    // Write ret to mram buffer, assume 8B-aligned
    mram_write(&ret, retp, request_ret_size);
  }

  return 0;
}
