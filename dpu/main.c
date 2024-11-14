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
#include "interface.h"
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
uint8_t priority_frontier;

int main() {
  if (me() == 0) {
    // Initialize
    if (!initialized) {
      args_reader_cache = seqread_alloc();
      process_init_global();
      initialized = true;
    }

    // Setup args reader
    args_ptr = seqread_init(args_reader_cache, ARGS_BUF, &args_reader);
    args_offset = 0;
    args_total_offset = *(uint32_t*)args_ptr;
    args_ptr = seqread_get(args_ptr, sizeof(uint32_t), &args_reader);
    rets_ptr = (__mram_ptr uint8_t*)RETS_BUF;
    priority_frontier = 0;
  }
  barrier_wait(&main_barrier);

  // Process requests
  args_any_t arg;
  __dma_aligned rets_any_t ret;
  request_type_t request_type;
  uint32_t request_arg_size = 0;
  uint32_t request_ret_size = 0, request_ret_size_including_supp = 0;
  uint8_t curr_priority = 0;
  while (true) {
    __mram_ptr uint8_t *retp;

    // Mutex block: Copy args into local arg
    mutex_lock(args_mutex);

      // Check priority barrier
    if (curr_priority < priority_frontier) {
      assert_print(curr_priority + 1 == priority_frontier);
      curr_priority = priority_frontier;
      mutex_unlock(args_mutex);
      barrier_wait(&main_barrier);
      mutex_lock(args_mutex);
    }

      // Check if all done
    if (args_offset >= args_total_offset) {
      mutex_unlock(args_mutex);
      break;
    }

      // Fetch request type
    request_type = (request_type_t)(*(uint8_t*)args_ptr);
    args_ptr = seqread_get(args_ptr, sizeof(uint8_t), &args_reader);

      // Switch for request type
      // If priority separator, increment frontier and skip processing
    #define case_get_arg_and_size(name) \
    memcpy(&arg.name, args_ptr, sizeof(args_##name##_t)); \
    request_arg_size = sizeof(args_##name##_t); \
    request_ret_size = sizeof(rets_##name##_t); \
    request_ret_size_including_supp = req_##name##_rets_size(&arg.name);
    REQUEST_SWITCH_CASE(request_type, case_get_arg_and_size,
      ++priority_frontier;
      ++args_offset;
      mutex_unlock(args_mutex);
      continue;
    )
    #undef case_get_arg_and_size

      // Update pointers
    args_offset += (sizeof(uint8_t) + request_arg_size);
    args_ptr = seqread_get(args_ptr, request_arg_size, &args_reader);
    retp = rets_ptr;
    rets_ptr += request_ret_size_including_supp;
    // Mutex block end
    mutex_unlock(args_mutex);

    // Process request
    process_request(request_type, &arg, &ret, retp);

    // Write ret to mram buffer, assume 8B-aligned
    // Here, we only copy the fixed-size part (request_ret_size).
    // If request_ret_size_including_supp > request_ret_size, the supplementary
    // part is copied within process_request() function.
    // One exception: if request_ret_size_including_supp == 0, that means
    // there's no return value and copying is not required at all.
    if (request_ret_size_including_supp > 0) {
      mram_write(&ret, retp, request_ret_size);
    }

  }

  assert_print(priority_frontier == NUM_PRIORITIES - 1);
  return 0;
}
