#include <stdio.h>
#include <defs.h>
#include <stdint.h>
#include <mram.h>
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
uint32_t args_offset, args_total_offset;
__mram_ptr uint8_t *rets_ptr;
uint8_t priority_frontier;

static uint8_t *args_read_type(
    __mram_ptr uint8_t *argp, uint8_t *cache, uint32_t modulo) {
  // Fetch 2B at argp[modulo]
  // modulo = (argp % 8)
  mram_read(&argp[-modulo], cache, (modulo == 7) ? 16 : 8);
  return &cache[modulo];
}

static void args_read_content(
    args_any_t *arg_buf, __mram_ptr uint8_t *argp, uint8_t *cache,
    uint32_t modulo, uint32_t args_size) {
  // Assume args_read_type() is already called and part of cache is filled up.
  #define ROUNDUP8(x) (((x) + 7) & (~7))
  if (modulo == 7)
    mram_read(&argp[9], &cache[16], ROUNDUP8(args_size - 8));
  else
    mram_read(&argp[8 - modulo], &cache[8], ROUNDUP8(args_size - (7 - modulo)));
  memcpy(arg_buf, &cache[1 + modulo], ROUNDUP8(args_size));
  #undef ROUNDUP8
}

extern void check_do_nothing_buf();

int main() {
  // Per-thread buffer for argument reader
  __dma_aligned uint8_t args_reader_cache[REQUEST_MAX_ARGS_SIZE + 8];

  if (me() == 0) {
    // Initialize
    if (!initialized) {
      process_init_global();
      check_do_nothing_buf();
      initialized = true;
    }

    // Setup argument reader
    args_offset = 0;
    mram_read(&ARGS_BUF, args_reader_cache, 8);
    args_total_offset = *(uint32_t*)args_reader_cache;
    rets_ptr = (__mram_ptr uint8_t*)RETS_BUF;
    priority_frontier = 0;
  }
  barrier_wait(&main_barrier);

  // Process requests
  args_any_t arg;
  request_type_t request_type;
  uint32_t request_arg_size = 0;
  uint32_t request_ret_size = 0;
  uint32_t modulo = 0;
  uint8_t curr_priority = 0;
  while (true) {
    __mram_ptr uint8_t *marg, *mret;

    // Mutex block: manage args_reader
    {
      mutex_lock(args_mutex);
      // Check priority barrier
      if (curr_priority < priority_frontier) {
        assert(curr_priority + 1 == priority_frontier);
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
      // Increment args_offset and rets_ptr ASAP and unlock the mutex ASAP
      {
        // Fetch request type
        const uint32_t real_offset = sizeof(uint32_t) + args_offset;
        marg = &ARGS_BUF[real_offset];
        mret = rets_ptr;
        modulo = (uint32_t)(real_offset & 7);
        uint8_t *const type_p = args_read_type(marg, args_reader_cache, modulo);
        request_type = (request_type_t)type_p[0];
        // Determine arg_size
        // priority_separator: increment offset & frontier, and skip the rest
        #define case_arg_ret_sizes(name) \
        request_arg_size = sizeof(args_##name##_t); \
        request_ret_size = req_##name##_rets_size((args_##name##_t*)&type_p[1]);
        REQUEST_SWITCH_CASE(request_type, case_arg_ret_sizes,
          ++priority_frontier;
          ++args_offset;
          mutex_unlock(args_mutex);
          continue;
        )
        #undef case_arg_ret_sizes
        // Increment args_offset and rets_ptr
        args_offset += (sizeof(uint8_t) + request_arg_size);
        rets_ptr += request_ret_size;
      }
      mutex_unlock(args_mutex);
    }

    // Fetch arg content
    #define case_fetch_arg(name) \
    args_read_content(&arg, marg, args_reader_cache, modulo, request_arg_size);
    REQUEST_SWITCH_CASE(request_type, case_fetch_arg,
      always_assert(false);
    )
    #undef case_fetch_arg

    // Process request
    process_request(request_type, &arg, mret);
  }

  assert(priority_frontier == NUM_PRIORITIES - 1);
  return 0;
}
