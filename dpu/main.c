#include <stdio.h>
#include <defs.h>
#include <stdint.h>
#include <mram.h>
#include <seqread.h>
#include <assert.h>
#include <mutex.h>
#include <barrier.h>
#include <stdbool.h>
#include "common.h"
#include "global.h"

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
  typedef uint32_t arg_t;
  typedef uint64_t ret_t;
  static_assert(sizeof(ret_t) % 8 == 0, "");
  arg_t arg;
  __dma_aligned ret_t ret;
  while (true) {
    __mram_ptr uint8_t *retp;
    // Copy arg into tasklet-local buffer
    mutex_lock(args_mutex);
    if (args_offset >= args_total_offset) {
      mutex_unlock(args_mutex);
      break;
    }
    arg = *(arg_t*)args_ptr;
    args_offset += sizeof(arg_t);
    args_ptr = seqread_get(args_ptr, sizeof(arg_t), &args_reader);
    retp = rets_ptr;
    rets_ptr += sizeof(ret_t);
    mutex_unlock(args_mutex);

    // Compute
    ret = arg + 7;
    //printf("[%d] %u -> %lu\n", me(), arg, ret);

    // Write ret to mram buffer, assume 8B-aligned
    mram_write(&ret, retp, sizeof(ret_t));
  }

  return 0;
}
