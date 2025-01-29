#include <mutex.h>
#include <defs.h>
#include "gc.h"
#include "interface.h"
#include "global.h"

#define GC_PROB DPU_GC_PROB_SYMBOL
#define GC_PROB_BASE_MASK ((DPU_GC_PROB_BASE) - 1)

/* Host input */
__host uint64_t GC_PROB;

/* Global variables */
bool gc_enabled;
static uint8_t gc_prob;
static gc_lsn_t gc_lsn;
MUTEX_INIT(gc_lsn_mutex);

static uint32_t gc_rng[NR_TASKLETS];

/* Functions */
void gc_init_global() {
  gc_enabled = (GC_PROB > 0);
  gc_prob = (GC_PROB < DPU_GC_PROB_BASE) ? GC_PROB : DPU_GC_PROB_BASE;
  gc_lsn = 0;
  for (uint32_t i = 0; i < NR_TASKLETS; ++i) {
    gc_rng[i] = 1 + i * i;
  }
}

bool gc_coin_toss() {
  if (!gc_enabled) return false;
  // xorshift-32
  uint32_t *rp = &gc_rng[me()];
  uint32_t r = *rp;
  r ^= r << 13;
  r ^= r >> 17;
  r ^= r << 5;
  *rp = r;
  // modulo base
  return (r & GC_PROB_BASE_MASK) < gc_prob;
}

void gc_update_lsn(gc_lsn_t new_gc_lsn) {
  assert(gc_enabled);
  mutex_lock(gc_lsn_mutex);
  if (gc_lsn < new_gc_lsn) {
    gc_lsn = new_gc_lsn;
  }
  mutex_unlock(gc_lsn_mutex);
}

gc_lsn_t gc_get_lsn() {
  mutex_lock(gc_lsn_mutex);
  const gc_lsn_t ret = gc_lsn;
  mutex_unlock(gc_lsn_mutex);
  return ret;
}
