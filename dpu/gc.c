#include <mutex.h>
#include "gc.h"
#include "interface.h"
#include "global.h"

#define ENABLE_GC DPU_ENABLE_GC_SYMBOL

/* Host input */
__host uint64_t ENABLE_GC;

/* Global variables */
bool gc_enabled;
static gc_lsn_t gc_lsn;
MUTEX_INIT(gc_lsn_mutex);

/* Functions */
void gc_init_global() {
  gc_enabled = (ENABLE_GC != 0);
  gc_lsn = 0;
}

void gc_update_lsn(gc_lsn_t new_gc_lsn) {
  assert_print(gc_enabled);
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
