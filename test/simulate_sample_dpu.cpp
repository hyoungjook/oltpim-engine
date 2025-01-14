#include <iostream>
#include <assert.h>
#include <unistd.h>

extern "C" {
#include <dpu.h>
#include <dpu_management.h>
#include <dpu_memory.h>
}

#define POSIX_ASSERT(expr, description) \
if (!(expr)) {perror(description); exit(1);}

#define ASSERT(expr, description) \
if (!(expr)) {fprintf(stderr, description "\n"); exit(1);}

void load_core_dump(const char *dump, dpu_set_t dpu_set) {
  uint32_t wram_size = 0, mram_size = 0;
  FILE *f = fopen(dump, "rb");
  POSIX_ASSERT(f, "fopen");
  POSIX_ASSERT(fread(&wram_size, sizeof(wram_size), 1, f) == 1, "fread");
  uint8_t *wram = new uint8_t[wram_size];
  POSIX_ASSERT(fread(wram, sizeof(uint8_t), wram_size, f) == wram_size, "fread");
  POSIX_ASSERT(fread(&mram_size, sizeof(mram_size), 1, f) == 1, "fread");
  uint8_t *mram = new uint8_t[mram_size];
  POSIX_ASSERT(fread(mram, sizeof(uint8_t), mram_size, f) == mram_size, "fread");
  fclose(f);
  ASSERT(wram_size > 0 && mram_size > 0, "size is zero");

  dpu_set_t dpu_s;
  dpu_t *dpu;
  DPU_FOREACH(dpu_set, dpu_s) {
    dpu = dpu_from_set(dpu_s);
    ASSERT(dpu, "dpu set NULL");
    break;
  }
  uint32_t nb_words_in_wram = wram_size / sizeof(dpuword_t);
  DPU_ASSERT(dpu_copy_to_wram_for_dpu(dpu, 0, (dpuword_t*)wram, nb_words_in_wram));
  DPU_ASSERT(dpu_copy_to_mram(dpu, 0, mram, mram_size));
  delete [] wram;
  delete [] mram;
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("usage: %s dpu_binary core_dump trace_dir\n", argv[0]);
    exit(1);
  }
  const char *dpu_binary = argv[1];
  const char *core_dump = argv[2];
  const char *trace_dir = argv[3];

  // Enable trace
  setenv("UPMEM_TRACE_DIR", trace_dir, 1);

  // Allocate a functional simulator
  dpu_set_t dpu_set;
  DPU_ASSERT(dpu_alloc(1, "backend=simulator", &dpu_set));

  // Load program
  DPU_ASSERT(dpu_load(dpu_set, dpu_binary, NULL));

  // Load core dump
  load_core_dump(core_dump, dpu_set);

  // Launch
  DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));

  // Release
  DPU_ASSERT(dpu_free(dpu_set));

}
