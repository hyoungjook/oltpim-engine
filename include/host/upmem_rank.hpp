#pragma once
#include <cstdio>
#include <mutex>
#include <vector>

extern "C" {
  #include <dpu.h>
  #include <dpu_config.h>
  #include <dpu_debug.h>
  #include <dpu_elf.h>
  #include <dpu_loader.h>
  #include <dpu_log.h>
  #include <dpu_management.h>
  #include <dpu_memory.h>
  #include <dpu_program.h>
  #include <dpu_runner.h>
}

namespace upmem {

class rank {
 public:
  rank(): _rank(nullptr), _program(nullptr) {}
  void alloc(const char *profile = nullptr);
  static void *static_alloc(const char *profile = nullptr);
  void init(void *dpu_rank);
  ~rank();

  void load(const char *binary_path);
  bool launch(bool async = false);
  bool is_done(bool *fault = nullptr);

  uint32_t register_dpu_symbol(const char *symbol);
  void copy(uint32_t symbol_id, void **buffers, uint32_t length,
    bool direction_to_dpu, uint32_t symbol_offset = 0);
  void broadcast(uint32_t symbol_id, void *buffer, uint32_t length,
    uint32_t symbol_offset = 0);

  void log_read(FILE *stream);

  inline int num_dpus() {return _num_dpus;}
  inline int numa_node() {return _numa_node;}

  static int numa_node_of(void *dpu_rank);
  static void static_free(void *dpu_rank);

 private:
  enum memory_type {
    MRAM, WRAM, IRAM
  };

  inline memory_type fill_transfer_matrix(dpu_transfer_matrix *matrix,
    uint32_t symbol_id, uint32_t symbol_offset, uint32_t length);
  inline void *get_copy_fn(memory_type mem_type, bool direction_to_dpu);

  struct dpu_rank_t *_rank;
  int _num_dpus;
  int _numa_node;
  mram_size_t _mram_size;
  struct dpu_program_t *_program;

  std::vector<dpu_symbol_t> _registered_symbols;
  static std::mutex _alloc_mutex;
};

}
