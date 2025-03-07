#pragma once
#include <cstdio>
#include <mutex>
#include <vector>

struct dpu_transfer_matrix;
struct dpu_rank_t;
struct dpu_program_t;

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

  uint32_t register_dpu_transfer(const char *symbol, void **buffers,
    bool broadcast = false);
  void copy(uint32_t transfer_id, uint32_t length,
    bool direction_to_dpu, uint32_t symbol_offset = 0);

  void log_read(FILE *stream, bool fault_only = false, int dpu_id = -1);

  inline int num_dpus() {return _num_dpus;}
  inline int numa_node() {return _numa_node;}

  static int numa_node_of(void *dpu_rank);
  static void static_free(void *dpu_rank);

  void handle_fault();
  void core_dump(int dpu_id, const char *dump_file);

 private:
  enum memory_type {
    MRAM, WRAM, IRAM
  };

  inline memory_type fill_transfer_matrix(dpu_transfer_matrix *matrix,
    uint32_t symbol_offset, uint32_t length);
  inline void *get_copy_fn(memory_type mem_type, bool direction_to_dpu);

  struct dpu_rank_t *_rank;
  int _num_dpus;
  int _numa_node;
  uint32_t _mram_size;
  struct dpu_program_t *_program;
  const char *_binary_path;

  std::vector<dpu_transfer_matrix*> _registered_transfers;
  static std::mutex _alloc_mutex;
};

}
