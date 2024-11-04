#pragma once
#include <vector>
#include <atomic>
#include <thread>
#include "global.hpp"
#include "config.hpp"
#include "upmem_rank.hpp"
#include "interface.h"

namespace oltpim {

struct request {
  request *next;      // Used internally
  void *args, *rets;  // User input
  uint32_t dpu_id;    // Filled internally
  uint8_t alen, rlen; // User input
  uint8_t req_type;   // User input
  volatile bool done; // Used internally
  request() {}
  request(uint8_t req_type_, void *args_, void *rets_, uint8_t alen_, uint8_t rlen_)
    : args(args_), rets(rets_), alen(alen_), rlen(rlen_), req_type(req_type_) {}
};

struct rank_buffer {
  rank_buffer() {}
  void alloc(int num_dpus);
  ~rank_buffer();

  inline void reset_offsets();

  // Constructing the args buffer
  inline void push_args(request *req);
  inline void push_priority_separator();
  inline uint32_t finalize_args();

  // Distributing the rets buffer
  inline void pop_rets(request *req);

  int _num_dpus;
  uint8_t **bufs = nullptr;
  uint32_t *offsets = nullptr;
};

class alignas(CACHE_LINE) request_list {
 public:
  request_list();
  inline void push(request *req);
  inline request *move();

 private:
  std::atomic<request*> _head;
};

class rank_engine {
 public:
  struct config { // user-specified config
    void *dpu_rank;
  };
  struct information { // info passed from parent engine
    int rank_id;
    int num_numa_nodes;
    const char *dpu_binary;
    const char *dpu_args_symbol, *dpu_rets_symbol;
  };
  rank_engine() {}
  int init(config conf, information info);

  // Push request, called from the client side
  inline void push(request *req, int core_id);

  // Process requests; if conflict, do nothing
  bool process();

  inline upmem::rank &get_rank() {return _rank;}

 private:
  static constexpr uint32_t dpu_args_symbol_id = 0;
  static constexpr uint32_t dpu_rets_symbol_id = 1;

  int _rank_id;
  int _num_dpus;
  int _num_numa_nodes;

  // Rank controller
  upmem::rank _rank;

  // Request list
  static constexpr int num_priorities = NUM_PRIORITIES;
  array<request_list> _request_lists_per_numa_node;

  // PIM buffer
  rank_buffer _buffer;
  std::vector<uint32_t> _rets_offset_counter;
  uint32_t _max_rlength;
  std::vector<request*> _reqlists;

  // Process locks
  std::atomic<bool> _process_lock;
  int _process_phase;
};

class engine {
 public:
  struct config {
    int num_ranks_per_numa_node;
  };

  engine() {}
  void init(config conf);

  // push() and is_done() internally processes caller's numa node's
  // pending requests.
  void push(int pim_id, request *req, int sys_core_id);
  bool is_done(request *req, int sys_core_id);
  void drain_all(int sys_core_id);

  inline upmem::rank &get_rank(int rank_id) {return _rank_engines[rank_id].get_rank();}
  inline int num_pims() {return _num_dpus;}

 private:
  friend class rank_engine;
  int _num_ranks_per_numa_node;
  int _num_numa_nodes;
  int _num_ranks;
  std::vector<rank_engine> _rank_engines;

  // Buffers
  std::vector<rank_buffer> _rank_buffers;

  // Structures for pim_id -> (rank_id, dpu_id)
  int _num_dpus;
  std::vector<int> _num_dpus_per_rank;
  inline void pim_id_to_rank_dpu_id(int pim_id, int &rank_id, int &dpu_id);

  // Structures for numa_node_id -> [rank_id]s
  std::vector<std::vector<int>> _numa_id_to_rank_ids;
  bool process_local_numa_rank(int sys_core_id);

  // Structrues for core_id -> numa_node_id
  static std::vector<int> numa_node_of_core_id;

};

}
