#pragma once
#include <vector>
#include <atomic>
#include <thread>
#include "global.hpp"
#include "upmem_rank.hpp"
#include "interface.h"

namespace oltpim {

struct request_base {
public:
  request_base *next;     // Used internally
  uint16_t rank_id;       // Used internally
  uint8_t dpu_id;         // Used internally
  std::atomic<bool> done; // Used internally
  uint8_t req_type;       // USER INPUT
  uint8_t alen, rlen;     // USER INPUT
  request_base() {}
  request_base(uint8_t req_type_, uint8_t alen_, uint8_t rlen_)
    : req_type(req_type_), alen(alen_), rlen(rlen_) {}
  inline void *args() {return (void*)(this + 1);}
  inline void *rets() {return (void*)(((uint8_t*)(this + 1)) + alen);}
};

template <request_type_t TYPE, typename arg_t, typename ret_t>
struct request {
  request_base meta;
  arg_t args;
  ret_t rets;
  request(): meta(TYPE, sizeof(arg_t), sizeof(ret_t)) {}
  using _arg_t = arg_t;
};

template <request_type_t TYPE, typename arg_t>
struct request_norets {
  request_base meta;
  arg_t args;
  request_norets(): meta(TYPE, sizeof(arg_t), 0) {}
};

struct alignas(CACHE_LINE) rank_buffer {
  using buf_alloc_fn = void*(*)(size_t,int);
  static auto wrap_alloc_fn(buf_alloc_fn alloc_fn);

  rank_buffer() {}
  void alloc(int num_dpus, buf_alloc_fn alloc_fn, int numa_id);
  ~rank_buffer();

  inline void reset_offsets(bool both);

  // Constructing the args buffer
  inline void push_args(request_base *req);
  inline void push_priority_separator();
  inline void finalize_args();

  // Distributing the rets buffer
  inline void pop_rets(request_base *req);

  int _num_dpus;
  uint8_t **bufs = nullptr;
  uint32_t *offsets = nullptr, *rets_offsets = nullptr;
  uint32_t max_alength, max_rlength;
};

class alignas(CACHE_LINE) request_list {
 public:
  request_list();
  inline void push(request_base *req);
  inline request_base *move();

 private:
  std::atomic<request_base*> _head;
};

class rank_engine {
 public:
  struct config { // user-specified config
    void *dpu_rank;
    uint32_t num_indexes;
    index_info index_infos[DPU_MAX_NUM_INDEXES];
    rank_buffer::buf_alloc_fn alloc_fn;
  };
  struct information { // info passed from parent engine
    int rank_id;
    int num_numa_nodes;
    const char *dpu_binary;
    const char *dpu_args_symbol, *dpu_rets_symbol;
    const char *dpu_num_indexes_symbol, *dpu_index_infos_symbol;
  };
  rank_engine() {}
  int init(config conf, information info);

  // Push request, called from the client side
  inline void push(request_base *req);

  // Process requests; if conflict, do nothing
  bool process();

  void print_log(int dpu_id = -1);

 private:
  friend class engine;
  static constexpr uint32_t NUM_DPUS_PER_RANK = 64;
  static constexpr uint32_t dpu_args_transfer_id = 0;
  static constexpr uint32_t dpu_rets_transfer_id = 1;

  int _rank_id;
  int _num_dpus;
  int _num_numa_nodes;

  // Rank controller
  upmem::rank _rank;

  // Request list
  static constexpr int num_priorities = NUM_PRIORITIES;
  std::vector<request_list*> _request_lists_per_numa;

  // PIM buffer
  rank_buffer _buffer;
  request_base *_saved_requests;

  // Process locks
  std::atomic<bool> _process_lock;
  int _process_phase;

public:
  // Statistics
  struct stats {
    enum CNTR: int {
      NUM_ROUNDS,
      NUM_REQUESTS,
      LAUNCH_US,
      PREP1_US,
      COPY1_US,
      COPY2_US,
      PREP2_US,
      NUM_CNTRS
    };
    static constexpr int NUM_COUNTERS = CNTR::NUM_CNTRS;
    uint64_t cnt[NUM_COUNTERS] = {0,};
    uint64_t &operator[](int idx) {return cnt[idx];}
    stats &operator+=(const stats &other) {for (int i = 0; i < NUM_COUNTERS; ++i) cnt[i] += other.cnt[i]; return *this;}
    stats &operator/=(int div) {for (int i = 0; i < NUM_COUNTERS; ++i) cnt[i] /= div; return *this;}
    stats operator-(const stats &other) const {stats s; for (int i = 0; i < NUM_COUNTERS; ++i) s.cnt[i] = cnt[i] - other.cnt[i]; return s;}
    std::string to_string() {std::string s; for (int i = 0; i < NUM_COUNTERS; ++i) s += (std::to_string(cnt[i]) + ","); return s;}
  } stat;
private:
  uint64_t __launch_start_us;
};

class engine {
 public:
  static engine g_engine;

  // initialize
  int add_index(index_info info);
  struct config {
    // PIM ranks per numa node to use.
    int num_ranks_per_numa_node;
    // allocation function for PIM buffers.
    // if nullptr, use default malloc.
    rank_buffer::buf_alloc_fn alloc_fn;
  };
  void init(config conf);

  void register_worker_thread(int sys_core_id);

  // push() and is_done() internally processes caller's numa node's
  // pending requests.
  void push(int pim_id, request_base *req);
  bool is_done(request_base *req);

  void print_log(int pim_id = -1);

  template <typename req_t>
  inline void push(int pim_id, req_t *req) {push(pim_id, (request_base*)req);}
  template <typename req_t>
  inline bool is_done(req_t *req) {return is_done((request_base*)req);}

  inline upmem::rank &get_rank(int rank_id) {return _rank_engines[rank_id]->_rank;}
  inline int num_pims() {return _num_dpus;}
  inline int num_pims_per_numa_node() {return _num_dpus_per_numa_node;}

  rank_engine::stats get_stats();

 private:
  engine();
  bool _initialized;
  // used before initialization
  std::vector<index_info> _index_infos;

  // properties
  friend class rank_engine;
  static constexpr uint32_t NUM_DPUS_PER_RANK = rank_engine::NUM_DPUS_PER_RANK;
  int _num_ranks_per_numa_node;
  int _num_numa_nodes;
  int _num_ranks;
  std::vector<rank_engine*> _rank_engines;

  // Structures for pim_id -> (rank_id, dpu_id)
  int _num_dpus, _num_dpus_per_numa_node;
  inline void pim_id_to_rank_dpu_id(int pim_id, uint16_t &rank_id, uint8_t &dpu_id);
};

}
