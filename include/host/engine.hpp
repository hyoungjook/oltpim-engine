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
  inline void mark_done() {done.store(true, std::memory_order_release);}
  inline bool is_done() {return done.load(std::memory_order_acquire);}
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
  inline void push_gc_lsn(uint64_t gc_lsn);
  inline void finalize_args();

  // Distributing the rets buffer
  inline request_base *pop_rets(request_base *req);

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
    bool enable_gc;
    bool enable_measure_energy;
  };
  struct information { // info passed from parent engine
    int rank_id;
    int num_numa_nodes;
    const char *dpu_binary;
    const char *dpu_args_symbol, *dpu_rets_symbol;
    const char *dpu_num_indexes_symbol, *dpu_index_infos_symbol;
    const char *dpu_enable_gc_symbol;
  };
  rank_engine() {}
  int init(config conf, information info);

  // Push request, called from the client side
  inline void push(request_base *req);

  // Process requests; if conflict, do nothing
  void process();

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
  std::atomic_flag _process_lock;
  int _process_phase;
  // Ignores requests from the workers in other numa nodes.
  // enable this only if using numa_local_key option.
  bool _process_collect_only_numa_local_requests;

  // GC
  bool _enable_gc;
  uint64_t _sent_gc_lsn, _recent_gc_lsn;

  // Sample DPU execution for energy estimation
  bool _enable_measure_energy;
  bool _entered_measurement;
  bool _core_dump_sampled;
  double _avg_pim_time_us; // sum of (pim_time * rank_util)
  uint64_t _pim_time_t0;
  float _rank_util;
  inline void try_sample_dpu_profiling();
  void start_measure_pim_time();

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
    // Enable garbage collection?
    bool enable_gc;
    // Enable energy measurement?
    bool enable_measure_energy;
  };
  void init(config conf);
  void optimize_for_numa_local_key();

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

  void update_gc_lsn(uint64_t gc_lsn);
  rank_engine::stats get_stats();

  void start_measurement();
  double compute_dpu_power(double elapsed_sec);

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
  bool _enable_gc;

  // Structures for pim_id -> (rank_id, dpu_id)
  int _num_dpus, _num_dpus_per_numa_node;
  inline void pim_id_to_rank_dpu_id(int pim_id, uint16_t &rank_id, uint8_t &dpu_id);
};

}
