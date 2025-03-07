#include <filesystem>
#include <algorithm>
#include <iostream>
#include <numa.h>
#include <assert.h>
#include <unistd.h>
#include <sys/wait.h>
#include "engine.hpp"

// These should be defined at compile time.
#ifndef DPU_BINARY
#define DPU_BINARY "oltpim_dpu"
#endif

#ifndef ANALYZE_SAMPLE_DPU_PATH
#define ANALYZE_SAMPLE_DPU_PATH "analyze_sample_dpu.py"
#endif

namespace oltpim {

uint64_t cur_us() {
  struct timeval tv;
  gettimeofday(&tv, 0);
  return ((uint64_t)tv.tv_sec) * 1000000 + tv.tv_usec;
}

// physical core id
static thread_local int my_numa_id = -1;

// Priorities of request type.
static const int request_type_priority[] = {
#define REQUEST_TYPE_PRIORITY(_1, _2, priority, ...) priority,
REQUEST_TYPES_LIST(REQUEST_TYPE_PRIORITY)
#undef REQUEST_TYPE_PRIORITY
};

auto rank_buffer::wrap_alloc_fn(buf_alloc_fn alloc_fn) {
  if (!alloc_fn) {
    alloc_fn = [](size_t size, int node) -> void* {
      return numa_alloc_onnode(size, node);
    };
  }
  return [alloc_fn](size_t size, int node) -> void* {
    static constexpr size_t align = CACHE_LINE;
    uintptr_t underlying = (uintptr_t)alloc_fn(size + align, node);
    underlying = (underlying + align - 1) / align * align;
    return (void*)underlying;
  };
}

void rank_buffer::alloc(int num_dpus, buf_alloc_fn alloc_fn, int numa_id) {
  _num_dpus = num_dpus;
  auto wrapped_alloc_fn = wrap_alloc_fn(alloc_fn);

  bufs = (uint8_t**)malloc(sizeof(uint8_t*) * num_dpus);
  for (int each_dpu = 0; each_dpu < num_dpus; ++each_dpu) {
    bufs[each_dpu] = (uint8_t*)wrapped_alloc_fn(DPU_BUFFER_SIZE, numa_id);
  }
  offsets = (uint32_t*)wrapped_alloc_fn(2 * sizeof(uint32_t) * num_dpus, numa_id);
  rets_offsets = &offsets[num_dpus];
  reset_offsets(true);
}

rank_buffer::~rank_buffer() {
  if (bufs) {
    for (int each_dpu = 0; each_dpu < _num_dpus; ++each_dpu) {
      // skip freeing this; might allocated with external allocator
      //free(bufs[each_dpu]);
    }
    free(bufs);
  }
  if (offsets) {
    free(offsets);
  }
}

void rank_buffer::reset_offsets(bool both) {
  memset(offsets, 0, (both ? 2 : 1) * sizeof(uint32_t) * _num_dpus);
}

void rank_buffer::push_args(request_base *req) {
  assert(!req->is_done());
  const uint32_t dpu_id = req->dpu_id;
  const uint8_t alen = req->alen;
  // First sizeof(uint32_t) bytes stores the offset 
  uint8_t *buf = &bufs[dpu_id][sizeof(uint32_t) + offsets[dpu_id]];
  buf[0] = req->req_type;
  memcpy(buf + sizeof(uint8_t), req->args(), alen);
  offsets[dpu_id] += (sizeof(uint8_t) + alen);
  rets_offsets[dpu_id] += req->rlen;
}

void rank_buffer::push_priority_separator() {
  for (int each_dpu = 0; each_dpu < _num_dpus; ++each_dpu) {
    // Separator is request_type 0xFF
    bufs[each_dpu][sizeof(uint32_t) + offsets[each_dpu]] = (uint8_t)request_type_priority_separator;
    ++offsets[each_dpu];
  }
}

void rank_buffer::push_gc_lsn(uint64_t gc_lsn) {
  // assume args_gc_t is solely gc_lsn
  for (int each_dpu = 0; each_dpu < _num_dpus; ++each_dpu) {
    uint8_t *buf = &bufs[each_dpu][sizeof(uint32_t) + offsets[each_dpu]];
    buf[0] = (uint8_t)request_type_gc;
    memcpy(&buf[1], &gc_lsn, sizeof(uint64_t));
    offsets[each_dpu] += (1 + sizeof(uint64_t));
  }
}

void rank_buffer::finalize_args() {
  max_alength = 0;
  max_rlength = 0;
  for (int each_dpu = 0; each_dpu < _num_dpus; ++each_dpu) {
    uint32_t offset = offsets[each_dpu];
    if (__builtin_expect(offset >= DPU_BUFFER_SIZE, 0)) abort();
    // Store offset to the beginning of the buffer
    *(uint32_t*)bufs[each_dpu] = offset;
    // Compute max offsets
    max_alength = std::max(max_alength, offset);
    max_rlength = std::max(max_rlength, rets_offsets[each_dpu]);
  }
  max_alength = ALIGN8(sizeof(uint32_t) + max_alength);
  max_rlength = ALIGN8(max_rlength);
}

request_base *rank_buffer::pop_rets(request_base *req) {
  const uint32_t dpu_id = req->dpu_id;
  const uint8_t rlen = req->rlen;
  // no-return requests are already popped before launch()
  assert(rlen > 0);
  memcpy(req->rets(), &bufs[dpu_id][offsets[dpu_id]], rlen);
  offsets[dpu_id] += rlen;
  return req->next;
}

request_list::request_list() {
  _head.store(nullptr);
}

void request_list::push(request_base *req) {
  // assume req is pre-allocated and not released until we set *done
  req->next = _head.load(std::memory_order_relaxed);
  while (!_head.compare_exchange_weak(
    req->next, req,
    std::memory_order_release, std::memory_order_relaxed));
}

request_base *request_list::move() {
  // other is not concurrently accessed
  return _head.exchange(nullptr, std::memory_order_seq_cst);
}

int rank_engine::init(config conf, information info) {
  // Initialize rank controller
  _rank_id = info.rank_id;
  _rank.init(conf.dpu_rank);
  _rank.load(info.dpu_binary);
  _num_dpus = _rank.num_dpus();
  if (_num_dpus != NUM_DPUS_PER_RANK) {
    std::cerr << "oltpim-engine assumes all DPUs in a rank are enabled.\n";
    std::cerr << "rank[" << _rank_id << "] has only " << _num_dpus <<
      " DPUs available.\n";
    std::abort();
  }

  // Request list
  _num_numa_nodes = info.num_numa_nodes;
  auto wrapped_alloc_fn = rank_buffer::wrap_alloc_fn(conf.alloc_fn);
  _request_lists_per_numa = std::vector<request_list*>(_num_numa_nodes);
  for (int each_node = 0; each_node < _num_numa_nodes; ++each_node) {
    auto *rl = (request_list*)wrapped_alloc_fn(sizeof(request_list) * num_priorities, each_node);
    for (int p = 0; p < num_priorities; ++p) {
      new (&rl[p]) request_list;
    }
    _request_lists_per_numa[each_node] = rl;
  }

  // Buffers
  _buffer.alloc(_num_dpus, conf.alloc_fn, _rank.numa_node());

  // Register transfers in advance
  {
    uint32_t args_transfer_id = _rank.register_dpu_transfer(
      info.dpu_args_symbol, (void**)_buffer.bufs);
    uint32_t rets_transfer_id = _rank.register_dpu_transfer(
      info.dpu_rets_symbol, (void**)_buffer.bufs);
    OLTPIM_ASSERT(args_transfer_id == dpu_args_transfer_id);
    OLTPIM_ASSERT(rets_transfer_id == dpu_rets_transfer_id);
  }

  // Initialize index info & config
  {
    uint64_t num_indexes_buf = conf.num_indexes;
    uint32_t num_indexes_transfer_id = _rank.register_dpu_transfer(
      info.dpu_num_indexes_symbol, (void**)(&num_indexes_buf), true);
    uint32_t index_infos_transfer_id = _rank.register_dpu_transfer(
      info.dpu_index_infos_symbol, (void**)(&conf.index_infos), true);
    _rank.copy(num_indexes_transfer_id, sizeof(uint64_t), true);
    _rank.copy(index_infos_transfer_id, sizeof(index_info) * DPU_MAX_NUM_INDEXES, true);
    uint64_t gc_prob_buf = (uint64_t)conf.gc_prob;
    uint32_t gc_transfer_id = _rank.register_dpu_transfer(
      info.dpu_gc_prob_symbol, (void**)(&gc_prob_buf), true);
    _rank.copy(gc_transfer_id, sizeof(uint64_t), true);
  }

  // Lock
  _process_lock.clear();
  _process_phase = 0;
  _process_collect_only_numa_local_requests = false;

  // GC
  _enable_gc = conf.gc_prob > 0;
  _sent_gc_lsn = 0;
  _recent_gc_lsn = 0;

  // Energy
  _enable_measure_energy = conf.enable_measure_energy;
  _entered_measurement = false;
  _core_dump_sampled = false;
  _avg_pim_time_us = 0;
  _pim_time_t0 = 0;
  _rank_util = 0;

  // Interleave
  _enable_interleave = conf.enable_interleave;

  // Return number of dpus
  return _num_dpus;
}

void rank_engine::push(request_base *req) {
  int priority = request_type_priority[req->req_type];
  _request_lists_per_numa[my_numa_id][priority].push(req);
}

bool rank_engine::_process_phase_0() {
  // Phase 0: Push args and launch

  // Construct buffer & saved_requests (linked list of all requests)
  _saved_requests = nullptr;
  request_base *last_req = nullptr;
  bool something_exists = false;
  assert(
    (!_process_collect_only_numa_local_requests) ||
    (my_numa_id == _rank.numa_node())
  ); // on numa_local_key option, process() should be only called from numa-local thread.
  for (int priority = 0; priority < num_priorities; ++priority) {
    for (int each_node = 0; each_node < _num_numa_nodes; ++each_node) {
      if (_process_collect_only_numa_local_requests && (each_node != my_numa_id)) {
        // This prevents the below move() function to do unnecessary
        // atomic exchange on numa-remote variable.
        continue;
      }
      request_base *req = _request_lists_per_numa[each_node][priority].move();
      if (req && !something_exists) {
        // lazy buffer initialization
        something_exists = true;
        _buffer.reset_offsets(true);
        for (int p = 0; p < priority; ++p) {
          _buffer.push_priority_separator();
        }
      }
      while (req) {
        assert(req->rank_id == (uint16_t)_rank_id);
        // target of the current iteration: *req
        // push req to buffer
        _buffer.push_args(req);
        // save next req
        request_base *req_next = req->next;
        // check there's return value
        if (req->rlen == 0) {
          // if req has no return value, don't push to the global linked list
          // and mark done now
          req->mark_done();
          if (last_req) {
            last_req->next = req_next;
          }
        }
        else {
          // connect req to the global linked list
          if (last_req) {
            last_req->next = req;
          }
          else { // the globally first req
            _saved_requests = req;
          }
          // update iterators
          last_req = req;
        }
        req = req_next;
      }
    }
    // Insert separator
    if (something_exists && (priority < num_priorities - 1)) {
      _buffer.push_priority_separator();
    }
  }
  if (something_exists) {
    // Finalize buffers
    if (_enable_gc) {
      uint64_t recent_gc_lsn = *(volatile uint64_t*)&_recent_gc_lsn;
      if (_sent_gc_lsn < recent_gc_lsn) {
        _sent_gc_lsn = recent_gc_lsn;
        _buffer.push_gc_lsn(recent_gc_lsn);
      }
    }
    _buffer.finalize_args();
    // Copy args to rank
    _rank.copy(dpu_args_transfer_id, _buffer.max_alength, true);
    try_sample_dpu_profiling();
  }
  return something_exists;
}

bool rank_engine::_process_phase_1() {
  // Phase 1: Check if PIM program is done
  // Then distribute return values and move to phase 0

  // Check done
  bool pim_fault = false;
  bool pim_done = _rank.is_done(&pim_fault);
  if (pim_fault) {
    while (!_rank.is_done());
    fprintf(stderr, "Rank[%d] in fault\n", _rank_id);
    _rank.log_read(stderr, true);
    _rank.handle_fault();
    abort();
  }
  if (pim_done) {
    //_rank.log_read(stdout); // debug
    if (_entered_measurement) {
      _avg_pim_time_us += (double)(cur_us() - _pim_time_t0) * _rank_util;
    }

    // Copy rets from rank
    _rank.copy(dpu_rets_transfer_id, _buffer.max_rlength, false);

    // Distribute results: the traversal order should be the same as construction
    _buffer.reset_offsets(false);
    request_base *req = _saved_requests;
    while (req) {
      assert(req->rank_id == (uint16_t)_rank_id);
      auto *req_next = _buffer.pop_rets(req);
      req->mark_done();
      req = req_next;
    }
  }
  return pim_done;
}

void rank_engine::process() {
  // Try acquire spinlock for this rank
  if (!_process_lock.test_and_set(std::memory_order_acquire)) {
    if (_enable_interleave) {
      switch (_process_phase) {
      case 0: {
        if (_process_phase_0()) {
          _rank.launch(true);
          _process_phase = 1;
        }
      } break;
      case 1: {
        if (_process_phase_1()) {
          _process_phase = 0;
        }
      } break;
      }
    }
    else {
      if (_process_phase_0()) {
        _rank.launch(false);
        _process_phase_1();
      }
    }
    // Release spinlock
    _process_lock.clear(std::memory_order_release);
  }
}

void rank_engine::print_log(int dpu_id) {
  _rank.log_read(stdout, false, dpu_id);
}

#define SAMPLE_DPU_CORE_DUMP_FILE "/tmp/sample_dpu_core_dump"

void rank_engine::try_sample_dpu_profiling() {
  if (!_entered_measurement) return;
  // Statistics for utilization
  uint32_t max_offset = 0, total_offset = 0;
  for (int d = 0; d < _buffer._num_dpus; ++d) {
    if (_buffer.offsets[d] >= max_offset) {
      max_offset = _buffer.offsets[d];
    }
    total_offset += _buffer.offsets[d];
  }
  _rank_util = (float)total_offset / max_offset / _buffer._num_dpus;

  // Sample only once, in random
  if (__builtin_expect(_rank_id == 0 && !_core_dump_sampled, 0)) {
    // Sample if input has sufficient length
    static constexpr uint32_t min_offset = 256;
    if (_buffer.max_alength >= sizeof(uint32_t) + min_offset) {
      // Core dump
      int sample_dpu = 0;
      for (int d = 0; d < _buffer._num_dpus; ++d) {
        if (_buffer.offsets[d] >= max_offset) sample_dpu = d;
      }
      _rank.core_dump(sample_dpu, SAMPLE_DPU_CORE_DUMP_FILE);
      _core_dump_sampled = true;
    }
  }

  // record pim time
  _pim_time_t0 = cur_us();
}

void rank_engine::start_measure_pim_time() {
  if (_enable_measure_energy) {
    _entered_measurement = true;
  }
}

engine engine::g_engine;

engine::engine(): _initialized(false) {}

int engine::add_index(index_info info) {
  assert(!_initialized);
  int index_id = (int)_index_infos.size();
  _index_infos.push_back(info);
  assert(_index_infos.size() <= DPU_MAX_NUM_INDEXES);
  return index_id;
}

void engine::init(config conf) {
  assert(!_initialized);
  _initialized = true;

  // Information
  rank_engine::information rank_info;
  rank_info.dpu_binary = DPU_BINARY;
  rank_info.dpu_args_symbol = TOSTRING(DPU_ARGS_SYMBOL);
  rank_info.dpu_rets_symbol = TOSTRING(DPU_RETS_SYMBOL);
  rank_info.dpu_num_indexes_symbol = TOSTRING(DPU_NUM_INDEXES_SYMBOL);
  rank_info.dpu_index_infos_symbol = TOSTRING(DPU_INDEX_INFOS_SYMBOL);
  rank_info.dpu_gc_prob_symbol = TOSTRING(DPU_GC_PROB_SYMBOL);

  OLTPIM_ASSERT(numa_available() >= 0);
  _num_numa_nodes = numa_max_node() + 1;
  _num_ranks_per_numa_node = conf.num_ranks_per_numa_node;
  _num_ranks = _num_ranks_per_numa_node * _num_numa_nodes;
  rank_info.num_numa_nodes = _num_numa_nodes;

  // Allocate all physical ranks, filter out ranks per numa node
  std::vector<std::vector<void*>> dpu_ranks(_num_numa_nodes);
  std::vector<void*> overfull_dpu_ranks;
  while (true) {
    void *dpu_rank = upmem::rank::static_alloc();
    if (!dpu_rank) break;
    int numa_id = upmem::rank::numa_node_of(dpu_rank);
    assert(0 <= numa_id && numa_id < _num_numa_nodes);
    if (dpu_ranks[numa_id].size() < (size_t)_num_ranks_per_numa_node) {
      dpu_ranks[numa_id].push_back(dpu_rank);
    }
    else {
      overfull_dpu_ranks.push_back(dpu_rank);
    }
  }
  for (void *overfull_rank: overfull_dpu_ranks) {
    upmem::rank::static_free(overfull_rank);
  }
  overfull_dpu_ranks.clear();
  // Check if enough ranks available per numa node
  for (int each_node = 0; each_node < _num_numa_nodes; ++each_node) {
    if (dpu_ranks[each_node].size() != (size_t)_num_ranks_per_numa_node) {
      std::cerr << "NUMA node " << each_node << "doesn't have enough PIM ranks: " <<
        dpu_ranks[each_node].size() << " < " << _num_ranks_per_numa_node << "\n";
      for (auto &ranks: dpu_ranks) {
        for (void *rank: ranks) {
          upmem::rank::static_free(rank);
        }
      }
      std::abort();
    }
  }

  // Allocate rank engines
  _rank_engines = std::vector<rank_engine*>(_num_ranks);
  _num_dpus = 0;
  int rank_id = 0;
  auto wrapped_alloc_fn = rank_buffer::wrap_alloc_fn(conf.alloc_fn);
  uint32_t converted_gc_prob = (uint32_t)(conf.gc_prob * DPU_GC_PROB_BASE + 0.5);
  converted_gc_prob = std::min<uint32_t>(converted_gc_prob, DPU_GC_PROB_BASE);
  for (int each_node = 0; each_node < _num_numa_nodes; ++each_node) {
    assert(dpu_ranks[each_node].size() == (size_t)_num_ranks_per_numa_node);
    for (int each_rank = 0; each_rank < _num_ranks_per_numa_node; ++each_rank) {
      // fill config and info
      rank_engine::config rank_config;
      rank_config.dpu_rank = dpu_ranks[each_node][each_rank];
      rank_config.num_indexes = (uint32_t)_index_infos.size();
      memcpy(&rank_config.index_infos, &_index_infos[0], sizeof(index_info) * _index_infos.size());
      rank_config.alloc_fn = conf.alloc_fn;
      rank_config.gc_prob = converted_gc_prob;
      rank_config.enable_interleave = conf.enable_interleave;
      rank_config.enable_measure_energy = conf.enable_measure_energy;
      rank_info.rank_id = rank_id;
      // allocate rank_engine in its local numa node
      auto* re = (rank_engine*)wrapped_alloc_fn(sizeof(rank_engine), each_node);
      new (re) rank_engine();
      int num_dpus_this_rank = re->init(rank_config, rank_info);
      _num_dpus += num_dpus_this_rank;
      assert(num_dpus_this_rank == NUM_DPUS_PER_RANK); // assume all DPUs are enabled
      assert(num_dpus_this_rank <= 255); // ensure dpu_id is uint8_t
      _rank_engines[rank_id] = re;
      ++rank_id;
    }
  }
  OLTPIM_ASSERT(rank_id == _num_ranks);
  OLTPIM_ASSERT(_num_dpus == _num_ranks * (int)NUM_DPUS_PER_RANK);
  _num_dpus_per_numa_node = _num_dpus / _num_numa_nodes;
  _enable_gc = converted_gc_prob > 0;

  printf("Engine initialized for %d NUMA node(s) x %d PIM rank(s) (total %d DPUs)\n",
    _num_numa_nodes, _num_ranks_per_numa_node, _num_dpus);
}

void engine::optimize_for_numa_local_key() {
  for (auto *re: _rank_engines) {
    re->_process_collect_only_numa_local_requests = true;
  }
}

void engine::register_worker_thread(int sys_core_id) {
  my_numa_id = numa_node_of_cpu(sys_core_id);
}

void engine::push(int pim_id, request_base *req) {
  assert(_initialized && my_numa_id >= 0);
  // Push to the rank engine
  pim_id_to_rank_dpu_id(pim_id, req->rank_id, req->dpu_id);
  req->done.store(false, std::memory_order_release);
  _rank_engines[req->rank_id]->push(req);
}

bool engine::is_done(request_base *req) {
  assert(_initialized);
  if (req->is_done()) return true;
  _rank_engines[req->rank_id]->process();
  return req->is_done();
}

void engine::update_gc_lsn(uint64_t gc_lsn) {
  if (!_enable_gc) return;
  for (auto &re: _rank_engines) {
    *(volatile uint64_t*)&re->_recent_gc_lsn = gc_lsn;
  }
}

void engine::print_log(int pim_id) {
  if (pim_id < 0) {
    for (auto &re: _rank_engines) {
      re->print_log();
    }
  }
  else {
    uint16_t rank_id = 0;
    uint8_t dpu_id = 0;
    pim_id_to_rank_dpu_id(pim_id, rank_id, dpu_id);
    _rank_engines[rank_id]->print_log((int)dpu_id);
  }
}

void engine::pim_id_to_rank_dpu_id(int pim_id, uint16_t &rank_id, uint8_t &dpu_id) {
  assert(0 <= pim_id && pim_id < _num_dpus);
  // Simple division, as we assume all DPUs are enabled
  rank_id = (uint16_t)((uint32_t)pim_id / NUM_DPUS_PER_RANK);
  dpu_id = (uint8_t)((uint32_t)pim_id % NUM_DPUS_PER_RANK);
}

rank_engine::stats engine::get_stats() {
  rank_engine::stats s;
  for (auto &re: _rank_engines) {
    s += re->stat;
  }
  s /= _num_ranks;
  return s;
}

void engine::start_measurement() {
  // Used to estimate pim energy
  for (auto &re: _rank_engines) re->start_measure_pim_time();
}

void engine::compute_dpu_stats(double elapsed_sec,
    double &pim_util, double &wram_ratio, double &mram_ratio, double &mram_avg_size) {
  if (!_rank_engines[0]->_core_dump_sampled) {
    fprintf(stderr, "Core Dump is not sampled! Try reducing min_offset in oltpim::rank_engine::try_sample_dpu_profiling().\n");
    return;
  }

  // PIM utilization
  pim_util = 0;
  for (auto &re: _rank_engines) {
    double util = ((double)re->_avg_pim_time_us / 1000000.0) / elapsed_sec;
    pim_util += util;
  }
  pim_util /= _rank_engines.size();

  // Compute power factor using SAMPLE_DPU_CORE_DUMP_FILE
  pid_t compute_pid;
  int compute_fd[2];
  if (pipe(compute_fd) != 0) {perror("pipe"); return;}
  compute_pid = fork();
  if (compute_pid == 0) {
    close(compute_fd[0]);
    dup2(compute_fd[1], STDOUT_FILENO);
    exit(execl("/usr/bin/python3", "python3", ANALYZE_SAMPLE_DPU_PATH,
      "--dpu-binary", DPU_BINARY, "--dump-file", SAMPLE_DPU_CORE_DUMP_FILE,
      nullptr));
  }
  close(compute_fd[1]);
  waitpid(compute_pid, nullptr, 0);
  FILE *f = fdopen(compute_fd[0], "r");
  if (!f) {perror("fdopen"); return;}
  if (fscanf(f, "%lf", &wram_ratio) == EOF) {perror("fscanf"); return;}
  if (fscanf(f, "%lf", &mram_ratio) == EOF) {perror("fscanf"); return;}
  if (fscanf(f, "%lf", &mram_avg_size) == EOF) {perror("fscanf"); return;}
}

}
