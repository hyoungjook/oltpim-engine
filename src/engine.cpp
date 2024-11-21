#include <filesystem>
#include <algorithm>
#include <numa.h>
#include <assert.h>
#include <unistd.h>
#include "engine.hpp"

// These should be defined at compile time.
#ifndef DPU_BINARY
#define DPU_BINARY "oltpim_dpu"
#endif

namespace oltpim {

// physical core id
static thread_local int sys_core_id = -1;

// Priorities of request type.
static const int request_type_priority[] = {
#define REQUEST_TYPE_PRIORITY(_1, _2, priority, ...) priority,
REQUEST_TYPES_LIST(REQUEST_TYPE_PRIORITY)
#undef REQUEST_TYPE_PRIORITY
};

void rank_buffer::alloc(int num_dpus) {
  _num_dpus = num_dpus;
  bufs = (uint8_t**)malloc(sizeof(uint8_t*) * num_dpus);
  for (int each_dpu = 0; each_dpu < num_dpus; ++each_dpu) {
    bufs[each_dpu] = (uint8_t*)aligned_alloc(
      CACHE_LINE, DPU_BUFFER_SIZE
    );
  }
  offsets = (uint32_t*)malloc(sizeof(uint32_t) * num_dpus);
  reset_offsets();
}

rank_buffer::~rank_buffer() {
  if (bufs) {
    for (int each_dpu = 0; each_dpu < _num_dpus; ++each_dpu) {
      free(bufs[each_dpu]);
    }
    free(bufs);
  }
  if (offsets) {
    free(offsets);
  }
}

void rank_buffer::reset_offsets() {
  memset(offsets, 0, sizeof(uint32_t) * _num_dpus);
}

void rank_buffer::push_args(request *req) {
  uint32_t dpu_id = req->dpu_id;
  uint8_t alen = req->alen;
  uint8_t req_type = req->req_type;
  // First sizeof(uint32_t) bytes stores the offset 
  uint8_t *buf = &bufs[dpu_id][sizeof(uint32_t) + offsets[dpu_id]];
  buf[0] = req_type;
  memcpy(buf + sizeof(uint8_t), req->args, alen);
  offsets[dpu_id] += (sizeof(uint8_t) + alen);
}

void rank_buffer::push_priority_separator() {
  for (int each_dpu = 0; each_dpu < _num_dpus; ++each_dpu) {
    // Separator is request_type 0xFF
    bufs[each_dpu][sizeof(uint32_t) + offsets[each_dpu]] = (uint8_t)request_type_priority_separator;
    ++offsets[each_dpu];
  }
}

uint32_t rank_buffer::finalize_args() {
  uint32_t max_value = 0;
  for (int each_dpu = 0; each_dpu < _num_dpus; ++each_dpu) {
    uint32_t offset = offsets[each_dpu];
    OLTPIM_ASSERT(offset < DPU_BUFFER_SIZE);
    // Store offset to the beginning of the buffer
    *(uint32_t*)bufs[each_dpu] = offset;
    // Compute max offset
    max_value = std::max<uint32_t>(max_value, offset);
  }
  return sizeof(uint32_t) + max_value;
}

void rank_buffer::pop_rets(request *req) {
  uint32_t dpu_id = req->dpu_id;
  uint8_t rlen = req->rlen;
  memcpy(req->rets, &bufs[dpu_id][offsets[dpu_id]], rlen);
  offsets[dpu_id] += rlen;
  __atomic_store_1(&req->done, true, __ATOMIC_RELEASE);
}

request_list::request_list() {
  _head.store(nullptr);
}

void request_list::push(request *req) {
  // assume req is pre-allocated and not released until we set *done
  request *curr_head = _head.load(std::memory_order_seq_cst);
  do {
    req->next = curr_head;
  } while (
    !_head.compare_exchange_strong(curr_head, req, std::memory_order_seq_cst)
  );
}

request *request_list::move() {
  // other is not concurrently accessed
  return _head.exchange(nullptr, std::memory_order_seq_cst);
}

int rank_engine::init(config conf, information info) {
  // Initialize rank controller
  _rank_id = info.rank_id;
  _rank.init(conf.dpu_rank);
  _rank.load(info.dpu_binary);
  {
    uint32_t args_symbol_id = _rank.register_dpu_symbol(info.dpu_args_symbol);
    uint32_t rets_symbol_id = _rank.register_dpu_symbol(info.dpu_rets_symbol);
    OLTPIM_ASSERT(args_symbol_id == dpu_args_symbol_id);
    OLTPIM_ASSERT(rets_symbol_id == dpu_rets_symbol_id);
  }
  _num_dpus = _rank.num_dpus();

  // Initialize index info
  {
    uint32_t num_indexes_symbol_id = _rank.register_dpu_symbol(info.dpu_num_indexes_symbol);
    uint32_t index_infos_symbol_id = _rank.register_dpu_symbol(info.dpu_index_infos_symbol);
    uint64_t num_indexes_buf = conf.num_indexes;
    _rank.broadcast(num_indexes_symbol_id, &num_indexes_buf, sizeof(uint64_t));
    _rank.broadcast(index_infos_symbol_id, &conf.index_infos, sizeof(index_info) * DPU_MAX_NUM_INDEXES);
  }

  // Request list
  _num_numa_nodes = info.num_numa_nodes;
  _request_lists_per_numa_node.alloc(_num_numa_nodes * num_priorities);

  // Buffers
  _buffer.alloc(_num_dpus);
  _rets_offset_counter = std::vector<uint32_t>(_num_dpus, 0);
  _reqlists = std::vector<request*>(_num_numa_nodes * num_priorities);

  // Lock
  _process_lock.store(false);
  _process_phase = 0;

  // Return number of dpus
  return _num_dpus;
}

void rank_engine::push(request *req) {
  int numa_node = engine::numa_node_of_core_id[sys_core_id];
  int priority = request_type_priority[req->req_type];
  _request_lists_per_numa_node[numa_node * num_priorities + priority].push(req);
}

bool rank_engine::process() {
  bool lock_acquired = false;
  bool something_exists = false;
  // Try acquire spinlock for this rank
  if (_process_lock.compare_exchange_strong(lock_acquired, true)) {
    if (_process_phase == 0) {
      // Phase 0: Push args and launch PIM program asynchronously
      // Then move to phase 1

      // Gather requests to this rank
      for (int priority = 0; priority < num_priorities; ++priority) {
        for (int each_node = 0; each_node < _num_numa_nodes; ++each_node) {
          request *const reqlist = _request_lists_per_numa_node[each_node * num_priorities + priority].move();
          _reqlists[priority * _num_numa_nodes + each_node] = reqlist;
          something_exists = something_exists || (reqlist != nullptr);
        }
      }

      if (something_exists) {
        #ifndef NSTATS
        const auto __t0 = oltpim::now_us();
        uint64_t __nr = 0;
        #endif
        // Construct buffer
        _buffer.reset_offsets();
        memset(&_rets_offset_counter[0], 0, sizeof(uint32_t) * _num_dpus);
        for (int priority = 0; priority < num_priorities; ++priority) {
          for (int each_node = 0; each_node < _num_numa_nodes; ++each_node) {
            request **req_ptr = &_reqlists[priority * _num_numa_nodes + each_node];
            request *req;
            while ((req = *req_ptr)) {
              something_exists = true;
              _buffer.push_args(req);
              if (req->rlen > 0) {
                // increment rets offset counter
                _rets_offset_counter[req->dpu_id] += req->rlen;
                req_ptr = &(req->next);
              }
              else {
                // if req has no return value, remove it from the list
                *req_ptr = req->next;
                __atomic_store_1(&req->done, true, __ATOMIC_RELEASE);
              }
              #ifndef NSTATS
              ++__nr;
              #endif
            }
          }
          // Insert separator
          if (priority < num_priorities - 1) {
            _buffer.push_priority_separator();
          }
        }

        // Compute max lengths
        uint32_t max_alength = _buffer.finalize_args();
        max_alength = ALIGN8(max_alength);
        uint32_t max_rlength = 0;
        for (uint32_t &roffset: _rets_offset_counter) {
          max_rlength = std::max<uint32_t>(max_rlength, roffset);
        }
        _max_rlength = ALIGN8(max_rlength);

        // Copy args to rank
        #ifndef NSTATS
        stat[stats::CNTR::NUM_REQUESTS] += __nr;
        const auto __t1 = oltpim::now_us();
        stat[stats::CNTR::PREP1_US] += (__t1 - __t0);
        #endif
        _rank.copy(dpu_args_symbol_id, (void**)_buffer.bufs, max_alength, true);

        // Launch
        #ifndef NSTATS
        __launch_start_us = oltpim::now_us();
        stat[stats::CNTR::COPY1_US] += (__launch_start_us - __t1);
        #endif
        _rank.launch(true);

        // Move to phase 1
        _process_phase = 1;
      }
    }
    else if (_process_phase == 1) {
      // Phase 1: Check if PIM program is done
      // If it's done, distribute return values and move to phase 0

      // Check done
      bool pim_fault = false;
      bool pim_done = _rank.is_done(&pim_fault);
      if (pim_fault) {
        while (!_rank.is_done());
        fprintf(stderr, "Rank[%d] in fault\n", _rank_id);
        _rank.log_read(stderr, true);
        exit(1);
      }
      if (pim_done) {
        #ifndef NSTATS
        const auto __t2 = oltpim::now_us();
        stat[stats::CNTR::LAUNCH_US] += (__t2 - __launch_start_us);
        #endif
        //_rank.log_read(stdout); // debug
        // Copy rets from rank
        _rank.copy(dpu_rets_symbol_id, (void**)_buffer.bufs, _max_rlength, false);
        #ifndef NSTATS
        const auto __t3 = oltpim::now_us();
        stat[stats::CNTR::COPY2_US] += (__t3 - __t2);
        #endif

        // Distribute results: the traversal order should be the same as construction
        _buffer.reset_offsets();
        for (int priority = 0; priority < num_priorities; ++priority) {
          for (int each_node = 0; each_node < _num_numa_nodes; ++each_node) {
            request *req = _reqlists[priority * _num_numa_nodes + each_node];
            while (req) {
              _buffer.pop_rets(req);
              req = req->next;
            }
          }
        }

        // Move to phase 0
        _process_phase = 0;
        something_exists = true;
        #ifndef NSTATS
        const auto __t4 = oltpim::now_us();
        stat[stats::CNTR::PREP2_US] += (__t4 - __t3);
        ++stat[stats::CNTR::NUM_ROUNDS];
        #endif
      }
    }
    // Release spinlock
    _process_lock.store(false);
  }
  return something_exists;
}

engine engine::g_engine;
std::vector<int> engine::numa_node_of_core_id;

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
  _numa_ignore = false;

  // Information
  rank_engine::information rank_info;
  rank_info.dpu_binary = DPU_BINARY;
  rank_info.dpu_args_symbol = TOSTRING(DPU_ARGS_SYMBOL);
  rank_info.dpu_rets_symbol = TOSTRING(DPU_RETS_SYMBOL);
  rank_info.dpu_num_indexes_symbol = TOSTRING(DPU_NUM_INDEXES_SYMBOL);
  rank_info.dpu_index_infos_symbol = TOSTRING(DPU_INDEX_INFOS_SYMBOL);

  OLTPIM_ASSERT(numa_available() >= 0);
  _num_numa_nodes = numa_max_node() + 1;
  _num_ranks_per_numa_node = conf.num_ranks_per_numa_node;
  _num_ranks = _num_ranks_per_numa_node * _num_numa_nodes;
  rank_info.num_numa_nodes = _num_numa_nodes;

  // Core -> Numa shortcut
  if (numa_node_of_core_id.empty()) {
    int num_cores = std::thread::hardware_concurrency();
    numa_node_of_core_id = std::vector<int>(num_cores);
    for (int core_id = 0; core_id < num_cores; ++core_id) {
      numa_node_of_core_id[core_id] = numa_node_of_cpu(core_id);
    }
  }

  // Allocate all physical ranks
  std::vector<void*> dpu_ranks;
  while (true) {
    void *dpu_rank = upmem::rank::static_alloc();
    if (!dpu_rank) break;
    dpu_ranks.push_back(dpu_rank);
  }

  // Filter out ranks per numa node
  int rank_count_per_numa_node[_num_numa_nodes];
  memset(&rank_count_per_numa_node, 0, sizeof(int) * _num_numa_nodes);
  for (void* &dpu_rank: dpu_ranks) {
    int numa_id = upmem::rank::numa_node_of(dpu_rank);
    if (rank_count_per_numa_node[numa_id] >= _num_ranks_per_numa_node) {
      upmem::rank::static_free(dpu_rank);
      dpu_rank = nullptr;
    }
    else {
      ++rank_count_per_numa_node[numa_id];
    }
  }
  // Check if enough ranks available per numa node
  for (int each_node = 0; each_node < _num_numa_nodes; ++each_node) {
    if (rank_count_per_numa_node[each_node] != _num_ranks_per_numa_node) {
      fprintf(stderr, "Numa node %d doesn't have enough PIM ranks: %d < %d\n",
        each_node, rank_count_per_numa_node[each_node], _num_ranks_per_numa_node);
      for (void *dpu_rank: dpu_ranks) {
        if (dpu_rank) upmem::rank::static_free(dpu_rank);
      }
      exit(1);
    }
  }
  /*printf("Allocated PIM ranks: ");
  for (void *&dpu_rank: dpu_ranks) {
    printf("%d", dpu_rank != nullptr ? 1 : 0);
  }
  printf("\n");*/
  // Erase nullptrs
  dpu_ranks.erase(std::remove_if(dpu_ranks.begin(), dpu_ranks.end(),
    [](const void *dpu_rank){return dpu_rank == nullptr;}), dpu_ranks.end());
  OLTPIM_ASSERT(dpu_ranks.size() == (size_t)_num_ranks);

  // Allocate rank engines
  _rank_engines = std::vector<rank_engine>(_num_ranks);
  _rank_buffers = std::vector<rank_buffer>(_num_ranks);
  _num_dpus = 0;
  _num_dpus_per_rank = std::vector<int>(_num_ranks);
  _numa_id_to_rank_ids = std::vector<std::vector<int>>(_num_numa_nodes);
  for (int each_rank = 0; each_rank < _num_ranks; ++each_rank) {
    rank_engine::config rank_config;
    rank_config.dpu_rank = dpu_ranks[each_rank];
    rank_config.num_indexes = (uint32_t)_index_infos.size();
    memcpy(&rank_config.index_infos, &_index_infos[0], sizeof(index_info) * _index_infos.size());
    auto &re = _rank_engines[each_rank];
    rank_info.rank_id = each_rank;
    int num_dpus_this_rank = re.init(
      rank_config, rank_info
    );
    _rank_buffers[each_rank].alloc(num_dpus_this_rank);
    _num_dpus += num_dpus_this_rank;
    _num_dpus_per_rank[each_rank] = num_dpus_this_rank;
    _numa_id_to_rank_ids[re.get_rank().numa_node()].push_back(each_rank);
  }

  printf("Engine initialized for %d NUMA node(s) x %d PIM rank(s) (total %d DPUs)\n",
    _num_numa_nodes, _num_ranks_per_numa_node, _num_dpus);
}

void engine::register_worker_thread(int sys_core_id_) {
  sys_core_id = sys_core_id_;
}

int engine::get_worker_thread_core_id() {
  return sys_core_id;
}

void engine::push(int pim_id, request *req) {
  assert(_initialized && sys_core_id >= 0);
  // Push to the rank engine
  int rank_id = 0, dpu_id = 0;
  pim_id_to_rank_dpu_id(pim_id, rank_id, dpu_id);
  req->dpu_id = dpu_id;
  req->done = false;
  _rank_engines[rank_id].push(req);
}

bool engine::is_done(request *req) {
  assert(_initialized);
  if (req->done) return true;
  // Process this numa node's rank
  process_local_numa_rank();
  if (_numa_ignore) {
    process_all_ranks();
  }
  return req->done;
}

void engine::drain_all() {
  assert(_initialized);
  while (true) {
    bool something_exists = process_local_numa_rank();
    if (!something_exists) {
      sleep(1);
      something_exists = process_local_numa_rank();
      if (!something_exists) break;
    }
  }
}

void engine::pim_id_to_rank_dpu_id(int pim_id, int &rank_id, int &dpu_id) {
  assert(0 <= pim_id && pim_id < _num_dpus);
  int dpu_cnt = 0;
  for (rank_id = 0; rank_id < _num_ranks; ++rank_id) {
    int num_dpus_this_rank = _num_dpus_per_rank[rank_id];
    if (pim_id < dpu_cnt + num_dpus_this_rank) {
      dpu_id = pim_id - dpu_cnt;
      return;
    }
    dpu_cnt += num_dpus_this_rank;
  }
  assert(false);
}

bool engine::process_local_numa_rank() {
  assert(sys_core_id >= 0);
  bool something_exists = false;
  int numa_node_id = numa_node_of_core_id[sys_core_id];
  // Get ranks of this numa node
  auto &rank_ids = _numa_id_to_rank_ids[numa_node_id];
  for (int rank_id: rank_ids) {
    bool exists = _rank_engines[rank_id].process();
    something_exists = something_exists || exists;
  }
  return something_exists;
}

void engine::process_all_ranks() {
  for (auto &re: _rank_engines) {
    re.process();
  }
}

rank_engine::stats engine::get_stats() {
  rank_engine::stats s;
  for (auto &re: _rank_engines) {
    s += re.stat;
  }
  s /= _num_ranks;
  return s;
}

}
