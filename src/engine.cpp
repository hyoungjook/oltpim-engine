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
static thread_local int my_numa_id = -1;

// Priorities of request type.
static const int request_type_priority[] = {
#define REQUEST_TYPE_PRIORITY(_1, _2, priority, ...) priority,
REQUEST_TYPES_LIST(REQUEST_TYPE_PRIORITY)
#undef REQUEST_TYPE_PRIORITY
};

void rank_buffer::alloc(int num_dpus, buf_alloc_fn alloc_fn) {
  _num_dpus = num_dpus;
  if (!alloc_fn) {
    alloc_fn = [](size_t size) -> void* {
      return (void*)aligned_alloc(CACHE_LINE, size);
    };
  }

  auto aligned_alloc_fn = [&](size_t align, size_t size) -> void* {
    uintptr_t underlying = (uintptr_t)alloc_fn(size + align);
    underlying = (underlying + align - 1) / align * align;
    return (void*)underlying;
  };

  bufs = (uint8_t**)malloc(sizeof(uint8_t*) * num_dpus);
  for (int each_dpu = 0; each_dpu < num_dpus; ++each_dpu) {
    bufs[each_dpu] = (uint8_t*)aligned_alloc_fn(CACHE_LINE, DPU_BUFFER_SIZE);
  }
  offsets = (uint32_t*)aligned_alloc(CACHE_LINE, 2 * sizeof(uint32_t) * num_dpus);
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

void rank_buffer::finalize_args() {
  max_alength = 0;
  max_rlength = 0;
  for (int each_dpu = 0; each_dpu < _num_dpus; ++each_dpu) {
    uint32_t offset = offsets[each_dpu];
    OLTPIM_ASSERT(offset < DPU_BUFFER_SIZE);
    // Store offset to the beginning of the buffer
    *(uint32_t*)bufs[each_dpu] = offset;
    // Compute max offsets
    max_alength = std::max(max_alength, offset);
    max_rlength = std::max(max_rlength, rets_offsets[each_dpu]);
  }
  max_alength = ALIGN8(sizeof(uint32_t) + max_alength);
  max_rlength = ALIGN8(max_rlength);
}

void rank_buffer::pop_rets(request_base *req) {
  const uint32_t dpu_id = req->dpu_id;
  const uint8_t rlen = req->rlen;
  memcpy(req->rets(), &bufs[dpu_id][offsets[dpu_id]], rlen);
  offsets[dpu_id] += rlen;
  req->done.store(true, std::memory_order_release);
}

request_list::request_list() {
  _head.store(nullptr);
}

void request_list::push(request_base *req) {
  // assume req is pre-allocated and not released until we set *done
  request_base *curr_head = _head.load(std::memory_order_seq_cst);
  do {
    req->next = curr_head;
  } while (
    !_head.compare_exchange_strong(curr_head, req, std::memory_order_seq_cst)
  );
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
  _buffer.alloc(_num_dpus, conf.alloc_fn);

  // Lock
  _process_lock.store(false);
  _process_phase = 0;

  // Numa launch
  _enable_numa_launch = info.enable_numa_launch;
  _numa_scheduler = (numa_run::scheduler*)info.numa_scheduler;

  // Return number of dpus
  return _num_dpus;
}

void rank_engine::push(request_base *req) {
  int priority = request_type_priority[req->req_type];
  _request_lists_per_numa_node[my_numa_id * num_priorities + priority].push(req);
}

bool rank_engine::process() {
  bool lock_acquired = false;
  bool something_exists = false;
  // Try acquire spinlock for this rank
  if (_process_lock.compare_exchange_strong(lock_acquired, true)) {
    if (_process_phase == 0) {
      // Phase 0: Push args and launch PIM program asynchronously
      // Then move to phase 1

      // Construct buffer & saved_requests (linked list of all requests)
      _saved_requests = nullptr;
      request_base *last_req = nullptr;
      bool something_exists = false;
      for (int priority = 0; priority < num_priorities; ++priority) {
        for (int each_node = 0; each_node < _num_numa_nodes; ++each_node) {
          request_base *req = _request_lists_per_numa_node[each_node * num_priorities + priority].move();
          if (req && !something_exists) {
            // lazy buffer initialization
            something_exists = true;
            _buffer.reset_offsets(true);
            for (int p = 0; p < priority; ++p) {
              _buffer.push_priority_separator();
            }
          }
          while (req) {
            // target of the current iteration: *req
            // push req to buffer
            _buffer.push_args(req);
            // save next req
            request_base *req_next = req->next;
            // check there's return value
            if (req->rlen == 0) {
              // if req has no return value, don't push to the global linked list
              // and mark done now
              req->done.store(true, std::memory_order_release);
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
        _buffer.finalize_args();
        // Copy args to rank
        _rank.copy(dpu_args_symbol_id, (void**)_buffer.bufs, _buffer.max_alength, true);
        // Launch
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
        _rank.handle_fault();
        abort();
      }
      if (pim_done) {
        //_rank.log_read(stdout); // debug
        // Copy rets from rank
        _rank.copy(dpu_rets_symbol_id, (void**)_buffer.bufs, _buffer.max_rlength, false);

        // Distribute results: the traversal order should be the same as construction
        _buffer.reset_offsets(false);
        request_base *req = _saved_requests;
        while (req) {
          _buffer.pop_rets(req);
          req = req->next;
        }

        // Move to phase 0
        _process_phase = 0;
        something_exists = true;
      }
    }
    // Release spinlock
    _process_lock.store(false);
  }
  return something_exists;
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

  OLTPIM_ASSERT(numa_available() >= 0);
  _num_numa_nodes = numa_max_node() + 1;
  _num_ranks_per_numa_node = conf.num_ranks_per_numa_node;
  _num_ranks = _num_ranks_per_numa_node * _num_numa_nodes;
  rank_info.num_numa_nodes = _num_numa_nodes;

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

  rank_info.enable_numa_launch = conf.enable_numa_launch;
  if (conf.enable_numa_launch) {
    if ((size_t)(conf.numa_launch_num_workers_per_numa_node * _num_numa_nodes) >
        (std::thread::hardware_concurrency() / 2)) {
      std::cerr << "Too many workers_per_numa_node on numa_scheduler " <<
                   "will result in cpu oversubscription.\n";
      std::abort();
    }
    // NUMA launch settings
    _numa_scheduler = std::make_unique<numa_run::scheduler>(
      conf.numa_launch_num_workers_per_numa_node,
      // avoid transaction workers on the physical cores
      std::thread::hardware_concurrency() / 2
    );
    rank_info.numa_scheduler = (void*)_numa_scheduler.get();
  }

  // Allocate rank engines
  _rank_engines = std::vector<rank_engine>(_num_ranks);
  _num_dpus = 0;
  _num_dpus_per_rank = std::vector<int>(_num_ranks);
  _numa_id_to_rank_ids = std::vector<std::vector<int>>(_num_numa_nodes);
  for (int each_rank = 0; each_rank < _num_ranks; ++each_rank) {
    rank_engine::config rank_config;
    rank_config.dpu_rank = dpu_ranks[each_rank];
    rank_config.num_indexes = (uint32_t)_index_infos.size();
    memcpy(&rank_config.index_infos, &_index_infos[0], sizeof(index_info) * _index_infos.size());
    rank_config.alloc_fn = conf.alloc_fn;
    auto &re = _rank_engines[each_rank];
    rank_info.rank_id = each_rank;
    int num_dpus_this_rank = re.init(
      rank_config, rank_info
    );
    _num_dpus += num_dpus_this_rank;
    _num_dpus_per_rank[each_rank] = num_dpus_this_rank;
    assert(num_dpus_this_rank <= 255); // ensure dpu_id is uint8_t
    _numa_id_to_rank_ids[re.get_rank().numa_node()].push_back(each_rank);
  }

  printf("Engine initialized for %d NUMA node(s) x %d PIM rank(s) (total %d DPUs)\n",
    _num_numa_nodes, _num_ranks_per_numa_node, _num_dpus);
}

void engine::register_worker_thread(int sys_core_id) {
  my_numa_id = numa_node_of_cpu(sys_core_id);
}

void engine::push(int pim_id, request_base *req) {
  assert(_initialized && my_numa_id >= 0);
  // Push to the rank engine
  pim_id_to_rank_dpu_id(pim_id, req->rank_id, req->dpu_id);
  req->done.store(false);
  _rank_engines[req->rank_id].push(req);
}

bool engine::is_done(request_base *req) {
  assert(_initialized);
  if (req->done.load(std::memory_order_acquire)) return true;
  _rank_engines[req->rank_id].process();
  return req->done.load(std::memory_order_acquire);
}

void engine::pim_id_to_rank_dpu_id(int pim_id, uint16_t &rank_id, uint8_t &dpu_id) {
  assert(0 <= pim_id && pim_id < _num_dpus);
  int dpu_cnt = 0;
  for (int r = 0; r < _num_ranks; ++r) {
    int num_dpus_this_rank = _num_dpus_per_rank[r];
    if (pim_id < dpu_cnt + num_dpus_this_rank) {
      rank_id = (uint16_t)r;
      dpu_id = (uint8_t)(pim_id - dpu_cnt);
      return;
    }
    dpu_cnt += num_dpus_this_rank;
  }
  assert(false);
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
