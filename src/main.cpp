#include <iostream>
#include <cstdlib>
#include <random>
#include <thread>
#include <coroutine>
#include <barrier>
#include <unistd.h>
#include <chrono>
#include <numa.h>
#include <set>
#include <functional>
#include "engine.hpp"
#include "interface.h"

thread_local std::random_device _random_device;
thread_local std::mt19937 _rand_gen(_random_device());

struct alignas(64) counter {
  uint64_t commits = 0;
  uint64_t fails = 0;
  uint64_t conflicts = 0;
  void operator+=(const counter& other) {
    commits += other.commits;
    fails += other.fails;
    conflicts += other.conflicts;
  }
  counter operator-(const counter& other) {
    return counter{
      commits - other.commits,
      fails - other.fails,
      conflicts - other.conflicts
    };
  }
  void increment(int status) {
    if (status == STATUS_SUCCESS) ++commits;
    else if (status == STATUS_FAILED) ++fails;
    else ++conflicts;
  }
  uint64_t total() {
    return commits + fails + conflicts;
  }
};
struct coro_task {
  struct promise_type {
    coro_task get_return_object() {return {this};}
    std::suspend_always initial_suspend() noexcept {return {};}
    std::suspend_always final_suspend() noexcept {return {};}
    void unhandled_exception() {}
  };
  promise_type *_p;
};

template <typename F>
struct coro_scheduler {
public:
  using handle = std::coroutine_handle<coro_task::promise_type>;
  coro_scheduler(int num_tasks, F func, int tid_offset, int core_id, bool repeat)
    : _num_tasks(num_tasks), _next_task(0), _tid_offset(tid_offset),
      _core_id(core_id), _repeat(repeat), _func(func),
      _handles(num_tasks)
   {
    for (int i = 0; i < num_tasks; ++i) {
      _handles[i] = handle::from_promise(*(_func(tid_offset + i, core_id))._p);
    }
  }
  ~coro_scheduler() {
    for (auto &h: _handles) {
      if (h) h.destroy();
    }
  }
  bool step() {
    int initial_task = _next_task;
    while (true) {
      auto &h = _handles[_next_task];
      if (h) {
        if (!h.done()) {
          h.resume();
          break;
        }
        else {
          h.destroy();
          if (_repeat) {
            h = handle::from_promise(*(_func(_tid_offset + _next_task, _core_id))._p);
            break;
          }
          else {
            h = handle::from_address(nullptr); // mark destroyed
          }
        }
      }
      _next_task = (_next_task + 1) % _num_tasks;
      if (_next_task == initial_task) return false; // all done
    }
    _next_task = (_next_task + 1) % _num_tasks;
    return true;
  }
private:
  int _num_tasks, _next_task, _tid_offset, _core_id;
  bool _repeat;
  F _func;
  std::vector<handle> _handles;
};

thread_local std::mt19937_64 rg(7777);

int main(int argc, char *argv[]) {
  // Parse arguments
  int num_ranks_per_numa_node = 1;
  int num_threads_per_numa_node = 1;
  int batch_size = 1;
  int table_size = 1000000;
  int seconds = 10;

  int c;
  while ((c = getopt(argc, argv, "hr:t:b:s:n:")) != -1) {
    switch (c) {
    case 'r':
      num_ranks_per_numa_node = atoi(optarg);
      break;
    case 't':
      num_threads_per_numa_node = atoi(optarg);
      break;
    case 'n':
      seconds = atoi(optarg);
      break;
    case 'b':
      batch_size = atoi(optarg);
      break;
    case 's':
      table_size = atoi(optarg);
      break;
    case 'h':
    default:
      printf("%s [-r num_ranks_per_numa_node] [-t num_threads_per_numa_node] [-b batch_size] [-s table_size] [-n seconds]\n", argv[0]);
      exit(0);
    }
  }
  printf("Test run: %d ranks/numa, %d threads/numa %d batch size, %d seconds\n",
    num_ranks_per_numa_node, num_threads_per_numa_node, batch_size, seconds);

  // Initialize engine
  oltpim::engine::config engine_config;
  engine_config.num_ranks_per_numa_node = num_ranks_per_numa_node;
  engine_config.num_indexes = 1;
  engine_config.index_infos[0].primary = true;
  oltpim::engine engine;
  engine.init(engine_config);

  // Required to pin host threads to cores
  int num_cores = std::thread::hardware_concurrency();
  int num_numa_nodes = numa_max_node() + 1;
  std::vector<std::vector<int>> core_list_per_numa(num_numa_nodes);
  for (int core_id = 0; core_id < num_cores; ++core_id) {
    int numa_id = numa_node_of_cpu(core_id);
    core_list_per_numa[numa_id].push_back(core_id);
  }
  for (int numa_id = 0; numa_id < num_numa_nodes; ++numa_id) {
    if (core_list_per_numa[numa_id].size() < (size_t)num_threads_per_numa_node) {
      fprintf(stderr, "Numa node %d has less cores %lu < %d\n",
        numa_id, core_list_per_numa[numa_id].size(), num_threads_per_numa_node);
      exit(1);
    }
  }
  std::vector<int> core_list;
  for (int numa_id = 0; numa_id < num_numa_nodes; ++numa_id) {
    for (int each_core = 0; each_core < num_threads_per_numa_node; ++each_core) {
      core_list.push_back(core_list_per_numa[numa_id][each_core]);
    }
  }

  // Etc variables
  std::vector<counter> counters = std::vector<counter>(num_cores);
  volatile bool done = false;
  const uint64_t num_pims = engine.num_pims();
  auto key_to_pim = [&](uint64_t key) {return key % num_pims;};
  std::atomic<uint64_t> g_xid(1), g_csn(1);
  auto get_new_xid = [&]() {return g_xid.fetch_add(1);};
  auto get_begin_csn = [&]() {return g_csn.load();};
  auto get_end_csn = [&]() {return g_csn.fetch_add(1);};

  // init txn
  const uint64_t keys_per_txn = 10;
  const int init_batch_size = batch_size;
  auto init_txn = [&](int global_batch_id, int sys_core_id) -> coro_task {
    const int num_total_batches = init_batch_size * num_threads_per_numa_node * num_numa_nodes;
    const int table_size_per_batch = (table_size + num_total_batches) / num_total_batches;
    const uint64_t start_key = 1 + global_batch_id * table_size_per_batch;
    const uint64_t end_key = 1 + std::min((global_batch_id + 1) * table_size_per_batch, table_size);
    oltpim::request reqs[keys_per_txn];
    args_any_t args[keys_per_txn];
    rets_any_t rets[keys_per_txn];

    for (uint64_t key_base = start_key; key_base < end_key; key_base += keys_per_txn) {
      uint64_t num_keys = std::min(keys_per_txn, end_key - key_base);
      // begin
      uint64_t xid = get_new_xid();
      uint64_t begin_csn = get_begin_csn();
      std::set<int> touched_pims;

      // batch insert
      for (uint64_t k = 0; k < num_keys; ++k) {
        uint64_t key = key_base + k;
        auto &arg = args[k].insert;
        arg.key = key;
        arg.value = key + 7;
        arg.xid = xid;
        arg.csn = begin_csn;
        arg.index_id = 0;
        reqs[k] = oltpim::request(
          request_type_insert, &args[k], &rets[k], 
          sizeof(args_insert_t), sizeof(rets_insert_t)
        );
        int pim_id = key_to_pim(key);
        touched_pims.insert(pim_id);
        engine.push(pim_id, &reqs[k], sys_core_id);
      }
      for (uint64_t k = 0; k < num_keys; ++k) {
        while (!engine.is_done(&reqs[k], sys_core_id)) {
          co_await std::suspend_always{};
        }
      }

      for (uint64_t k = 0; k < num_keys; ++k) {
        auto &ret = rets[k].insert;
        assert(ret.status == STATUS_SUCCESS);
        (void)ret;
      }

      // commit
      uint64_t end_csn = get_end_csn();
      int cnt = 0;
      for (int pim_id: touched_pims) {
        auto &arg = args[cnt].commit;
        arg.xid = xid;
        arg.csn = end_csn;
        reqs[cnt] = oltpim::request(
          request_type_commit, &args[cnt], &rets[cnt],
          sizeof(args_commit_t), sizeof(rets_commit_t)
        );
        engine.push(pim_id, &reqs[cnt], sys_core_id);
        ++cnt;
      }
      for (int i = 0; i < cnt; ++i) {
        while (!engine.is_done(&reqs[i], sys_core_id)) {
          co_await std::suspend_always{};
        }
      }

      ++counters[sys_core_id].commits;
    }
  };

  // test txn
  const int tests_per_txn = 10;
  std::uniform_int_distribution<uint64_t> rand_key_distr(1, table_size + 1);
  std::discrete_distribution<int> txn_type_distr({9, 1}); // {read%, update%}
  auto test_txn = [&](int batch_id, int sys_core_id) -> coro_task {
    oltpim::request reqs[tests_per_txn];
    args_any_t args[tests_per_txn];
    rets_any_t rets[tests_per_txn];

    // begin
    uint64_t xid = get_new_xid();
    uint64_t begin_csn = get_begin_csn();
    std::set<int> touched_pims;

    static constexpr int txn_type_read = 0;
    //static constexpr int txn_type_update = 1;
    int txn_type = txn_type_distr(rg);

    for (int i = 0; i < tests_per_txn; ++i) {
      uint64_t key = rand_key_distr(rg);
      if (txn_type == txn_type_read) {
        auto &arg = args[i].get;
        arg.key = key;
        arg.xid = xid;
        arg.csn = begin_csn;
        arg.index_id = 0;
        reqs[i] = oltpim::request(
          request_type_get, &args[i], &rets[i],
          sizeof(args_get_t), sizeof(rets_get_t)
        );
      }
      else {
        auto &arg = args[i].update;
        arg.key = key;
        arg.xid = xid;
        arg.csn = begin_csn;
        arg.new_value = key + 77;
        arg.index_id = 0;
        reqs[i] = oltpim::request(
          request_type_update, &args[i], &rets[i],
          sizeof(args_update_t), sizeof(rets_update_t)
        );
      }
      int pim_id = key_to_pim(key);
      touched_pims.insert(pim_id);
      engine.push(pim_id, &reqs[i], sys_core_id);
    }
    for (int i = 0; i < tests_per_txn; ++i) {
      while (!engine.is_done(&reqs[i], sys_core_id)) {
        co_await std::suspend_always{};
      }
    }
    int status = STATUS_SUCCESS;
    for (int i = 0; i < tests_per_txn; ++i) {
      if (txn_type == txn_type_read) {
        uint64_t key = args[i].get.key;
        auto &ret = rets[i].get;

        // check
        if (key <= (uint64_t)table_size) {
          assert(ret.status != STATUS_FAILED);
          if (ret.status == STATUS_SUCCESS) {
            assert(ret.value == key + 7 || ret.value == key + 77);
          }
        }
        else {
          assert(ret.status != STATUS_SUCCESS);
        }

        // status
        if (ret.status == STATUS_FAILED) {
          status = STATUS_FAILED;
        }
        else if (ret.status == STATUS_CONFLICT && status == STATUS_SUCCESS) {
          status = STATUS_CONFLICT;
        }
      }
      else {
        uint64_t key = args[i].update.key;
        auto &ret = rets[i].update;

        // check
        if (key <= (uint64_t)table_size) {
          assert(ret.status != STATUS_FAILED);
          if (ret.status == STATUS_SUCCESS) {
            assert(ret.old_value == key + 7 || ret.old_value == key + 77);
          }
        }
        else {
          assert(ret.status != STATUS_SUCCESS);
        }

        // status
        if (ret.status == STATUS_FAILED) {
          status = STATUS_FAILED;
        }
        else if (ret.status == STATUS_CONFLICT && status == STATUS_SUCCESS) {
          status = STATUS_CONFLICT;
        }
      }
    }

    // commit/abort
    int cnt = 0;
    for (int pim_id: touched_pims) {
      if (status == STATUS_SUCCESS) {
        uint64_t end_csn = get_end_csn();
        auto &arg = args[cnt].commit;
        arg.xid = xid;
        arg.csn = end_csn;
        reqs[cnt] = oltpim::request(
          request_type_commit, &args[cnt], &rets[cnt],
          sizeof(args_commit_t), sizeof(rets_commit_t)
        );
      }
      else {
        auto &arg = args[cnt].abort;
        arg.xid = xid;
        reqs[cnt] = oltpim::request(
          request_type_abort, &args[cnt], &rets[cnt],
          sizeof(args_abort_t), sizeof(rets_abort_t)
        );
      }
      engine.push(pim_id, &reqs[cnt], sys_core_id);
      ++cnt;
    }
    for (int i = 0; i < cnt; ++i) {
      while (!engine.is_done(&reqs[i], sys_core_id)) {
        co_await std::suspend_always{};
      }
    }

    counters[sys_core_id].increment(status);
  };

  // barrier
  auto init_barrier_completion = []() noexcept {
    printf("Initialized!\n");
  };
  std::barrier init_barrier(
    num_numa_nodes * num_threads_per_numa_node, init_barrier_completion);
  
  // worker function
  auto worker_fn = [&](int sys_core_id, int tid) {
    {
      // Pin thread to core
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      CPU_SET(sys_core_id, &cpuset);
      pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
    }

    {
      // initialize
      coro_scheduler init_coros(init_batch_size, init_txn, tid * init_batch_size, sys_core_id, false);
      while (true) {
        if (!init_coros.step()) break;
        if (done) break;
      }
      engine.drain_all(sys_core_id);
    }

    // barrier
    init_barrier.arrive_and_wait();

    {
      // test
      coro_scheduler test_coros(batch_size, test_txn, 0, sys_core_id, true);
      while (true) {
        test_coros.step();
        if (done) break;
      }
      engine.drain_all(sys_core_id);
    }
  };

  // lap function
  auto lap_fn = [&]() {
    auto start_time = std::chrono::system_clock::now();
    auto last_time = start_time;
    counter last_counter;
    printf("sec:commits,fails,conflicts,CommitTPS,TotalTPS\n");
    for (int i = 0; i < seconds; ++i) {
      sleep(1);
      auto curr_time = std::chrono::system_clock::now();
      double curr_sec = (curr_time - start_time).count() / 1000000000.0;
      double real_sec = (curr_time - last_time).count() / 1000000000.0;
      last_time = curr_time;

      // get counter
      counter curr_counter;
      for (int core_id: core_list) {
        curr_counter += counters[core_id];
      }

      // diff
      counter sec_counter = curr_counter - last_counter;
      last_counter = curr_counter;
      double commit_tps = (double)sec_counter.commits / real_sec;
      double total_tps = (double)sec_counter.total() / real_sec;

      printf("%lf:%lu,%lu,%lu,%lf,%lf\n", 
        curr_sec, sec_counter.commits, sec_counter.fails, sec_counter.conflicts, 
        commit_tps, total_tps);
    }
    done = true;
  };

  // Launch threads
  std::vector<std::thread> threads;
  
  for (size_t tid = 0; tid < core_list.size(); ++tid) {
    int sys_core_id = core_list[tid];
    threads.emplace_back(worker_fn, sys_core_id, (int)tid);
  }
  lap_fn();
  for (auto &thread: threads) {
    thread.join();
  }
}
