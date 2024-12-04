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
#include "interface_host.hpp"

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
  coro_scheduler(int num_tasks, F func, bool repeat, void *args)
    : _num_tasks(num_tasks), _next_task(0),
      _repeat(repeat), _func(func), _args(args), _handles(num_tasks)
   {
    for (int i = 0; i < num_tasks; ++i) {
      _handles[i] = handle::from_promise(*(_func(i, _args))._p);
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
            h = handle::from_promise(*(_func(_next_task, _args))._p);
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
  int _num_tasks, _next_task;
  bool _repeat;
  F _func;
  void *_args;
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
  auto &engine = oltpim::engine::g_engine;
  engine.add_index(index_info{.primary = true});
  engine.init(oltpim::engine::config{
    .num_ranks_per_numa_node = num_ranks_per_numa_node,
    .alloc_fn = nullptr,
    .enable_gc = false
  });

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
  constexpr uint64_t scan_length = 8;
  auto key_to_pim = [&](uint64_t key) -> int {return (key / 8) % num_pims;};
  std::atomic<uint64_t> g_xid(1), g_csn(1);
  auto get_new_xid = [&]() {return g_xid.fetch_add(1);};
  auto get_begin_csn = [&]() {return g_csn.load();};
  auto get_end_csn = [&]() {return g_csn.fetch_add(1);};

  // init txn
  constexpr uint64_t keys_per_txn = 10;
  const int init_batch_size = batch_size;
  struct init_txn_args {
    int sys_core_id;
    int global_batch_offset;
  };
  auto init_txn = [&](int batch_id, void *a) -> coro_task {
    auto *args = (init_txn_args*)a;
    int global_batch_id = args->global_batch_offset + batch_id;
    const int num_total_batches = init_batch_size * num_threads_per_numa_node * num_numa_nodes;
    const int table_size_per_batch = (table_size + num_total_batches) / num_total_batches;
    const uint64_t start_key = 1 + global_batch_id * table_size_per_batch;
    const uint64_t end_key = 1 + std::min((global_batch_id + 1) * table_size_per_batch, table_size);
    const int worker_thd_core_id = args->sys_core_id;

    for (uint64_t key_base = start_key; key_base < end_key; key_base += keys_per_txn) {
      uint64_t num_keys = std::min(keys_per_txn, end_key - key_base);
      // begin
      uint64_t xid = get_new_xid();
      uint64_t begin_csn = get_begin_csn();
      std::set<int> touched_pims;

      // batch insert
      {
        oltpim::request_insert reqs[keys_per_txn];
        for (uint64_t k = 0; k < num_keys; ++k) {
          auto &arg = reqs[k].args;
          const uint64_t key = key_base + k;
          arg.key = key;
          arg.value = key + 7;
          arg.xid = xid;
          arg.csn = begin_csn;
          arg.index_id = 0;
          int pim_id = key_to_pim(key);
          touched_pims.insert(pim_id);
          engine.push(pim_id, &reqs[k]);
        }
        for (uint64_t k = 0; k < num_keys; ++k) {
          while (!engine.is_done(&reqs[k])) {
            co_await std::suspend_always{};
          }
        }

        for (uint64_t k = 0; k < num_keys; ++k) {
          assert(reqs[k].rets.status == STATUS_SUCCESS);
        }
      }

      // commit
      {
        oltpim::request_commit reqs[keys_per_txn];
        uint64_t end_csn = get_end_csn();
        int cnt = 0;
        for (int pim_id: touched_pims) {
          auto &arg = reqs[cnt].args;
          arg.xid = xid;
          arg.csn = end_csn;
          engine.push(pim_id, &reqs[cnt]);
          ++cnt;
        }
        for (int i = 0; i < cnt; ++i) {
          while (!engine.is_done(&reqs[i])) {
            co_await std::suspend_always{};
          }
        }
      }

      ++counters[worker_thd_core_id].commits;
    }

    // verify
    for (uint64_t key_base = start_key; key_base < end_key; key_base += keys_per_txn) {
      uint64_t num_keys = std::min(keys_per_txn, end_key - key_base);
      // begin
      uint64_t xid = get_new_xid();
      uint64_t begin_csn = get_begin_csn();
      std::set<int> touched_pims;

      // batch get
      {
        oltpim::request_get reqs[keys_per_txn];
        for (uint64_t k = 0; k < num_keys; ++k) {
          auto &arg = reqs[k].args;
          const uint64_t key = key_base + k;
          arg.key = key;
          arg.xid = xid;
          arg.csn = begin_csn;
          arg.index_id = 0;
          arg.oid_query = 0;
          int pim_id = key_to_pim(key);
          touched_pims.insert(pim_id);
          engine.push(pim_id, &reqs[k]);
        }
        for (uint64_t k = 0; k < num_keys; ++k) {
          while (!engine.is_done(&reqs[k])) {
            co_await std::suspend_always{};
          }
        }

        for (uint64_t k = 0; k < num_keys; ++k) {
          assert(reqs[k].rets.status == STATUS_SUCCESS);
          assert(reqs[k].rets.value == reqs[k].args.key + 7);
        }
      }

      // commit
      {
        oltpim::request_commit reqs[keys_per_txn];
        uint64_t end_csn = get_end_csn();
        int cnt = 0;
        for (int pim_id: touched_pims) {
          auto &arg = reqs[cnt].args;
          arg.xid = xid;
          arg.csn = end_csn;
          engine.push(pim_id, &reqs[cnt]);
          ++cnt;
        }
        for (int i = 0; i < cnt; ++i) {
          while (!engine.is_done(&reqs[i])) {
            co_await std::suspend_always{};
          }
        }
      }

      ++counters[worker_thd_core_id].commits;
    }
  };

  // test txn
  const int tests_per_txn = 10;
  std::uniform_int_distribution<uint64_t> rand_key_distr(1, table_size + 1);
  std::discrete_distribution<int> txn_type_distr({8, 1, 1}); // {read%, update%, scan%}
  struct test_txn_args {
    int sys_core_id;
  };
  auto test_txn = [&](int batch_id, void *a) -> coro_task {
    auto *args = (test_txn_args*)a;
    static constexpr int txn_type_read = 0;
    static constexpr int txn_type_update = 1;
    static constexpr int txn_type_scan = 2;
    int txn_type = txn_type_distr(rg);
    const int worker_thd_core_id = args->sys_core_id;

    if (txn_type == txn_type_scan) {
      uint64_t key = rand_key_distr(rg);
      uint64_t xid = get_new_xid();
      uint64_t begin_csn = get_begin_csn();

      typename oltpim::request_scan<scan_length>::t req;
      const uint64_t lkey = (key / scan_length) * scan_length;
      const uint64_t rkey = lkey + scan_length - 1;
      req.args.max_outs = scan_length;
      req.args.index_id = 0;
      req.args.keys[0] = lkey;
      req.args.keys[1] = rkey;
      req.args.xid = xid;
      req.args.csn = begin_csn;
      int pim_id = key_to_pim(lkey);
      assert(key_to_pim(rkey) == pim_id);
      engine.push(pim_id, &req);
      while (!engine.is_done(&req)) {
        co_await std::suspend_always{};
      }

      int status = req.rets.base.status;
      CHECK_VALID_STATUS(status);
      int expected_outs = 0;
      for (uint64_t key = lkey; key <= rkey; ++key) {
        if (1 <= key && key <= (uint64_t)table_size) {
          if (status == STATUS_SUCCESS) {
            [[maybe_unused]] uint64_t value = req.rets.values[expected_outs];
            assert(value == key + 7 || value == key + 77);
          }
          ++expected_outs;
        }
      }
      assert(expected_outs == req.rets.base.outs);
      if (expected_outs == 0) {
        assert(status == STATUS_FAILED);
      }
      else {
        assert(status == STATUS_SUCCESS);
      }

      // read-only; skip commit
      counters[worker_thd_core_id].increment(status);
    }
    else {
      // begin
      uint64_t xid = get_new_xid();
      uint64_t begin_csn = get_begin_csn();
      std::set<int> touched_pims;
      int status = STATUS_SUCCESS;
      
      if (txn_type == txn_type_read) {
        oltpim::request_get reqs[tests_per_txn];
        for (int i = 0; i < tests_per_txn; ++i) {
          uint64_t key = rand_key_distr(rg);
          auto &arg = reqs[i].args;
          arg.key = key;
          arg.xid = xid;
          arg.csn = begin_csn;
          arg.index_id = 0;
          arg.oid_query = 0;
          int pim_id = key_to_pim(key);
          touched_pims.insert(pim_id);
          engine.push(pim_id, &reqs[i]);
        }
        for (int i = 0; i < tests_per_txn; ++i) {
          while (!engine.is_done(&reqs[i])) {
            co_await std::suspend_always{};
          }
        }
        for (int i = 0; i < tests_per_txn; ++i) {
          uint64_t key = reqs[i].args.key;
          auto &ret = reqs[i].rets;

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
      }
      else if (txn_type == txn_type_update) {
        oltpim::request_update reqs[tests_per_txn];
        for (int i = 0; i < tests_per_txn; ++i) {
          uint64_t key = rand_key_distr(rg);
          auto &arg = reqs[i].args;
          arg.key = key;
          arg.xid = xid;
          arg.csn = begin_csn;
          arg.new_value = key + 77;
          arg.index_id = 0;
          int pim_id = key_to_pim(key);
          touched_pims.insert(pim_id);
          engine.push(pim_id, &reqs[i]);
        }
        for (int i = 0; i < tests_per_txn; ++i) {
          while (!engine.is_done(&reqs[i])) {
            co_await std::suspend_always{};
          }
        }
        for (int i = 0; i < tests_per_txn; ++i) {
          uint64_t key = reqs[i].args.key;
          auto &ret = reqs[i].rets;

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
      if (status == STATUS_SUCCESS) {
        oltpim::request_commit reqs[tests_per_txn];
        int cnt = 0;
        uint64_t end_csn = get_end_csn();
        for (int pim_id: touched_pims) {
          auto &arg = reqs[cnt].args;
          arg.xid = xid;
          arg.csn = end_csn;
          engine.push(pim_id, &reqs[cnt]);
          ++cnt;
        }
        for (int i = 0; i < cnt; ++i) {
          while (!engine.is_done(&reqs[i])) {
            co_await std::suspend_always{};
          }
        }
      }
      else {
        oltpim::request_abort reqs[tests_per_txn];
        int cnt = 0;
        for (int pim_id: touched_pims) {
          auto &arg = reqs[cnt].args;
          arg.xid = xid;
          engine.push(pim_id, &reqs[cnt]);
          ++cnt;
        }
        for (int i = 0; i < cnt; ++i) {
          while (!engine.is_done(&reqs[i])) {
            co_await std::suspend_always{};
          }
        }
      }

      counters[worker_thd_core_id].increment(status);
    }
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
      engine.register_worker_thread(sys_core_id);
    }

    {
      // initialize
      init_txn_args args;
      args.sys_core_id = sys_core_id;
      args.global_batch_offset = tid * init_batch_size;
      coro_scheduler init_coros(init_batch_size, init_txn, false, &args);
      while (true) {
        if (!init_coros.step()) break;
        if (done) break;
      }
    }

    // barrier
    init_barrier.arrive_and_wait();

    {
      // test
      test_txn_args args;
      args.sys_core_id = sys_core_id;
      coro_scheduler test_coros(batch_size, test_txn, true, &args);
      while (true) {
        test_coros.step();
        if (done) break;
      }
    }
  };

  // lap function
  auto lap_fn = [&]() {
    auto start_time = std::chrono::system_clock::now();
    auto last_time = start_time;
    counter last_counter;
    oltpim::rank_engine::stats last_stat;
    printf("sec:commits,fails,conflicts,CommitTPS,TotalTPS;engine_stats\n");
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
      auto curr_stat = oltpim::engine::g_engine.get_stats();

      // diff
      counter sec_counter = curr_counter - last_counter;
      last_counter = curr_counter;
      double commit_tps = (double)sec_counter.commits / real_sec;
      double total_tps = (double)sec_counter.total() / real_sec;
      auto sec_stat = curr_stat - last_stat;
      last_stat = curr_stat;

      printf("%lf:%lu,%lu,%lu,%lf,%lf;%s\n", 
        curr_sec, sec_counter.commits, sec_counter.fails, sec_counter.conflicts, 
        commit_tps, total_tps, sec_stat.to_string().c_str());
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
