#include <iostream>
#include <cstdlib>
#include <random>
#include <thread>
#include <unistd.h>
#include <chrono>
#include <numa.h>
#include "engine.hpp"
#include "common.h"

thread_local std::random_device _random_device;
thread_local std::mt19937 _rand_gen(_random_device());

struct alignas(64) counter {
  uint64_t val = 0;
};

int main(int argc, char *argv[]) {
  int num_ranks_per_numa_node = 1;
  int num_threads_per_numa_node = 1;
  int batch_size = 1;
  int seconds = 10;

  int c;
  while ((c = getopt(argc, argv, "hr:t:b:n:")) != -1) {
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
    case 'h':
    default:
      printf("%s [-r num_ranks_per_numa_node] [-t num_threads_per_numa_node] [-b batch_size] [-n seconds]\n", argv[0]);
      exit(0);
    }
  }
  printf("Test run: %d ranks/numa, %d threads/numa %d batch size, %d seconds\n",
    num_ranks_per_numa_node, num_threads_per_numa_node, batch_size, seconds);

  oltpim::engine::config engine_config;
  engine_config.num_ranks_per_numa_node = num_ranks_per_numa_node;

  oltpim::engine engine;
  engine.init(engine_config);

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

  std::vector<counter> counters = std::vector<counter>(num_cores);
  volatile bool done = false;

  auto worker_fn = [&](int sys_core_id) {
    {
      // Pin thread to core
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      CPU_SET(sys_core_id, &cpuset);
      pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
    }

    std::uniform_int_distribution<> rand_type(0, 1);
    std::uniform_int_distribution<uint32_t> rand_arg(1, 16384);
    uint32_t num_pims = engine.num_pims();
    while (true) {
      oltpim::request reqs[batch_size];
      args_any_t args[batch_size];
      rets_any_t rets[batch_size];
      request_type_t types[batch_size];
      for (int i = 0; i < batch_size; ++i) {
        types[i] = (request_type_t)rand_type(_rand_gen);
        uint32_t rand_val = rand_arg(_rand_gen);
        uint8_t args_size = 0, rets_size = 0;
        switch (types[i]) {
        case request_type_insert: {
          args[i].insert.value = rand_val;
          args_size = sizeof(args_insert_t);
          rets_size = sizeof(rets_insert_t);
        } break;
        case request_type_get: {
          args[i].get.key[0] = rand_val;
          args_size = sizeof(args_get_t);
          rets_size = sizeof(rets_get_t);
        } break;
        }
        reqs[i] = oltpim::request(types[i], &args[i], &rets[i], args_size, rets_size);
        engine.push(rand_val % num_pims, &reqs[i], sys_core_id);
      }

      for (int i = 0; i < batch_size; ++i) {
        while (!engine.is_done(&reqs[i], sys_core_id));
      }

      for (int i = 0; i < batch_size; ++i) {
        //printf("T[%d]: %u -> %lu\n", sys_core_id, arg, ret);
        switch (types[i]) {
        case request_type_insert: {
          if (args[i].insert.value + 7 != rets[i].insert.old_value) {
            fprintf(stderr, "%u + 7 != %u wrong\n", args[i].insert.value, rets[i].insert.old_value);
            exit(1);
          }
        } break;
        case request_type_get: {
          if (args[i].get.key[0] + 8 != rets[i].get.value) {
            fprintf(stderr, "%u + 8 != %u wrong\n", args[i].get.key[0], rets[i].get.value);
            exit(1);
          }
        } break;
        default:
          fprintf(stderr, "unknown request_type\n");
          exit(1);
        }
      }

      counters[sys_core_id].val += batch_size;

      if (done) break;
    }
    engine.drain_all(sys_core_id);
  };

  auto lap_fn = [&]() {
    auto start_time = std::chrono::system_clock::now();
    auto last_time = start_time;
    uint64_t last_counter = 0;
    for (int i = 0; i < seconds; ++i) {
      sleep(1);
      auto curr_time = std::chrono::system_clock::now();
      double curr_sec = (curr_time - start_time).count() / 1000000000.0;
      double real_sec = (curr_time - last_time).count() / 1000000000.0;
      last_time = curr_time;

      // get counter
      uint64_t curr_counter = 0;
      for (int core_id: core_list) {
        curr_counter += counters[core_id].val;
      }

      // diff
      uint64_t sec_counter = curr_counter - last_counter;
      last_counter = curr_counter;
      double tps = (double)sec_counter / real_sec;

      printf("%lf sec: %lu (%lf TPS)\n", curr_sec, sec_counter, tps);
    }
    done = true;
  };

  std::vector<std::thread> threads;
  for (int sys_core_id: core_list) {
    threads.emplace_back(worker_fn, sys_core_id);
  }
  lap_fn();
  for (auto &thread: threads) {
    thread.join();
  }
}
