#include <iostream>
#include <cstdlib>
#include <random>
#include <thread>
#include <unistd.h>
#include <chrono>
#include "engine.hpp"

thread_local std::random_device _random_device;
thread_local std::mt19937 _rand_gen(_random_device());

struct alignas(64) counter {
  uint64_t val = 0;
};

int main(int argc, char *argv[]) {
  int num_ranks_per_numa_node = 1;
  int seconds = 10;
  int batch_size = 1;

  int c;
  while ((c = getopt(argc, argv, "r:t:b:")) != -1) {
    switch (c) {
    case 'r':
      num_ranks_per_numa_node = atoi(optarg);
      break;
    case 't':
      seconds = atoi(optarg);
      break;
    case 'b':
      batch_size = atoi(optarg);
      break;
    default:
      printf("%s [-r num_ranks_per_numa_node] [-t seconds]\n", argv[0]);
      exit(1);
    }
  }
  printf("Test run: %d ranks/numa, %d seconds, %d batch size\n",
    num_ranks_per_numa_node, seconds, batch_size);

  oltpim::engine::config engine_config;
  engine_config.num_ranks_per_numa_node = num_ranks_per_numa_node;

  oltpim::engine engine;
  engine.init(engine_config);

  int num_cores = std::thread::hardware_concurrency();

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

    std::uniform_int_distribution<uint32_t> rand_arg(1, 16384);
    uint32_t num_pims = engine.num_pims();
    while (true) {
      oltpim::request req[batch_size];
      uint32_t arg[batch_size];
      uint64_t ret[batch_size];
      for (int i = 0; i < batch_size; ++i) {
        arg[i] = rand_arg(_rand_gen);
        ret[i] = 0;
        req[i] = oltpim::request(&arg[i], &ret[i], sizeof(uint32_t), sizeof(uint64_t));
        engine.push(arg[i] % num_pims, &req[i], sys_core_id);
      }

      for (int i = 0; i < batch_size; ++i) {
        while (!engine.is_done(&req[i], sys_core_id));
      }

      for (int i = 0; i < batch_size; ++i) {
        //printf("T[%d]: %u -> %lu\n", sys_core_id, arg, ret);
        if (arg[i] + 7 != ret[i]) {
          fprintf(stderr, "%d != %lu wrong\n", arg[i], ret[i]);
          exit(1);
        }
      }

      counters[sys_core_id].val += batch_size;

      if (done) break;
    }

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
      for (int tid = 0; tid < num_cores; ++tid) {
        curr_counter += counters[tid].val;
      }

      // diff
      uint64_t sec_counter = curr_counter - last_counter;
      last_counter = curr_counter;
      double tps = (double)sec_counter / real_sec;

      printf("%lf sec: %lu (%lf TPS)\n", curr_sec, sec_counter, tps);
    }
    done = true;
  };

  //int num_cores = 1;
  std::vector<std::thread> threads;
  for (int sys_core_id = 0; sys_core_id < num_cores; sys_core_id++) {
    threads.emplace_back(worker_fn, sys_core_id);
  }
  lap_fn();
  for (int sys_core_id = 0; sys_core_id < num_cores; sys_core_id++) {
    threads[sys_core_id].join();
  }
}
