#include <iostream>
#include <cstdlib>
#include <random>
#include <thread>
#include "engine.hpp"

thread_local std::random_device _random_device;
thread_local std::mt19937 _rand_gen(_random_device());

int main(int argc, char *argv[]) {

  oltpim::engine::config engine_config;
  engine_config.num_ranks_per_numa_node = atoi(argv[1]);

  oltpim::engine engine;
  engine.init(engine_config);

  auto worker_fn = [](oltpim::engine *engine, int sys_core_id) {
    {
      // Pin thread to core
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      CPU_SET(sys_core_id, &cpuset);
      pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
    }

    std::uniform_int_distribution<uint32_t> rand_arg(1, 16384);
    uint32_t num_pims = engine->num_pims();
    while (true) {
      oltpim::request req;
      uint32_t arg = rand_arg(_rand_gen);
      uint64_t ret = 0;
      req = oltpim::request(&arg, &ret, sizeof(uint32_t), sizeof(uint64_t));

      engine->push(arg % num_pims, &req, sys_core_id);
      while (!engine->is_done(&req, sys_core_id));

      printf("T[%d]: %u -> %lu\n", sys_core_id, arg, ret);
      if (arg + 7 != ret) {
        fprintf(stderr, "%d != %lu wrong\n", arg, ret);
        exit(1);
      }
    }

  };

  int num_cores = std::thread::hardware_concurrency();
  //int num_cores = 1;
  std::vector<std::thread> threads;
  for (int sys_core_id = 1; sys_core_id < num_cores; sys_core_id++) {
    threads.emplace_back(worker_fn, &engine, sys_core_id);
    threads.back().detach();
  }
  worker_fn(&engine, 0);
}
