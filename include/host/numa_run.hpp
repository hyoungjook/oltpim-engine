#pragma once

#include <array>
#include <atomic>
#include <iostream>
#include <numa.h>
#include <thread>
#include <vector>

/**
 * Provides lightweight NUMA-aware task distribution
 * based on modified parlaylib (https://github.com/cmuparlay/parlaylib).
 * Usage:
 *    // Create scheduler and worker threads
 *    numa_run::scheduler sched(num_workers_per_numa_node);
 * 
 *    // Execute per-numa task on all numa nodes
 *    sched.foreach_numa([](int numa_id) {
 *      // task that should be done in numa_node[numa_id]
 *    }, my_numa_id_of_caller);
 * 
 */
namespace numa_run {

namespace internal {
// https://github.com/cmuparlay/parlaylib/blob/36459f42a84207330eae706c47e6fab712e6a149/include/parlay/internal/work_stealing_job.h
struct Job {
  Job() : done(false) {}
  virtual ~Job() = default;
  void operator()() {
    execute();
    done.store(true, std::memory_order_release);
  }
  bool finished() const {
    return done.load(std::memory_order_acquire);
  }
 protected:
  virtual void execute() = 0;
  std::atomic<bool> done;
};

template <typename F>
struct alignas(64) JobImpl : Job {
  explicit JobImpl(F &_f) : Job(), f(_f) {}
  void execute() {f();}
 private:
  F &f;
};

template <typename F>
JobImpl<F> make_job(F& f) {return JobImpl(f);}

// https://github.com/cmuparlay/parlaylib/blob/36459f42a84207330eae706c47e6fab712e6a149/include/parlay/internal/work_stealing_deque.h
// Deque from Arora, Blumofe, and Plaxton (SPAA, 1998).
//
// Supports:
//
// push_bottom:   Only the owning thread may call this
// pop_bottom:    Only the owning thread may call this
// pop_top:       Non-owning threads may call this
//
template <typename J>
struct Deque {
  using qidx = unsigned int;
  using tag_t = unsigned int;
  // use std::atomic<age_t> for atomic access.
  // Note: Explicit alignment specifier required
  // to ensure that Clang inlines atomic loads.
  struct alignas(int64_t) age_t {
    tag_t tag;
    qidx top;
  };
  // align to avoid false sharing
  struct alignas(64) padded_job {
    std::atomic<J*> job;
  };

  static constexpr int q_size = 1000;
  std::atomic<qidx> bot;
  std::atomic<age_t> age;
  std::array<padded_job, q_size> deq;
  Deque() : bot(0), age(age_t{0, 0}) {}

  // Adds a new job to the bottom of the queue. Only the owning
  // thread can push new items. This must not be called by any
  // other thread.
  //
  // Returns true if the queue was empty before this push
  bool push_bottom(J *job) {
    auto local_bot = bot.load(std::memory_order_acquire);      // atomic load
    deq[local_bot].job.store(job, std::memory_order_release);  // shared store
    local_bot += 1;
    if (local_bot == q_size) {
      std::cerr << "internal error: scheduler queue overflow\n";
      std::abort();
    }
    bot.store(local_bot, std::memory_order_seq_cst);  // shared store
    return (local_bot == 1);
  }

  // Pop an item from the top of the queue, i.e., the end that is not
  // pushed onto. Threads other than the owner can use this function.
  //
  // Returns {job, empty}, where empty is true if job was the
  // only job on the queue, i.e., the queue is now empty
  std::pair<J*, bool> pop_top() {
    auto old_age = age.load(std::memory_order_acquire);    // atomic load
    auto local_bot = bot.load(std::memory_order_acquire);  // atomic load
    if (local_bot > old_age.top) {
      auto job = deq[old_age.top].job.load(std::memory_order_acquire);  // atomic load
      auto new_age = old_age;
      new_age.top = new_age.top + 1;
      if (age.compare_exchange_strong(old_age, new_age))
        return {job, (local_bot == old_age.top + 1)};
      else
        return {nullptr, (local_bot == old_age.top + 1)};
    }
    return {nullptr, true};
  }

  // Pop an item from the bottom of the queue. Only the owning
  // thread can pop from this end. This must not be called by any
  // other thread.
  J* pop_bottom() {
    Job* result = nullptr;
    auto local_bot = bot.load(std::memory_order_acquire);  // atomic load
    if (local_bot != 0) {
      local_bot--;
      bot.store(local_bot, std::memory_order_release);  // shared store
      std::atomic_thread_fence(std::memory_order_seq_cst);
      auto job =
          deq[local_bot].job.load(std::memory_order_acquire);  // atomic load
      auto old_age = age.load(std::memory_order_acquire);      // atomic load
      if (local_bot > old_age.top)
        result = job;
      else {
        bot.store(0, std::memory_order_release);  // shared store
        auto new_age = age_t{old_age.tag + 1, 0};
        if ((local_bot == old_age.top) &&
            age.compare_exchange_strong(old_age, new_age))
          result = job;
        else {
          age.store(new_age, std::memory_order_seq_cst);  // shared store
          result = nullptr;
        }
      }
    }
    return result;
  }
};
} // namespace internal

// https://github.com/cmuparlay/parlaylib/blob/36459f42a84207330eae706c47e6fab712e6a149/include/parlay/scheduler.h
// modified to support numa-aware scheduling
struct scheduler {
 private:
  using Job = internal::Job;
  // After YIELD_FACTOR * P unsuccessful steal attempts, a
  // a worker will sleep briefly for SLEEP_FACTOR * P nanoseconds
  // to give other threads a chance to work and save some cycles.
  constexpr static size_t YIELD_FACTOR = 200;
  constexpr static size_t SLEEP_FACTOR = 200;

  static inline thread_local uint32_t worker_id;

  uint32_t num_numa_nodes;
  uint32_t num_workers_per_numa_node;
  uint32_t num_total_workers;
  // Exclude cpus with id less than below. Use if you want to
  // schedule the workers only to the non-physical cores.
  uint32_t exclude_cpus_under;
  // N numa nodes and W workers per node: total NW workers.
  // The ordering is as follows:
  // numa_id = (global_id % N), local_id = (global_id / N)
  // so two workers are in the same numa node if
  // (global_id1 % N) == (global_id2 % N).
  std::vector<std::thread> worker_threads;

 public:
  explicit scheduler(size_t _num_workers_per_numa_node, size_t _exclude_cpus_under = 0)
    : num_numa_nodes(numa_max_node() + 1),
      num_workers_per_numa_node(_num_workers_per_numa_node),
      num_total_workers(num_numa_nodes * num_workers_per_numa_node),
      exclude_cpus_under(_exclude_cpus_under),
      deques(num_total_workers),
      attempts(num_total_workers),
      finished_flag(false)
  {
    if (num_workers_per_numa_node <= 0) {
        std::cerr << "At least 1 worker per numa node is required.\n";
        std::abort();
    }
    for (uint32_t id = 0; id < num_total_workers; ++id) {
      worker_threads.emplace_back([&, id]() {
        worker_id = id;
        pin_to_numa_node();
        worker();
      });
    }
  }

  ~scheduler() {
    shutdown();
  }

  // Only interface function
  // Execute f(numa_id) on each numa node.
  // For each numa_id in [0, num_numa_nodes);
  //    if (numa_id == my_numa_id):
  //        then the f is executed on the calling thread.
  //    else:
  //        the f is executed on the worker thread in
  //        that specific numa id.
  template <typename F>
  void foreach_numa(F&& f, uint32_t my_numa_id) {
    my_numa_id = my_numa_id % num_numa_nodes; // just to ensure
    foreach_numa_(f, my_numa_id, (my_numa_id + 1) % num_numa_nodes);
  }

 private:
  // Push to a worker in numa_id
  void spawn(Job* job, int numa_id) {
    deques[numa_id].push_bottom(job);
  }

  // Wait until the given condition is true
  template <typename F>
  void wait_until(F&& done) {
    while (true) {
      Job* job = get_job(done);
      if (!job) return;
      (*job)();
    }
  }

  // Pop from local stack
  Job* get_own_job() {
    return deques[worker_id].pop_bottom();
  }

  bool finished() const {
    return finished_flag.load(std::memory_order_acquire);
  }

  // Spawn to numa_id=i and recurse to (i + 1) % N.
  // If (i + 1) % N == my_numa_id, run here and return.
  template <typename F>
  void foreach_numa_(F&& f, uint32_t my_numa_id, uint32_t i) {
    auto curr_f = [&](){std::forward<F>(f)(i);};
    auto curr_job = internal::make_job(curr_f);
    spawn(&curr_job, i);
    uint32_t next_i = (i + 1) % num_numa_nodes;
    if (next_i == my_numa_id) {
      std::forward<F>(f)(my_numa_id);
    }
    else {
      foreach_numa_(f, my_numa_id, next_i);
    }
    wait_until([&](){return curr_job.finished();});
  }

 private:
  struct alignas(128) attempt {
    size_t val;
  };

  std::vector<internal::Deque<Job>> deques;
  std::vector<attempt> attempts;
  std::atomic<bool> finished_flag;

  void worker() {
    while (!finished()) {
      Job* job = get_job([&]() {return finished();});
      if (job) (*job)();
    }
  }

  void shutdown() {
    finished_flag.store(true, std::memory_order_release);
    for (uint32_t i = 0; i < num_total_workers; ++i) {
      worker_threads[i].join();
    }
  }

  // Find a job, first trying local stack, then random steals.
  //
  // Returns nullptr if break_early() returns true before a job
  // is found.
  template <typename F>
  Job* get_job(F&& break_early) {
    if (break_early()) return nullptr;
    Job* job = get_own_job();
    if (job) return job;
    else job = steal_job(std::forward<F>(break_early));
    return job;
  }

  // Find a job with random steals.
  //
  // Returns nullptr if break_early() returns true before a job
  // is found.
  template <typename F>
  Job* steal_job(F&& break_early) {
    while (true) {
      // By coupon collector's algorithm, this should touch all.
      for (size_t i = 0; i < YIELD_FACTOR * num_total_workers; ++i) {
        if (break_early()) return nullptr;
        Job* job = try_steal(worker_id);
        if (job) return job;
      }
      std::this_thread::sleep_for(std::chrono::nanoseconds(num_total_workers * 100));
    }
  }

  static size_t hash(uint64_t x) {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return static_cast<size_t>(x);
  }

  inline uint32_t local_id(uint32_t global_id) {return global_id / num_numa_nodes;}
  inline uint32_t numa_id(uint32_t global_id) {return global_id % num_numa_nodes;}
  inline uint32_t global_id(uint32_t numa_id, uint32_t local_id) {
    return local_id * num_numa_nodes + numa_id;
  }

  Job* try_steal(uint32_t id) {
    // use hashing to get "random" target in the same numa node
    uint32_t target = global_id(
      numa_id(id),
      (hash(id) + hash(attempts[id].val)) % num_workers_per_numa_node
    );
    ++attempts[id].val;
    auto [job, empty] = deques[target].pop_top();
    return job;
  }

  void pin_to_numa_node() {
    uint32_t num_cores = std::thread::hardware_concurrency();
    uint32_t numa_id_ = numa_id(worker_id);
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (uint32_t cpu = 0; cpu < num_cores; ++cpu) {
      if (cpu < exclude_cpus_under) {
        continue;
      }
      if ((uint32_t)numa_node_of_cpu(cpu) == numa_id_) {
        CPU_SET(cpu, &cpuset);
      }
    }
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
  }

};

} // namespace numa_run
