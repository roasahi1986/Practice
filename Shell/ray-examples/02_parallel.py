#!/usr/bin/env python3
"""
Example 2: Parallel Computation

This example shows:
- How Ray parallelizes work across the cluster
- Comparing sequential vs parallel execution time
- How tasks are distributed to workers
"""

import ray
import time


def main():
    ray.init()

    # A CPU-intensive task
    @ray.remote
    def compute_heavy(x):
        """Simulate heavy computation"""
        import socket
        total = 0
        for i in range(5_000_000):
            total += i * x
        return {"result": total % 1000, "worker": socket.gethostname()}

    # Sequential execution (for comparison)
    print("Sequential execution (4 tasks)...")
    start = time.time()
    sequential_results = []
    for i in range(4):
        # Call without .remote() - runs locally
        total = 0
        for j in range(5_000_000):
            total += j * i
        sequential_results.append(total % 1000)
    sequential_time = time.time() - start
    print(f"  Time: {sequential_time:.2f}s")

    # Parallel execution with Ray
    print("\nParallel execution (4 tasks)...")
    start = time.time()
    futures = [compute_heavy.remote(i) for i in range(4)]
    parallel_results = ray.get(futures)
    parallel_time = time.time() - start
    print(f"  Time: {parallel_time:.2f}s")

    print(f"\nSpeedup: {sequential_time / parallel_time:.2f}x")

    print("\nTask distribution:")
    for i, r in enumerate(parallel_results):
        print(f"  Task {i}: computed on {r['worker']}")

    ray.shutdown()


if __name__ == "__main__":
    main()
