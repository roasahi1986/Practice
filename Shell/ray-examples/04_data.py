#!/usr/bin/env python3
"""
Example 4: Distributed Data Processing

This example shows:
- Processing data in parallel with Ray
- Map-reduce style operations
- Object store for sharing data between tasks
"""

import ray
import time


def main():
    ray.init()

    # Simulate a dataset
    data = list(range(1000))

    # Put data in Ray object store (shared memory)
    data_ref = ray.put(data)

    @ray.remote
    def process_chunk(data_ref, start, end):
        """Process a chunk of data"""
        import socket
        chunk = data_ref[start:end]

        # Simulate processing (e.g., feature extraction)
        result = sum(x ** 2 for x in chunk)

        return {
            "start": start,
            "end": end,
            "result": result,
            "worker": socket.gethostname()
        }

    @ray.remote
    def reduce_results(results):
        """Combine results from all chunks"""
        total = sum(r["result"] for r in results)
        return total

    # Split data into chunks and process in parallel
    print("Processing 1000 items in parallel chunks...")
    chunk_size = 100
    num_chunks = len(data) // chunk_size

    start_time = time.time()

    # Map phase: process chunks in parallel
    chunk_futures = [
        process_chunk.remote(data_ref, i * chunk_size, (i + 1) * chunk_size)
        for i in range(num_chunks)
    ]

    # Wait for all chunks to complete
    chunk_results = ray.get(chunk_futures)

    # Reduce phase: combine results
    final_result = ray.get(reduce_results.remote(chunk_results))

    elapsed = time.time() - start_time

    print(f"\nProcessing complete in {elapsed:.3f}s")
    print(f"Final result (sum of squares): {final_result}")

    print("\nChunk distribution:")
    workers = {}
    for r in chunk_results:
        worker = r["worker"]
        workers[worker] = workers.get(worker, 0) + 1
    for worker, count in workers.items():
        print(f"  {worker}: processed {count} chunks")

    ray.shutdown()


if __name__ == "__main__":
    main()
