#!/usr/bin/env python3
"""
Example 1: Ray Hello World

This example shows:
- How to initialize Ray
- How to define a remote function with @ray.remote
- How to call remote functions and get results
- How tasks are distributed across the cluster
"""

import ray
import time


def main():
    # Initialize Ray (connects to the cluster)
    ray.init()

    # Show cluster info first
    print("=" * 60)
    print("CLUSTER INFO")
    print("=" * 60)
    nodes = ray.nodes()
    print(f"Total nodes: {len(nodes)}")
    for node in nodes:
        node_type = "Head" if node.get("Resources", {}).get("node:__internal_head__") else "Worker"
        alive = "alive" if node["Alive"] else "dead"
        node_name = node["NodeName"]
        cpu = node["Resources"].get("CPU", 0)
        mem = node["Resources"].get("memory", 0) / 1e9
        print(f"  - {node_name} ({node_type}, {alive})")
        print(f"    Resources: CPU={cpu}, Memory={mem:.1f}GB")
    print(f"Cluster resources: {ray.cluster_resources()}")
    print("=" * 60)

    # Define a remote function
    # num_cpus=1 ensures each task needs a full CPU, forcing distribution
    @ray.remote(num_cpus=1)
    def hello(name):
        import socket
        import os
        # Add delay so tasks run concurrently and must use different workers
        time.sleep(0.5)
        hostname = socket.gethostname()
        pid = os.getpid()
        return f"Hello {name}! I ran on {hostname} (PID: {pid})"

    # Call the remote function
    print("\nSingle task:")
    future = hello.remote("World")
    result = ray.get(future)
    print(f"  {result}")

    # Run tasks that MUST use both nodes
    # We have 2 CPUs total (1 per node), so 2 concurrent tasks will use both nodes
    print("")
    print("Parallel hellos (2 concurrent tasks, each needs 1 CPU):")
    print("Since we have 2 nodes with 1 CPU each, tasks MUST distribute!")
    print("")

    # Launch 4 tasks - with 2 CPUs total, they will run in 2 batches
    names = ["Alice", "Bob", "Charlie", "Diana"]
    futures = [hello.remote(name) for name in names]
    results = ray.get(futures)

    for r in results:
        print(f"  {r}")

    # Show which nodes were used
    print("")
    print("Distribution summary:")
    node_counts = {}
    for r in results:
        node = r.split("on ")[1].split(" ")[0]
        node_counts[node] = node_counts.get(node, 0) + 1
    for node, count in sorted(node_counts.items()):
        print(f"  {node}: {count} tasks")

    if len(node_counts) > 1:
        print("")
        print("SUCCESS: Tasks were distributed across multiple nodes!")
    else:
        print("")
        print("NOTE: All tasks ran on one node. This can happen if:")
        print("  - Worker node is not ready yet")
        print("  - Tasks completed before scheduler distributed them")

    ray.shutdown()


if __name__ == "__main__":
    main()
