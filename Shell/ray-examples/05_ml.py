#!/usr/bin/env python3
"""
Example 5: Distributed ML Training (Hyperparameter Search)

This example shows:
- Parallel hyperparameter search with Ray
- Training multiple models simultaneously
- Collecting and comparing results
"""

import ray
import time
import random
import math


def main():
    ray.init()

    # Simulate a simple model training function
    @ray.remote
    def train_model(learning_rate, num_layers, hidden_size, epochs=5):
        """
        Simulate training a neural network.
        In real code, this would use PyTorch/JAX/TensorFlow.
        """
        import socket

        # Simulate training time
        time.sleep(0.5)

        # Simulate loss based on hyperparameters
        # (In reality, this would be actual training)
        base_loss = 1.0
        loss = base_loss

        for epoch in range(epochs):
            # Simulate loss decreasing
            lr_factor = 1 - math.exp(-learning_rate * 10)
            layer_factor = 1 - (num_layers / 20)
            size_factor = 1 - (hidden_size / 1000)

            improvement = (lr_factor + layer_factor + size_factor) / 3
            loss = loss * (0.8 + random.random() * 0.1) * (1 - improvement * 0.1)

        return {
            "learning_rate": learning_rate,
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "final_loss": round(loss, 4),
            "trained_on": socket.gethostname()
        }

    # Define hyperparameter search space
    hyperparameters = [
        {"learning_rate": 0.001, "num_layers": 2, "hidden_size": 64},
        {"learning_rate": 0.001, "num_layers": 4, "hidden_size": 128},
        {"learning_rate": 0.01, "num_layers": 2, "hidden_size": 64},
        {"learning_rate": 0.01, "num_layers": 4, "hidden_size": 128},
        {"learning_rate": 0.01, "num_layers": 6, "hidden_size": 256},
        {"learning_rate": 0.1, "num_layers": 2, "hidden_size": 128},
        {"learning_rate": 0.1, "num_layers": 4, "hidden_size": 256},
        {"learning_rate": 0.1, "num_layers": 8, "hidden_size": 512},
    ]

    print(f"Starting hyperparameter search with {len(hyperparameters)} configurations...")
    print("Training all models in parallel...\n")

    start_time = time.time()

    # Launch all training runs in parallel
    futures = [
        train_model.remote(**params)
        for params in hyperparameters
    ]

    # Collect results as they complete
    results = ray.get(futures)

    elapsed = time.time() - start_time

    # Sort by loss
    results.sort(key=lambda x: x["final_loss"])

    print(f"All {len(results)} models trained in {elapsed:.2f}s")
    print("\nResults (sorted by loss):")
    print("-" * 70)
    print(f"{'LR':<8} {'Layers':<8} {'Hidden':<8} {'Loss':<10} {'Worker':<20}")
    print("-" * 70)

    for r in results:
        print(f"{r['learning_rate']:<8} {r['num_layers']:<8} {r['hidden_size']:<8} {r['final_loss']:<10} {r['trained_on']:<20}")

    print("-" * 70)
    print(f"\nBest configuration:")
    best = results[0]
    print(f"  Learning rate: {best['learning_rate']}")
    print(f"  Num layers: {best['num_layers']}")
    print(f"  Hidden size: {best['hidden_size']}")
    print(f"  Final loss: {best['final_loss']}")

    ray.shutdown()


if __name__ == "__main__":
    main()
