#!/usr/bin/env python3
"""
Example 3: Ray Actors (Stateful Workers)

This example shows:
- How to create stateful workers with @ray.remote class
- Actors maintain state between method calls
- Actors can run on different nodes in the cluster
"""

import ray


def main():
    ray.init()

    # Define an Actor (stateful worker)
    @ray.remote
    class Counter:
        def __init__(self, start_value=0):
            self.value = start_value
            import socket
            self.hostname = socket.gethostname()

        def increment(self):
            self.value += 1
            return self.value

        def get_value(self):
            return self.value

        def get_info(self):
            return {"value": self.value, "running_on": self.hostname}

    # Create actor instances
    counter1 = Counter.remote(0)
    counter2 = Counter.remote(100)

    # Call methods on actors
    print("Incrementing counter1 three times:")
    for _ in range(3):
        result = ray.get(counter1.increment.remote())
        print(f"  counter1 = {result}")

    print("\nIncrementing counter2 twice:")
    for _ in range(2):
        result = ray.get(counter2.increment.remote())
        print(f"  counter2 = {result}")

    # Get actor info
    info1 = ray.get(counter1.get_info.remote())
    info2 = ray.get(counter2.get_info.remote())

    print(f"\nCounter1 info: {info1}")
    print(f"Counter2 info: {info2}")

    # Actors maintain separate state!
    print("\nKey insight: Each actor maintains its own state independently")

    ray.shutdown()


if __name__ == "__main__":
    main()
