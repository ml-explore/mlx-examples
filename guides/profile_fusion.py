# Copyright © 2025 Apple Inc.

import time

import mlx.core as mx


# A large tensor to stress memory bandwidth
shape = (4096, 4096)


def element_wise_chain(x):
    """A chain of element-wise operations.

    In eager mode, each line reads/writes to memory (slow).
    In compiled mode, these fuse into a single kernel (fast).
    """
    x = mx.sin(x)
    x = mx.cos(x) * 2
    x = mx.exp(x) + 1
    x = mx.log(mx.abs(x))
    return x


# Compile the function
compiled_op = mx.compile(element_wise_chain)


def benchmark():
    """Run benchmark comparing eager vs compiled execution"""
    x = mx.random.uniform(shape=shape)
    mx.eval(x)

    # 1. Warm-up
    print("Warming up...")
    for _ in range(10):
        mx.eval(element_wise_chain(x))
        mx.eval(compiled_op(x))

    # 2. Eager Benchmark
    print("Running Eager Mode (Capture this!)...")
    start = time.time()
    for _ in range(100):
        out = element_wise_chain(x)
        mx.eval(out)
    eager_time = time.time() - start
    print(f"Eager Time: {eager_time:.4f}s")

    # Visual separation in trace
    print("\n--- Separating traces (sleeping for 2 seconds) ---\n")
    time.sleep(2)

    # 3. Compiled Benchmark
    print("Running Compiled Mode (Capture this!)...")
    start = time.time()
    for _ in range(100):
        out = compiled_op(x)
        mx.eval(out)
    compiled_time = time.time() - start
    print(f"Compiled Time: {compiled_time:.4f}s")

    # Summary
    print(f"\nSpeedup: {eager_time / compiled_time:.2f}x")
    print("Check Instruments to see the difference in GPU utilization!")


if __name__ == "__main__":
    benchmark()
