"""
Example of Fuzzy C-Means clustering with MLX.
"""

import argparse
import time

import mlx.core as mx
import numpy as np

from fcm import FuzzyCMeans


def generate_clustered_data(n_samples, n_features, n_clusters, random_state=None):
    """Generate synthetic clustered data."""
    if random_state is not None:
        mx.random.seed(random_state)

    centers = mx.random.normal((n_clusters, n_features)) * 5.0
    assignments = mx.random.randint(0, n_clusters, (n_samples,))

    points = []
    for i in range(n_samples):
        cluster = int(assignments[i])
        point = centers[cluster] + mx.random.normal((n_features,)) * 0.5
        points.append(point)

    X = mx.stack(points)

    return X, assignments


def main():
    parser = argparse.ArgumentParser(description="Fuzzy C-Means clustering with MLX")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10000,
        help="Number of samples to generate",
    )
    parser.add_argument("--n-features", type=int, default=50, help="Number of features")
    parser.add_argument("--n-clusters", type=int, default=5, help="Number of clusters")
    parser.add_argument(
        "--fuzziness",
        type=float,
        default=2.0,
        help="Fuzziness parameter m (must be > 1)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="Maximum number of iterations",
    )
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark",
    )
    args = parser.parse_args()

    print("=== Fuzzy C-Means Clustering with MLX ===\n")
    print(f"Device: {mx.default_device()}")
    print(
        f"Dataset: {args.n_samples} samples, {args.n_features} features, {args.n_clusters} clusters\n"
    )

    print("Generating data...")
    X, true_labels = generate_clustered_data(
        args.n_samples,
        args.n_features,
        args.n_clusters,
        random_state=args.random_seed,
    )

    print("Running Fuzzy C-Means...")
    fcm = FuzzyCMeans(
        n_clusters=args.n_clusters,
        m=args.fuzziness,
        max_iter=args.max_iter,
        random_state=args.random_seed,
    )

    start = time.time()
    fcm.fit(X)
    mx.eval(fcm.cluster_centers_)
    elapsed = time.time() - start

    print(f"✓ Converged in {fcm.n_iter_} iterations")
    print(f"✓ Time: {elapsed:.4f}s")
    print(f"✓ Cluster centers shape: {fcm.cluster_centers_.shape}")
    print(f"✓ Labels shape: {fcm.labels_.shape}\n")

    labels_np = np.array(fcm.labels_)
    print("Cluster distribution:")
    for i in range(args.n_clusters):
        count = np.sum(labels_np == i)
        pct = count / args.n_samples * 100
        print(f"  Cluster {i}: {count} points ({pct:.1f}%)")

    print("\nFuzzy memberships (first 5 points):")
    u_np = np.array(fcm.u_[:5])
    for i, memberships in enumerate(u_np):
        print(f"  Point {i}: {memberships}")

    if args.benchmark:
        print("\n=== Performance Benchmark ===\n")
        sizes = [
            (1000, 10, 5),
            (5000, 20, 10),
            (10000, 30, 15),
        ]

        for n_samples, n_features, n_clusters in sizes:
            print(
                f"Dataset: {n_samples} samples, {n_features} features, {n_clusters} clusters"
            )

            X_bench, _ = generate_clustered_data(
                n_samples, n_features, n_clusters, random_state=42
            )

            fcm_bench = FuzzyCMeans(
                n_clusters=n_clusters, m=2.0, max_iter=20, random_state=42
            )

            start = time.time()
            fcm_bench.fit(X_bench)
            mx.eval(fcm_bench.cluster_centers_)
            elapsed = time.time() - start

            print(f"  Time: {elapsed:.4f}s")
            print(f"  Iterations: {fcm_bench.n_iter_}\n")

    print("Done!")


if __name__ == "__main__":
    main()
