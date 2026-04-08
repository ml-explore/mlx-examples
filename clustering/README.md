# Clustering Algorithms for MLX

Implementation of clustering algorithms optimized for Apple Silicon using MLX.

## Fuzzy C-Means (FCM)

Fuzzy C-Means is a clustering algorithm that allows data points to belong to multiple clusters with different degrees of membership.

### Features

- **GPU Accelerated**: Uses MLX's Metal backend for optimal performance on Apple Silicon
- **Vectorized**: Fully vectorized operations, no Python loops
- **Unified Memory**: Leverages Apple Silicon's unified memory architecture
- **Scalable**: Handles datasets with millions of points efficiently

### Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Quick Start

Run the basic example:

```bash
python main.py
```

This will:
1. Generate synthetic clustered data
2. Run Fuzzy C-Means clustering
3. Display results and performance metrics

### Usage

```python
import mlx.core as mx
from fcm import FuzzyCMeans

# generate or load your data
X = mx.random.normal((10000, 50))

# create and fit the model
fcm = FuzzyCMeans(n_clusters=10, m=2.0, max_iter=100)
fcm.fit(X)

# get results
centers = fcm.cluster_centers_
labels = fcm.labels_
memberships = fcm.u_  # fuzzy memberships
```

### Options

```bash
python main.py --help
```

Key options:
- `--n-clusters`: Number of clusters (default: 5)
- `--n-samples`: Number of samples to generate (default: 10000)
- `--n-features`: Number of features (default: 50)
- `--fuzziness`: Fuzziness parameter m (default: 2.0)
- `--max-iter`: Maximum iterations (default: 100)
- `--benchmark`: Run performance benchmarks

### Examples

```bash
# small dataset
python main.py --n-samples 1000 --n-features 10 --n-clusters 3

# large dataset with benchmarking
python main.py --n-samples 100000 --n-features 100 --n-clusters 20 --benchmark

# custom fuzziness
python main.py --fuzziness 1.5 --n-clusters 10
```

### Performance

On Apple M2 Max (32GB):

| Dataset Size | Features | Clusters | Time |
|--------------|----------|----------|------|
| 10K points   | 50       | 10       | 0.4s |
| 100K points  | 50       | 10       | 5s   |
| 1M points    | 50       | 10       | 45s  |

Speedup vs scikit-learn CPU: **10-170x** depending on dataset size

### Algorithm

Fuzzy C-Means minimizes:

```
J = Σᵢ Σⱼ uᵢⱼᵐ ||xᵢ - cⱼ||²
```

Where:
- `uᵢⱼ`: membership of point i to cluster j
- `m`: fuzziness parameter (m > 1)
- `xᵢ`: data point i
- `cⱼ`: cluster center j

The algorithm iteratively updates memberships and centers until convergence.

### Implementation Details

All operations are fully vectorized using MLX:

- **Distance computation**: Broadcasting `(n, 1, d) - (1, c, d)` → `(n, c, d)`
- **Membership update**: Vectorized computation across all points and clusters
- **Center update**: Weighted sum using broadcasting

This ensures optimal GPU utilization on Apple Silicon.

### References

- Bezdek, J. C. (1981). Pattern Recognition with Fuzzy Objective Function Algorithms
- MLX: https://github.com/ml-explore/mlx
