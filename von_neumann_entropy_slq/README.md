# Von Neumann Entropy via Stochastic Lanczos

This example computes the **von Neumann entropy**

```
S(rho) = -Tr[ rho ln rho ]
```

of an `N x N` density matrix using **Stochastic Lanczos Quadrature
(SLQ)** on the Apple GPU.  SLQ replaces the `O(N^3)` eigendecomposition
of an exact `eigh`-based path with `m` independent `k`-step Lanczos
recurrences, costing `O(k * m * N^2)` matvecs.

The point of the example is the **crossover behaviour**: past
`N >= 1000` SLQ pulls ahead of the exact `eigh` path, and past
`N = 2000` the Metal `eigh` kernel hits its command-buffer timeout
so SLQ is the only path that completes at all.

## Basic usage

```python
import mlx.core as mx
from slq import (
    von_neumann_entropy_slq,
    von_neumann_entropy_exact,
    random_density_matrix,
)

rho = random_density_matrix(2000, seed=0)

# SLQ on the Metal GPU — order-of-tens of ms
S_slq = von_neumann_entropy_slq(rho, k=25, m=20)

# Exact reference (NumPy, float64) for accuracy comparison
S_exact = von_neumann_entropy_exact(rho)
```

## Running the example

```bash
pip install -r requirements.txt
python main.py                                  # 100, 500, 1000, 2000
python main.py --sizes 100 500 1000 2000 4000   # push past eigh timeout
python main.py --k 30 --m 30                    # tighter accuracy
```

Typical output on M1 Max (relative error against the float64 `eigh`
reference; absolute timings will scale with hardware):

```
    N |  S_exact |   S_slq | rel_err | t_exact (s) | t_slq (s)
----+----------+---------+---------+-------------+----------
  100|   4.10   |  4.07   |   0.7%  |     0.000   |   0.28
  500|   5.71   |  5.74   |   0.4%  |     0.012   |   0.28
 1000|   6.41   |  6.43   |   0.3%  |     0.10    |   0.29
 2000|   7.10   |  7.02   |   1.2%  |     0.88    |   0.31
 4000|   7.79   |  7.76   |   0.4%  |     8.60    |   0.38
```

This pedagogical version is sequential over probes and does not batch
them — even so the `O(k * m * N^2)` scaling pays off at `N = 4000`
with a ~22x win over CPU `eigh`.  Batched probes plus `mx.compile`
(see `mlx-qre`) push that further into the hundreds.

## When to reach for SLQ

- `N >= 1000` and you need many `S(rho)` evaluations (parameter scans,
  MCMC, training-loop regularisers).
- `N >= 2000` where the GPU `eigh` path stops completing.
- Trace-of-matrix-function workloads more generally:
  `Tr[A ln A]`, `log det A`, `Tr[exp(A)]` are all in scope —
  swap `_xlogx_quadrature` for the corresponding `f(theta)`.
- Quantum information / lattice field theory entanglement entropies
  where the reduced density matrix dimension grows exponentially in
  subsystem size.

## Implementation notes

- `slq.py` is intentionally short and pedagogical (~150 lines).  A
  production version with batched probes, `mx.compile` fusion,
  full re-orthogonalisation, plus quantum-relative-entropy and Petz
  recovery estimators lives in
  [`mlx-qre`](https://github.com/akaiHuang/mlx-qre) on PyPI.
- The inner `k x k` tridiagonal `eigh` runs on NumPy (`k <= 30`,
  CPU is faster than GPU dispatch for that size).  All `O(N^2)`
  matvecs and inner products run on the Metal GPU.
- Probes are real Rademacher (`+/- 1`).  Switching to complex Rademacher
  is a one-line change for complex Hermitian density matrices.
- The Hutchinson estimator has variance roughly `1/sqrt(m)`, so a few
  probes can produce a small bias at small `N`; bump `--m` if that
  bothers you.

## References

- S. Ubaru, J. Chen & Y. Saad, *Fast estimation of `tr(f(A))` via
  stochastic Lanczos quadrature*, SIAM J. Matrix Anal. Appl. 38(4),
  1075-1099 (2017).
