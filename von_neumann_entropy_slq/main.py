"""Stochastic Lanczos Quadrature for von Neumann entropy on MLX.

Compares two paths for `S(rho) = -Tr[rho ln rho]`:

  - exact NumPy `eigh` (CPU, float64)               -- reference
  - Stochastic Lanczos Quadrature on MLX            -- this example

SLQ replaces the `O(N^3)` eigendecomposition with `m` independent
`k`-step Lanczos recurrences, costing `O(k * m * N^2)` matvecs.  For
`k = 25, m = 20` the crossover is around `N >= 1000` on M1 Max; past
`N = 2000` the exact eigh GPU kernel hits Metal's command-buffer
timeout, so SLQ is the only path that completes.

Run::

    pip install -r requirements.txt
    python main.py                       # default sizes
    python main.py --sizes 100 500 1000 2000 4000
    python main.py --k 30 --m 30
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx

from slq import (
    random_density_matrix,
    von_neumann_entropy_exact,
    von_neumann_entropy_slq,
)


def time_call(fn, *args, **kwargs) -> tuple[float, float]:
    """Run `fn(*args)` once for warm-up, then time the second call."""
    fn(*args, **kwargs)
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return float(out), time.perf_counter() - t0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", type=int, nargs="+",
                   default=[100, 500, 1000, 2000])
    p.add_argument("--k", type=int, default=25,
                   help="Lanczos steps per probe (default 25)")
    p.add_argument("--m", type=int, default=20,
                   help="number of stochastic probes (default 20)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"# Stochastic Lanczos Quadrature for von Neumann entropy")
    print(f"# device : {mx.default_device()}")
    print(f"# k = {args.k}, m = {args.m}, seed = {args.seed}")
    print()
    header = (f"{'N':>5} | {'S_exact':>10} | {'S_slq':>10} | "
              f"{'rel_err':>8} | {'t_exact (s)':>12} | {'t_slq (s)':>10}")
    print(header)
    print("-" * len(header))

    for N in args.sizes:
        rho = random_density_matrix(N, seed=args.seed)

        S_exact, t_exact = time_call(von_neumann_entropy_exact, rho)
        S_slq, t_slq = time_call(
            von_neumann_entropy_slq, rho,
            k=args.k, m=args.m, seed=args.seed)

        rel = abs(S_slq - S_exact) / max(abs(S_exact), 1e-12)
        print(f"{N:>5} | {S_exact:>10.4f} | {S_slq:>10.4f} | "
              f"{rel:>8.1%} | {t_exact:>12.3f} | {t_slq:>10.3f}")


if __name__ == "__main__":
    main()
