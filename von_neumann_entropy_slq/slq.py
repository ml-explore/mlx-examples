"""Stochastic Lanczos Quadrature for Tr[f(A)] on Apple Silicon (MLX).

Implements the trace estimator of Ubaru, Chen & Saad (2017) for a
matrix function applied to a Hermitian operator:

    Tr[f(A)] ~ (1 / m) * sum_{i=1..m} v_i^T f(A) v_i

where each `v_i^T f(A) v_i` is approximated via a `k`-step Lanczos
recurrence on `A` starting from `v_i`.  The eigendecomposition of the
small `k x k` tridiagonal yields a Gaussian-quadrature rule for
`f(A)`, so we pay `O(k * N^2)` matvecs per probe instead of the
`O(N^3)` of a full `eigh` on `A`.

Specialising to `f(x) = x ln x` and using Rademacher probes whose
covariance is the identity, this gives the **von Neumann entropy**

    S(rho) = -Tr[rho ln rho]

of an `N x N` density matrix without any eigendecomposition.

This file is intentionally short and pedagogical (~150 lines).  A
production-quality version with batched probes, `mx.compile` fusion,
and full reorthogonalisation lives in
[mlx-qre](https://github.com/akaiHuang/mlx-qre) on PyPI, alongside
the Petz recovery, channel and quantum-relative-entropy estimators.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np


# ---------------------------------------------------------------------------
# Lanczos tridiagonalisation
# ---------------------------------------------------------------------------


def lanczos_tridiag(
    A: mx.array,
    v0: mx.array,
    k: int,
    *,
    reorth: bool = True,
) -> tuple[mx.array, mx.array]:
    """Run `k` steps of Lanczos on `A` starting from `v0`.

    Parameters
    ----------
    A : (N, N) Hermitian array on the MLX device.
    v0 : (N,) starting vector — caller is responsible for normalisation.
    k : number of Lanczos steps.
    reorth : if True, run modified Gram-Schmidt against the full Krylov
        basis at each step.  Doubles work per step but suppresses ghost
        eigenvalues that float32 + small `k` produce otherwise.

    Returns
    -------
    alpha : (k,) diagonal of the tridiagonal `T_k`.
    beta  : (k,) sub-diagonal.  `beta[k-1]` is the post-last residual
            norm and is not used in the tridiagonal itself.
    """
    Q = [v0]
    alpha = []
    beta = []

    q_prev = mx.zeros_like(v0)
    b_prev = 0.0

    for j in range(k):
        q = Q[-1]
        Aq = A @ q
        a = mx.real(mx.sum(mx.conj(q) * Aq))
        alpha.append(a)

        r = Aq - a * q - b_prev * q_prev

        if reorth:
            for qi in Q:
                r = r - mx.sum(mx.conj(qi) * r) * qi

        b = mx.sqrt(mx.real(mx.sum(mx.conj(r) * r)) + 1e-30)
        beta.append(b)

        if j + 1 < k:
            q_next = r / b
            Q.append(q_next)
            q_prev = q
            b_prev = b
        mx.eval(a, b)

    return mx.stack(alpha), mx.stack(beta)


# ---------------------------------------------------------------------------
# Quadrature on the small tridiagonal
# ---------------------------------------------------------------------------


def _build_tridiag(alpha: mx.array, beta: mx.array) -> np.ndarray:
    """Build the small `k x k` symmetric tridiagonal as a NumPy array.

    `k` is at most a few dozen, so we materialise on the host and use
    `numpy.linalg.eigh` for the inner eigendecomposition — the cost is
    negligible compared with the `O(k * N^2)` Lanczos matvecs that
    produced `(alpha, beta)` in the first place.
    """
    a = np.asarray(alpha, dtype=np.float64)
    b = np.asarray(beta, dtype=np.float64)
    k = a.shape[0]
    T = np.zeros((k, k), dtype=np.float64)
    T[np.arange(k), np.arange(k)] = a
    if k > 1:
        offd = b[: k - 1]
        T[np.arange(k - 1), np.arange(1, k)] = offd
        T[np.arange(1, k), np.arange(k - 1)] = offd
    return T


def _xlogx_quadrature(alpha: mx.array, beta: mx.array, v_norm_sq: float) -> float:
    """Estimate `v^T (A ln A) v` from the Lanczos tridiagonal of `A`.

    Builds `T = diag(alpha) + diag(beta_<k-1>, +/- 1)`, computes its
    eigendecomposition `T = U diag(theta) U^T`, and returns
    `v_norm_sq * sum_j (U[0, j])^2 * theta_j * ln(theta_j)`.
    """
    T = _build_tridiag(alpha, beta)
    theta, U = np.linalg.eigh(T)
    weights = U[0, :] ** 2
    safe = theta > 0.0
    val = float(np.sum(weights[safe] * theta[safe] * np.log(theta[safe])))
    return v_norm_sq * val


# ---------------------------------------------------------------------------
# Public estimator
# ---------------------------------------------------------------------------


def stochastic_lanczos_logtr(
    A: mx.array,
    k: int = 25,
    m: int = 20,
    *,
    seed: int = 0,
) -> float:
    """Estimate `Tr[A ln A]` with `m` Rademacher probes and `k` Lanczos steps.

    `A` should be Hermitian PSD (eigenvalues > 0). For density matrices
    the trace is normalised, so `Tr[A ln A]` is non-positive and equal to
    `-S(A)` where `S` is the von Neumann entropy.
    """
    N = A.shape[0]
    rng = np.random.default_rng(seed)
    probes = (rng.integers(0, 2, size=(m, N), dtype=np.int8) * 2 - 1).astype(np.float32)
    total = 0.0
    for v_np in probes:
        v_norm_sq = float(np.dot(v_np, v_np))                  # = N for Rademacher
        v0 = mx.array(v_np / math.sqrt(v_norm_sq))
        alpha, beta = lanczos_tridiag(A, v0, k)
        mx.eval(alpha, beta)
        total += _xlogx_quadrature(alpha, beta, v_norm_sq)
    return total / m


def von_neumann_entropy_slq(
    rho: mx.array, k: int = 25, m: int = 20, *, seed: int = 0
) -> float:
    """`S(rho) = -Tr[rho ln rho]` via Stochastic Lanczos Quadrature."""
    return -stochastic_lanczos_logtr(rho, k=k, m=m, seed=seed)


# ---------------------------------------------------------------------------
# Reference: exact eigh-based entropy for accuracy comparison
# ---------------------------------------------------------------------------


def von_neumann_entropy_exact(rho: mx.array) -> float:
    """`S(rho) = -Tr[rho ln rho]` via NumPy `eigh` — for accuracy reference."""
    rho_np = np.asarray(rho, dtype=np.float64)
    eigs = np.linalg.eigvalsh(rho_np)
    eigs = eigs[eigs > 1e-12]
    return float(-np.sum(eigs * np.log(eigs)))


def random_density_matrix(N: int, *, seed: int = 0) -> mx.array:
    """Random PSD trace-1 matrix via Wishart sampling (real, float32)."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N, N)).astype(np.float32)
    rho = A @ A.T
    rho = rho / np.trace(rho)
    return mx.array(rho)
