"""
Metrics for the constraint-aware MAS/DPS benchmark suite.

Two families:

1. Communication metrics (``CommLogger``) — a sender-anchored, status-aware
   accumulator mirroring the wrapper's instrumentation, for benchmarks that use
   the channel directly (the synthetic native-communication environments).

2. Control-theoretic metrics (module-level functions) — Lyapunov boundedness,
   consensus convergence rate, algebraic connectivity, integral error (IAE/ISE),
   and rate-distortion helpers, computed offline from logged trajectories.

All functions are pure NumPy.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
import numpy as np

from core import DELIVERED_STATUSES, RECV_STATUSES


# ============================================================
# Communication logger (for native-channel synthetic envs)
# ============================================================
class CommLogger:
    """Accumulates communication statistics from channel recv() metadata.

    Definitions match the wrapper's instrumentation so that synthetic and
    wrapped benchmarks report comparable numbers:
      - comm_mse:        mean over delivered messages of MSE(delivered, sent)
      - comm_zero_frac:  fraction of zeroed entries over all receive attempts
      - comm_age_mean:   mean Age-of-Information over delivered messages
      - comm_age_peak:   peak Age-of-Information
      - delivery_rate:   delivered_real / sent
      - bits_per_delivered: mean bits per delivered message
    """

    def __init__(self, bits_per_entry: float = 32.0):
        self.bits_per_entry = float(bits_per_entry)
        self.reset()

    def reset(self):
        self.mse_sum = 0.0; self.mse_n = 0
        self.zero_sum = 0; self.zero_n = 0
        self.age_sum = 0.0; self.age_n = 0; self.age_peak = 0.0
        self.sent = 0; self.recv_attempts = 0; self.delivered = 0
        self.delivered_bits = 0.0
        self.status_counts = {s: 0 for s in RECV_STATUSES}

    def on_send(self, n: int = 1):
        self.sent += int(n)

    def on_recv(self, delivered: np.ndarray, meta: Dict[str, Any]):
        self.recv_attempts += 1
        status = meta.get("status", "none")
        self.status_counts[status] = self.status_counts.get(status, 0) + 1

        rec = np.asarray(delivered, dtype=float).reshape(-1)
        if rec.size:
            self.zero_sum += int(np.sum(rec == 0.0))
            self.zero_n += int(rec.size)

        if status in DELIVERED_STATUSES:
            self.delivered += 1
            age = float(meta.get("age", 0))
            self.age_sum += age
            self.age_n += 1
            self.age_peak = max(self.age_peak, age)
            self.delivered_bits += rec.size * self.bits_per_entry
            src = meta.get("src_payload", None)
            if src is not None:
                src = np.asarray(src, dtype=float).reshape(-1)
                m = min(src.shape[0], rec.shape[0])
                if m > 0:
                    self.mse_sum += float(np.mean((src[:m] - rec[:m]) ** 2))
                    self.mse_n += 1

    def summary(self) -> Dict[str, float]:
        out = {
            "comm_mse": self.mse_sum / max(1, self.mse_n),
            "comm_zero_frac": self.zero_sum / max(1, self.zero_n),
            "comm_age_mean": self.age_sum / max(1, self.age_n),
            "comm_age_peak": float(self.age_peak),
            "sent_msgs": float(self.sent),
            "delivered_real_msgs": float(self.delivered),
            "delivery_rate": self.delivered / max(1, self.sent),
            "loss_rate": 1.0 - self.delivered / max(1, self.sent),
            "delivered_bits": float(self.delivered_bits),
            "bits_per_delivered": self.delivered_bits / max(1, self.delivered),
        }
        for s in RECV_STATUSES:
            out[f"status_{s}"] = float(self.status_counts.get(s, 0))
        return out


# ============================================================
# Control-theoretic metrics
# ============================================================
def disagreement(X: np.ndarray) -> np.ndarray:
    """
    Lyapunov candidate V(t) = sum_i ||x_i(t) - xbar(t)||^2 for a trajectory.

    X : array (T, n, d) or (T, n)  -> returns V of shape (T,)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 2:
        X = X[:, :, None]
    xbar = X.mean(axis=1, keepdims=True)          # (T,1,d)
    return np.sum((X - xbar) ** 2, axis=(1, 2))   # (T,)


def monotone_decrease_fraction(V: np.ndarray, tol: float = 1e-9) -> float:
    """Fraction of steps with V(t+1) <= V(t) (empirical Lyapunov descent rate)."""
    V = np.asarray(V, dtype=float)
    if V.size < 2:
        return float("nan")
    d = np.diff(V)
    return float(np.mean(d <= tol))


def ultimate_bound(V: np.ndarray, tail_frac: float = 0.2) -> float:
    """Estimate limsup V via the mean over the trajectory tail."""
    V = np.asarray(V, dtype=float)
    k = max(1, int(round(tail_frac * V.size)))
    return float(np.mean(V[-k:]))


def mean_drift(V: np.ndarray) -> float:
    """Mean one-step drift E[V(t+1)-V(t)]; > 0 indicates instability."""
    V = np.asarray(V, dtype=float)
    if V.size < 2:
        return float("nan")
    return float(np.mean(np.diff(V)))


def convergence_rate(V: np.ndarray, floor: float = 1e-12) -> Dict[str, float]:
    """
    Empirical geometric convergence rate rho: fit log V(t) ~ a + (log rho) t over
    the strictly-decreasing initial phase. Returns rho and the fit R^2.

    rho < 1  => convergence;  rho >= 1 => no contraction.
    """
    V = np.asarray(V, dtype=float)
    V = np.maximum(V, floor)
    if V.size < 3:
        return {"rho": float("nan"), "r2": float("nan"), "n_fit": 0}

    # use the phase from start until V stops meaningfully decreasing
    logV = np.log(V)
    # cut at the argmin to avoid the noisy floor region
    cut = int(np.argmin(V))
    cut = max(2, cut + 1)
    t = np.arange(cut, dtype=float)
    y = logV[:cut]
    if t.size < 3 or np.allclose(y, y[0]):
        return {"rho": 1.0, "r2": 0.0, "n_fit": int(t.size)}
    A = np.vstack([t, np.ones_like(t)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ np.array([slope, intercept])
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-30
    r2 = 1.0 - ss_res / ss_tot
    return {"rho": float(np.exp(slope)), "r2": float(r2), "n_fit": int(t.size)}


def time_to_threshold(V: np.ndarray, eps: float) -> Optional[int]:
    """First step index t with V(t) < eps, or None (censored / no convergence)."""
    V = np.asarray(V, dtype=float)
    idx = np.where(V < eps)[0]
    return int(idx[0]) if idx.size else None


def iae(err: np.ndarray) -> float:
    """Integral of Absolute Error over time (sum of |err| per step)."""
    return float(np.sum(np.abs(np.asarray(err, dtype=float))))


def ise(err: np.ndarray) -> float:
    """Integral of Squared Error over time."""
    e = np.asarray(err, dtype=float)
    return float(np.sum(e * e))


def laplacian_lambda2(adj: np.ndarray) -> float:
    """
    Algebraic connectivity (2nd-smallest Laplacian eigenvalue) of a graph.

    adj : (n, n) symmetric adjacency (0/1 or weighted). Larger lambda2 => faster
    guaranteed consensus convergence.
    """
    A = np.asarray(adj, dtype=float)
    A = 0.5 * (A + A.T)
    d = A.sum(axis=1)
    L = np.diag(d) - A
    w = np.linalg.eigvalsh(L)
    w = np.sort(w)
    return float(w[1]) if w.size >= 2 else 0.0


def effective_lambda2_series(eff_adjs: Sequence[np.ndarray]) -> float:
    """Mean algebraic connectivity of the per-step *delivered* communication graph."""
    vals = [laplacian_lambda2(a) for a in eff_adjs if a is not None and np.size(a)]
    return float(np.mean(vals)) if vals else 0.0


def rate_distortion_points(records: List[Dict[str, Any]],
                           rate_key: str = "bits_per_delivered",
                           distortion_key: str = "final_disagreement") -> Dict[str, np.ndarray]:
    """Collect (rate, distortion) pairs from a list of result records, sorted by rate."""
    pts = [(r.get(rate_key, np.nan), r.get(distortion_key, np.nan)) for r in records]
    pts = [(x, y) for x, y in pts if np.isfinite(x) and np.isfinite(y)]
    pts.sort(key=lambda p: p[0])
    if not pts:
        return {"rate": np.array([]), "distortion": np.array([])}
    rate, dist = zip(*pts)
    return {"rate": np.asarray(rate), "distortion": np.asarray(dist)}
