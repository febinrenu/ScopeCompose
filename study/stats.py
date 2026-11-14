"""Inference for the study: every test treats the **document** as the unit.

Several questions are asked about the same gov.uk page, share its chunks and
its retrieval behaviour, and so are not independent. Intervals and tests
therefore resample pages, not questions. The interval and test implementations
are the repository's (``metrics.cluster``, ``metrics.stats``); this module only
arranges the study's data into the clustered shape they expect, and adds a
vectorised sign-flip test because the pure-Python one is too slow at this size.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np

from metrics.cluster import bca
from metrics.stats import Interval, holm_adjust, mcnemar  # noqa: F401  (re-exported for the analysis)

N_RESAMPLES = 2000
N_FLIPS = 20000


def _clusters(values: Mapping[str, float], cluster_of: Mapping[str, int]) -> list[list[float]]:
    groups: dict[int, list[float]] = defaultdict(list)
    for key, v in values.items():
        groups[cluster_of[key]].append(float(v))
    return [groups[k] for k in sorted(groups)]


def _flat_mean(units: Sequence[Sequence[float]]) -> float:
    total = sum(len(u) for u in units)
    return sum(sum(u) for u in units) / total if total else 0.0


def mean_ci(values: Mapping[str, float], cluster_of: Mapping[str, int], *, seed: int = 0) -> Interval:
    """BCa interval for a mean over questions, resampling documents."""
    return bca(_clusters(values, cluster_of), _flat_mean, n_resamples=N_RESAMPLES, seed=seed)


def sign_flip_p(diffs: Mapping[str, float], cluster_of: Mapping[str, int], *, seed: int = 0) -> float:
    """Two-sided paired permutation p-value for ``mean(diff) == 0``, flipping whole documents."""
    sums = np.array([sum(c) for c in _clusters(diffs, cluster_of)], dtype=np.float64)
    if not len(sums) or not np.any(sums):
        return 1.0
    obs = abs(sums.sum())
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(N_FLIPS // 2000):
        signs = rng.choice((-1.0, 1.0), size=(2000, len(sums)))
        extreme += int((np.abs(signs @ sums) >= obs - 1e-12).sum())
    return (extreme + 1) / (N_FLIPS + 1)


def paired(a: Mapping[str, float], b: Mapping[str, float], cluster_of: Mapping[str, int], *, seed: int = 0) -> dict:
    """``a - b`` on the questions both have: mean difference, BCa interval, sign-flip p."""
    keys = sorted(set(a) & set(b))
    diffs = {k: float(a[k]) - float(b[k]) for k in keys}
    iv = mean_ci(diffs, cluster_of, seed=seed)
    return {"n": len(keys), "a": float(np.mean([a[k] for k in keys])) if keys else float("nan"),
            "b": float(np.mean([b[k] for k in keys])) if keys else float("nan"),
            "diff": iv.point, "low": iv.low, "high": iv.high, "p": sign_flip_p(diffs, cluster_of, seed=seed)}


def interval_dict(iv: Interval) -> dict:
    return {"point": iv.point, "low": iv.low, "high": iv.high, "n_clusters": iv.n}
