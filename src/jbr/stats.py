"""OLS with a cluster bootstrap (resampling batted balls), and small helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

N_BOOT = 2000
SEED = 20251024


def design(df: pd.DataFrame, terms: list[str]) -> tuple[np.ndarray, list[str]]:
    """Columns for ``terms``: numeric names as-is, 'a:b' as products. Intercept first."""
    cols = [np.ones(len(df))]
    for t in terms:
        parts = t.split(":")
        col = np.ones(len(df))
        for p in parts:
            col = col * df[p].to_numpy(dtype=float)
        cols.append(col)
    return np.column_stack(cols), ["intercept", *terms]


def ols(df: pd.DataFrame, y: str, terms: list[str]) -> pd.Series:
    X, names = design(df, terms)
    beta, *_ = np.linalg.lstsq(X, df[y].to_numpy(dtype=float), rcond=None)
    return pd.Series(beta, index=names)


def cluster_bootstrap(df: pd.DataFrame, cluster: str, fn, n_boot: int = N_BOOT, seed: int = SEED) -> pd.DataFrame:
    """Apply ``fn`` (DataFrame -> Series) to ``n_boot`` resamples of whole clusters."""
    rng = np.random.default_rng(seed)
    groups = {k: g for k, g in df.groupby(cluster)}
    keys = np.array(list(groups))
    out = []
    for _ in range(n_boot):
        pick = rng.choice(keys, size=len(keys), replace=True)
        sample = pd.concat([groups[k] for k in pick], ignore_index=True)
        try:
            out.append(fn(sample))
        except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
            continue
    return pd.DataFrame(out)


def ci(draws: pd.Series, level: float = 0.95) -> tuple[float, float]:
    lo, hi = np.nanpercentile(draws, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def mean_ci(df: pd.DataFrame, value: str, cluster: str, n_boot: int = N_BOOT) -> dict:
    """Mean of ``value`` with a cluster-bootstrap CI."""
    if df.empty:
        return {"mean": None, "lo": None, "hi": None, "n": 0, "n_clusters": 0}
    draws = cluster_bootstrap(df, cluster, lambda s: pd.Series({"m": s[value].mean()}), n_boot)["m"]
    lo, hi = ci(draws)
    return {"mean": float(df[value].mean()), "lo": lo, "hi": hi, "n": int(len(df)),
            "n_clusters": int(df[cluster].nunique())}
