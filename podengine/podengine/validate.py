"""The validation gate: is the result real, or the best of many tries?

* Probabilistic Sharpe Ratio (Bailey & Lopez de Prado 2012) - probability the
  true Sharpe exceeds a benchmark given sample length, skew and fat tails.
* Deflated Sharpe Ratio (2014) - the same, with the benchmark raised to the
  best Sharpe you would expect from N *skill-less* trials. N comes from the
  trial log, which records every configuration ever run on this data.
* Probability of Backtest Overfitting (Bailey, Borwein, Lopez de Prado & Zhu
  2017) - how often the configuration that looks best in-sample lands in the
  bottom half out-of-sample, over all symmetric splits of history.
* Stationary-bootstrap confidence interval for the Sharpe ratio.
"""
import hashlib
import itertools
import json
import os
import time

import numpy as np
from scipy import stats

ANN = 252
EULER = 0.5772156649015329


def sharpe(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    return float(x.mean() / sd * np.sqrt(ANN)) if sd > 0 else 0.0


def psr(x, sr_benchmark_ann=0.0):
    """P(true Sharpe > benchmark). `x` = daily excess returns."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 30 or x.std(ddof=1) == 0:
        return float("nan")
    sr = x.mean() / x.std(ddof=1)                       # per-period Sharpe
    sr0 = sr_benchmark_ann / np.sqrt(ANN)
    g3 = stats.skew(x)
    g4 = stats.kurtosis(x, fisher=False)
    den = np.sqrt(max(1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr * sr, 1e-12))
    return float(stats.norm.cdf((sr - sr0) * np.sqrt(n - 1) / den))


def expected_max_sharpe(n_trials, sr_sd_ann):
    """Expected best annualised Sharpe among n skill-less trials whose Sharpe
    ratios have standard deviation `sr_sd_ann` (False Strategy Theorem)."""
    n = max(int(n_trials), 1)
    if n == 1 or sr_sd_ann <= 0:
        return 0.0
    return float(sr_sd_ann * ((1 - EULER) * stats.norm.ppf(1 - 1.0 / n)
                              + EULER * stats.norm.ppf(1 - 1.0 / (n * np.e))))


def deflated_sharpe(x, n_trials, trial_sharpes=None):
    """Deflated Sharpe Ratio. The hurdle is the expected best Sharpe of
    `n_trials` skill-less trials. Their spread is taken as the larger of
    (a) the spread of the Sharpe ratios actually logged and (b) the sampling
    error of a zero-skill Sharpe over this sample. Near-duplicate trials make
    (a) small, so (b) keeps the test from going soft. Returns (dsr, hurdle, detail)."""
    x = np.asarray(x, dtype=float)
    years = max(len(x) / ANN, 1e-9)
    sd_zero = float(np.sqrt(1.0 / years))
    sd_trials = float(np.std(trial_sharpes, ddof=1)) if trial_sharpes is not None and len(trial_sharpes) >= 3 else 0.0
    h_trials = expected_max_sharpe(n_trials, sd_trials)
    h_zero = expected_max_sharpe(n_trials, sd_zero)
    hurdle = max(h_trials, h_zero)
    detail = {"hurdle_trials": h_trials, "dsr_trials": psr(x, h_trials),
              "hurdle_zero_skill": h_zero, "dsr_zero_skill": psr(x, h_zero)}
    return psr(x, hurdle), hurdle, detail


def min_track_record_years(x, sr_benchmark_ann=0.0, conf=0.95):
    """Years of data needed before the Sharpe beats the benchmark at `conf`."""
    x = np.asarray(x, dtype=float)
    sr = x.mean() / x.std(ddof=1)
    sr0 = sr_benchmark_ann / np.sqrt(ANN)
    if sr <= sr0:
        return float("inf")
    g3, g4 = stats.skew(x), stats.kurtosis(x, fisher=False)
    n = 1 + (1 - g3 * sr + (g4 - 1) / 4 * sr * sr) * (stats.norm.ppf(conf) / (sr - sr0)) ** 2
    return float(n / ANN)


def stationary_bootstrap_ci(x, n_boot=2000, block=21, conf=0.90, seed=0):
    """Confidence interval for the annualised Sharpe that respects serial
    dependence (Politis & Romano 1994)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    p = 1.0 / block
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n)
        jump = rng.random(n) < p
        jump[0] = True
        # a new block starts where `jump` is true; otherwise continue the previous block
        grp = np.cumsum(jump) - 1
        first = np.flatnonzero(jump)
        offs = np.arange(n) - first[grp]
        idx = (starts[first][grp] + offs) % n
        s = x[idx]
        sd = s.std(ddof=1)
        out[b] = s.mean() / sd * np.sqrt(ANN) if sd > 0 else 0.0
    a = (1 - conf) / 2
    return float(np.quantile(out, a)), float(np.quantile(out, 1 - a))


def pbo_cscv(M, n_blocks=10):
    """Probability of backtest overfitting by combinatorially symmetric
    cross-validation. M = (days x configurations) matrix of daily returns."""
    M = np.asarray(M, dtype=float)
    T, N = M.shape
    if N < 3 or T < n_blocks * 20:
        return {"pbo": float("nan"), "n_splits": 0}
    edges = np.linspace(0, T, n_blocks + 1).astype(int)
    blocks = [M[edges[i]:edges[i + 1]] for i in range(n_blocks)]
    sums = np.array([b.sum(axis=0) for b in blocks])
    sq = np.array([(b ** 2).sum(axis=0) for b in blocks])
    cnt = np.array([len(b) for b in blocks], dtype=float)

    def sr(sel):
        n = cnt[sel].sum()
        mu = sums[sel].sum(axis=0) / n
        var = sq[sel].sum(axis=0) / n - mu ** 2
        return mu / np.sqrt(np.maximum(var, 1e-18))

    logits, degr = [], []
    all_idx = set(range(n_blocks))
    for combo in itertools.combinations(range(n_blocks), n_blocks // 2):
        ins = list(combo)
        oos = sorted(all_idx - set(combo))
        s_in, s_out = sr(ins), sr(oos)
        best = int(np.argmax(s_in))
        rank = stats.rankdata(s_out)[best] / (N + 1.0)
        logits.append(np.log(rank / (1.0 - rank)))
        degr.append((s_in[best] * np.sqrt(ANN), s_out[best] * np.sqrt(ANN)))
    logits = np.array(logits)
    degr = np.array(degr)
    return {"pbo": float((logits <= 0).mean()), "n_splits": len(logits),
            "is_best_mean": float(degr[:, 0].mean()), "oos_of_best_mean": float(degr[:, 1].mean())}


# ----------------------------------------------------------------- trial log

def config_hash(cfg):
    keep = {k: v for k, v in cfg.items() if k not in ("seed", "mc_paths")}
    return hashlib.sha1(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()[:12]


def data_fingerprint(md):
    c = md.close
    return f"{md.source}|{c.shape[1]}|{c.index[0].date()}"


def log_trial(path, md, cfg, sr, label=""):
    """Append one line per (data, configuration) ever evaluated. The count of
    distinct configurations is the N that deflates the Sharpe ratio, so the
    more settings you try, the higher the bar becomes."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rec = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "data": data_fingerprint(md),
           "cfg": config_hash(cfg), "sharpe": round(float(sr), 4), "label": label}
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def read_trials(path, md):
    """Distinct configurations tried on this data set -> {hash: latest sharpe}."""
    out = {}
    if not os.path.exists(path):
        return out
    fp = data_fingerprint(md)
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("data") == fp:
                out[r["cfg"]] = r["sharpe"]
    return out


# --------------------------------------------------------- robustness grid

def variants(cfg):
    """One-at-a-time changes to the default settings. Used for the ablation
    table (does each layer earn its place?) and for PBO."""
    v = [("baseline", {}, True)]
    v += [("no pod stops", {"pod_stops": False}, True),
          ("no turbulence de-risking", {"turbulence_derisk": False}, True),
          ("equal risk per pod (no tilt)", {"tilt_power": 0.0}, True),
          ("full Kelly tilt", {"tilt_power": 2.0}, True),
          ("no Kelly cap", {"kelly_fraction": 50.0}, True),
          ("no trade band", {"trade_buffer": 0.0}, True),
          ("no risk-model calibration", {"risk_calibration": False}, True),
          ("portfolio stop on", {"port_stop": True}, True),
          ("monthly rebalance", {"rebalance": "monthly"}, True),
          ("slow trend only", {"trend_lookbacks": [126, 252]}, True),
          ("fast trend only", {"trend_lookbacks": [21, 63]}, True),
          # sensitivities, not strategy choices (excluded from PBO)
          ("trade at same close (no lag)", {"exec_lag": 0}, False),
          ("costs x2", {"cost_mult": 2.0}, False)]
    for pod in cfg["pods"]:
        if len(cfg["pods"]) > 1:
            v.append((f"without {pod}", {"pods": [p for p in cfg["pods"] if p != pod]}, True))
    return v
