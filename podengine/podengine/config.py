"""Settings. Every default below was fixed before any backtest was run and is
taken from the published literature or common practice, not tuned to results.
Override any of them with a JSON file: `--config my.json`."""
import copy
import json

PROFILES = {
    # futures / portfolio margin: long and short, leverage allowed
    "unconstrained": {"allow_short": True, "gross_cap": 6.0, "asset_cap": 2.5,
                      "financing_spread": 0.003, "short_fee": 0.002, "short_rebate": 1.0},
    # ordinary margin account: long and short, 2x gross
    "margin": {"allow_short": True, "gross_cap": 2.0, "asset_cap": 1.0,
               "financing_spread": 0.015, "short_fee": 0.005, "short_rebate": 0.0},
    # cash account: long only, no leverage
    "cash": {"allow_short": False, "gross_cap": 1.0, "asset_cap": 0.40,
             "financing_spread": 0.0, "short_fee": 0.0, "short_rebate": 0.0},
}

DEFAULTS = {
    "profile": "unconstrained",
    "vol_target": 0.12,          # annualised portfolio volatility target
    "kelly_fraction": 0.5,       # cap vol target at this fraction of full Kelly
    "pod_vol": 0.10,             # each pod is sized to this stand-alone volatility
    "pod_gross_cap": 5.0,
    "rebalance": "weekly",       # weekly | monthly
    "exec_lag": 1,               # days between signal and trade (1 = trade next close)
    "trade_buffer": 0.20,        # no-trade band as a fraction of target position size
    "score_halflife": 5,         # days of smoothing on cross-sectional rank scores
    "min_history": 504,          # days of data before an asset can be traded
    "seed": 7,

    "pods": ["trend", "value_momentum", "defensive", "risk_parity", "reversal"],
    # Sceptical prior Sharpe per pod (roughly half the published figures, since
    # effects decay after publication). `reversal` is experimental: prior zero,
    # so it gets capital only if its own out-of-sample record earns it.
    "pod_prior_sharpe": {"trend": 0.30, "value_momentum": 0.30, "defensive": 0.20,
                         "risk_parity": 0.30, "reversal": 0.0},
    "prior_sharpe_sd": 0.35,          # prior is worth roughly 8 years of evidence
    "tilt_power": 1.0,           # 0 = equal risk per pod, 2 = full Kelly tilt

    "trend_lookbacks": [21, 63, 126, 252],
    "mom_window": [252, 21],     # 12-1 month momentum
    "value_years": 5.0,
    "reversal_days": 5,

    # volatility / covariance
    "vol_horizon": 21,          # size positions on the forecast for the next month
    "ewma_halflife": 21,
    "corr_halflife": 126,
    "corr_window": 756,
    "corr_shrink": 0.10,
    "regime_states": 3,
    "regime_refit_days": 252,
    "regime_min_days": 756,

    # governor
    "pod_stops": True,
    "pod_soft_dd": 1.5,          # in multiples of pod_vol: halve the pod's risk
    "pod_hard_dd": 2.25,         # switch the pod off until it recovers
    "hwm_window": 504,
    "turbulence_derisk": True,
    "turbulence_floor": 0.6,
    "turbulence_start_pct": 0.75,
    "port_stop": False,
    "port_soft_dd": 1.5,
    "port_hard_dd": 2.5,
    "risk_calibration": True,    # rescale risk forecasts by their own realised/predicted record
    "es_cap_mult": 3.2,          # 1-day 97.5% expected shortfall cap, in daily-vol-target units

    # costs
    # financing_spread (paid per year on exposure above 1x), short_fee (per year
    # on short notional) and short_rebate (share of the cash rate credited on
    # short proceeds: 1 for futures / institutional, 0 for a retail margin
    # account) come from the profile unless set here
    "cost_mult": 1.0,            # scale all trading costs (2.0 = stress test)

    # Monte Carlo
    "mc_paths": 4000,
    "mc_block": 10,              # mean bootstrap block length (days)
    "garch_alpha": 0.06,
    "garch_beta": 0.90,
    "aci_gamma": 0.02,
}


def load_config(path=None, **overrides):
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path) as f:
            user = json.load(f)
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    if cfg["profile"] not in PROFILES:
        raise ValueError(f"profile must be one of {list(PROFILES)}")
    for k, v in PROFILES[cfg["profile"]].items():
        cfg.setdefault(k, v)
    return cfg
