"""Default instruments, asset classes, cross-sectional groups and futures specs.

Research is done on liquid ETFs because they have clean, adjusted, free price
history. Execution can be in the ETF itself or in the matching futures
contract; `FUTURES` maps each ETF to contracts and their multipliers.

`cls`   = asset class, used for risk balancing and cost assumptions.
`group` = peer group for long/short cross-sectional pods (None = not ranked).
"""

# ticker: (name, cls, group)
DEFAULT_UNIVERSE = {
    # --- US equity indices
    "SPY": ("S&P 500", "equity", "equity_index"),
    "QQQ": ("Nasdaq 100", "equity", "equity_index"),
    "IWM": ("Russell 2000", "equity", "equity_index"),
    "EFA": ("Developed ex-US", "equity", "equity_index"),
    "EEM": ("Emerging markets", "equity", "equity_index"),
    # --- US sectors
    "XLK": ("Technology", "equity", "sector"),
    "XLF": ("Financials", "equity", "sector"),
    "XLE": ("Energy", "equity", "sector"),
    "XLV": ("Health care", "equity", "sector"),
    "XLI": ("Industrials", "equity", "sector"),
    "XLP": ("Staples", "equity", "sector"),
    "XLY": ("Discretionary", "equity", "sector"),
    "XLU": ("Utilities", "equity", "sector"),
    "XLB": ("Materials", "equity", "sector"),
    # --- Countries
    "EWJ": ("Japan", "equity", "country"),
    "EWG": ("Germany", "equity", "country"),
    "EWU": ("United Kingdom", "equity", "country"),
    "EWC": ("Canada", "equity", "country"),
    "EWA": ("Australia", "equity", "country"),
    "EWZ": ("Brazil", "equity", "country"),
    "EWW": ("Mexico", "equity", "country"),
    "EWY": ("South Korea", "equity", "country"),
    "EWT": ("Taiwan", "equity", "country"),
    "FXI": ("China large-cap", "equity", "country"),
    # --- Rates
    "IEI": ("Treasuries 3-7y", "rates", "bonds"),
    "IEF": ("Treasuries 7-10y", "rates", "bonds"),
    "TLT": ("Treasuries 20y+", "rates", "bonds"),
    "TIP": ("TIPS", "rates", "bonds"),
    # --- Credit
    "LQD": ("IG corporates", "credit", "bonds"),
    "HYG": ("High yield", "credit", "bonds"),
    "EMB": ("EM sovereign USD", "credit", "bonds"),
    # --- Real assets
    "GLD": ("Gold", "real", "commodity"),
    "SLV": ("Silver", "real", "commodity"),
    "DBC": ("Broad commodities", "real", "commodity"),
    "USO": ("Crude oil", "real", "commodity"),
    "DBA": ("Agriculture", "real", "commodity"),
    "VNQ": ("US REITs", "real", None),
    # --- Currencies (vs USD)
    "FXE": ("Euro", "fx", "fx"),
    "FXY": ("Yen", "fx", "fx"),
    "FXB": ("Pound", "fx", "fx"),
    "FXA": ("Aussie dollar", "fx", "fx"),
    "FXC": ("Canadian dollar", "fx", "fx"),
}

# One-way trading cost assumption in basis points of traded notional.
COST_BPS = {"equity": 3.0, "rates": 2.0, "credit": 4.0, "real": 5.0, "fx": 4.0}

# Classes that earn a long-run risk premium and belong in the risk-parity pod.
RISK_PARITY_CLASSES = ("equity", "rates", "credit", "real")

# Sceptical prior Sharpe ratio for simply holding an asset class (used only to
# centre Monte Carlo drift; position sizing never relies on it).
CLASS_PRIOR_SHARPE = {"equity": 0.30, "rates": 0.20, "credit": 0.25, "real": 0.10, "fx": 0.0}

# ETF -> futures contracts. multiplier = dollars per 1.00 of quoted price.
# `ratio` = default futures notional per $1 of ETF exposure (duration/beta
# adjustment). It is re-estimated from data when futures prices are loaded.
FUTURES = {
    "SPY": {"yahoo": "ES=F", "contracts": {"ES": 50.0, "MES": 5.0}, "ratio": 1.0},
    "QQQ": {"yahoo": "NQ=F", "contracts": {"NQ": 20.0, "MNQ": 2.0}, "ratio": 1.0},
    "IWM": {"yahoo": "RTY=F", "contracts": {"RTY": 50.0, "M2K": 5.0}, "ratio": 1.0},
    "IEI": {"yahoo": "ZF=F", "contracts": {"ZF": 1000.0}, "ratio": 1.05},
    "IEF": {"yahoo": "ZN=F", "contracts": {"ZN": 1000.0}, "ratio": 1.20},
    "TLT": {"yahoo": "ZB=F", "contracts": {"ZB": 1000.0}, "ratio": 1.35},
    "GLD": {"yahoo": "GC=F", "contracts": {"GC": 100.0, "MGC": 10.0}, "ratio": 1.0},
    "SLV": {"yahoo": "SI=F", "contracts": {"SI": 5000.0, "SIL": 1000.0}, "ratio": 1.0},
    "USO": {"yahoo": "CL=F", "contracts": {"CL": 1000.0, "MCL": 100.0}, "ratio": 1.0},
    "FXE": {"yahoo": "6E=F", "contracts": {"6E": 125000.0, "M6E": 12500.0}, "ratio": 1.0},
    "FXY": {"yahoo": "6J=F", "contracts": {"6J": 12500000.0}, "ratio": 1.0},
    "FXB": {"yahoo": "6B=F", "contracts": {"6B": 62500.0}, "ratio": 1.0},
    "FXA": {"yahoo": "6A=F", "contracts": {"6A": 100000.0}, "ratio": 1.0},
    "FXC": {"yahoo": "6C=F", "contracts": {"6C": 100000.0}, "ratio": 1.0},
}


def build_meta(tickers, universe=None):
    """Return {ticker: {name, cls, group, cost_bps}} for the given tickers.

    Unknown tickers (a user's own list) default to class 'equity', group 'custom'.
    """
    universe = universe or DEFAULT_UNIVERSE
    meta = {}
    for t in tickers:
        if t in universe:
            name, cls, group = universe[t]
        else:
            name, cls, group = t, "equity", "custom"
        meta[t] = {"name": name, "cls": cls, "group": group, "cost_bps": COST_BPS.get(cls, 4.0)}
    return meta
