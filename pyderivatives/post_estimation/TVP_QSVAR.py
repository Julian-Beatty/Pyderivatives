import pandas as pd

# gld = pd.read_csv(
#     r"C:\Users\beatt\Spyder directory\State Price Density\Optiondata\GLD_stock.csv"
# )
# spy = pd.read_csv(
#     r"C:\Users\beatt\Spyder directory\State Price Density\Optiondata\SPY_stock.csv"
# )
# btc = pd.read_csv(
#     r"C:\Users\beatt\Spyder directory\State Price Density\BTC_data_new.csv"
# )
# xle = pd.read_csv(
#     r"C:\Users\beatt\Spyder directory\State Price Density\Optiondata\XLE_stock.csv"
# )
import pandas as pd
import numpy as np
def _as_series(x, name: str) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    s = x.copy()
    s.name = name
    return s

def zscore(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean()
    sd = df.std(ddof=0).replace(0, np.nan)
    return (df - mu) / sd

def build_feature_panel(
    moments_by_asset,
    horizon,
    feature,
    assets=("BTC","GLD","SLV","SPY","XLE"),
):
    # --- alias: placeholder names -> stored keys (NO computation) ---
    alias = {
        "phys_var": "phys_vol_ann",
        "rnd_var":  "rnd_vol_ann",
        "prem_var": "prem_vol_ann",
    }

    cols = []
    for a in assets:
        d_h = moments_by_asset[a][horizon]
        key = alias.get(feature, feature)  # resolve alias if needed
        s = _as_series(d_h[key], name=a)   # label node as BTC,GLD,...
        cols.append(s)

    panel = pd.concat(cols, axis=1).dropna()
    return panel

def clean_price_df(df, name):
    """
    Standardize a price DataFrame:
    - Uses ajexdi to adjust prices if present
    - Assumes ajexdi = 1 if missing
    - Returns DataFrame with date index and adjusted price column
    """
    df = df.copy()

    # ensure datetime
    df["date"] = pd.to_datetime(df["date"])

    # handle adjustment factor
    if "ajexdi" in df.columns:
        adj_factor = df["ajexdi"].astype(float)
    else:
        adj_factor = 1.0

    # adjusted price
    df[name] = df["price"].astype(float) / adj_factor

    return (
        df[["date", name]]
        .dropna()
        .sort_values("date")
        .reset_index(drop=True)
    )
# gld_c = clean_price_df(gld, "GLD")
# spy_c = clean_price_df(spy, "SPY")
# xle_c = clean_price_df(xle, "XLE")
# btc_c = clean_price_df(btc, "BTC")   # ajexdi not present → assumed = 1

# prices = (
#     gld_c
#     .merge(spy_c, on="date", how="inner")
#     .merge(xle_c, on="date", how="inner")
#     .merge(btc_c, on="date", how="inner")
#     .set_index("date")
# )

# print(prices.head())

# returns = np.log(prices).diff().dropna()
# returns.describe(percentiles=[0.01, 0.05, 0.95, 0.99])
"""
Quantile Connectedness (QVAR) + GFEVD + Diebold-Yilmaz spillovers
Adapted to 4 assets: BTC, GOLD, SILVER, SP500

Implements the paper's methodology:
- Fit quantile VAR (QVAR) at quantile tau using equation-by-equation Quantile Regression
- Convert to VAR(p) coefficient matrices A_1..A_p
- Compute MA coefficient matrices Psi_h up to horizon H
- Compute generalized FEVD (Pesaran-Shin style)
- Normalize rows to sum to 1
- Compute TO / FROM / NET / TCI
- Rolling-window dynamic spillovers

Dependencies:
  pip install numpy pandas statsmodels matplotlib

Notes:
- Inputs should be daily prices aligned on the same dates.
- Returns are computed as log differences.
"""


import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import statsmodels.api as sm


# ----------------------------
# Utilities: data preparation
# ----------------------------

def to_log_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns from price levels (aligned columns)."""
    px = price_df.astype(float).copy()
    rets = np.log(px).diff()
    return rets.dropna()


def make_var_design(Y: np.ndarray, p: int, add_const: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build VAR(p) regression design:
      y_t = c + A1 y_{t-1} + ... + Ap y_{t-p} + u_t

    Returns:
      X: (T-p, k*p + const)
      Yt: (T-p, k)
    """
    T, k = Y.shape
    if T <= p:
        raise ValueError("Not enough observations for chosen lag p.")
    Yt = Y[p:, :]
    X_parts = []
    for lag in range(1, p + 1):
        X_parts.append(Y[p - lag:T - lag, :])
    X = np.concatenate(X_parts, axis=1)  # (T-p, k*p)
    if add_const:
        X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    return X, Yt


# ----------------------------
# QVAR fit (equation-by-equation)
# ----------------------------

@dataclass
class QVARFit:
    tau: float
    p: int
    A: np.ndarray        # (k, k*p) stacked [A1 ... Ap] row-wise
    c: np.ndarray        # (k,) intercepts
    Sigma_u: np.ndarray  # (k, k) residual covariance for the fitted window


def fit_qvar_quantreg(returns_df: pd.DataFrame, tau: float, p: int) -> QVARFit:
    """
    Fit QVAR(p) at quantile tau via Quantile Regression for each equation.

    We estimate:
      y_{i,t} = c_i + sum_{lag=1..p} (A_l[i,:] y_{t-l}) + u_{i,t}(tau)

    Then build:
      A (k x k*p), c (k,), Sigma_u from residuals (using sample covariance).
    """
    Y = returns_df.values
    X, Yt = make_var_design(Y, p=p, add_const=True)   # X includes constant
    T_eff, k = Yt.shape

    c = np.zeros(k)
    A = np.zeros((k, k * p))
    resid = np.zeros((T_eff, k))

    for i in range(k):
        model = sm.QuantReg(Yt[:, i], X)
        res = model.fit(q=tau)
        params = res.params  # length 1 + k*p

        c[i] = params[0]
        A[i, :] = params[1:]
        resid[:, i] = Yt[:, i] - X @ params

    # Residual covariance (a practical plug-in; paper uses Ω(τ) in FEVD formulas)
    Sigma_u = np.cov(resid.T, bias=False)
    return QVARFit(tau=tau, p=p, A=A, c=c, Sigma_u=Sigma_u)


def unpack_A_mats(A_stacked: np.ndarray, k: int, p: int) -> List[np.ndarray]:
    """Convert stacked A (k x k*p) into list [A1..Ap], each (k x k)."""
    mats = []
    for lag in range(p):
        mats.append(A_stacked[:, lag * k:(lag + 1) * k])
    return mats


# ----------------------------
# VAR to MA(∞) coefficients Psi_h
# ----------------------------

def compute_ma_mats(A_list: List[np.ndarray], H: int) -> List[np.ndarray]:
    """
    Compute MA coefficient matrices Psi_0..Psi_{H-1} for VAR(p):
      y_t = sum_{l=1..p} A_l y_{t-l} + u_t

    Recursion:
      Psi_0 = I
      Psi_h = sum_{l=1..min(p,h)} A_l Psi_{h-l}

    Returns list length H.
    """
    p = len(A_list)
    k = A_list[0].shape[0]
    Psi = [np.eye(k)]
    for h in range(1, H):
        acc = np.zeros((k, k))
        for l in range(1, min(p, h) + 1):
            acc += A_list[l - 1] @ Psi[h - l]
        Psi.append(acc)
    return Psi


# ----------------------------
# Generalized FEVD (Pesaran-Shin style)
# ----------------------------

def gfevd(Psi: List[np.ndarray], Sigma_u: np.ndarray, H: int) -> np.ndarray:
    """
    Generalized FEVD Θ^g_{ij}(H) (non-normalized), consistent with Pesaran-Shin.

    A commonly used form in connectedness literature:

      theta_ij = (1 / sigma_jj) * sum_{h=0..H-1} (e_i' Psi_h Sigma e_j)^2
                / sum_{h=0..H-1} (e_i' Psi_h Sigma Psi_h' e_i)

    Where sigma_jj is the j-th diagonal element of Sigma.

    Returns:
      Theta: (k,k) matrix of FEVD contributions (rows i, cols j)
    """
    Sigma = Sigma_u
    k = Sigma.shape[0]
    diag = np.diag(Sigma)
    if np.any(diag <= 0):
        # numerical safeguard
        diag = np.clip(diag, 1e-12, None)

    num = np.zeros((k, k))
    den = np.zeros(k)

    for h in range(H):
        Ph = Psi[h]
        # Denominator parts: e_i' Ph Sigma Ph' e_i = (Ph Sigma Ph')_ii
        M = Ph @ Sigma @ Ph.T
        den += np.diag(M)

        # Numerator parts: e_i' Ph Sigma e_j = (Ph Sigma)_ij
        B = Ph @ Sigma
        num += (B ** 2) / diag.reshape(1, -1)

    Theta = num / den.reshape(-1, 1)
    return Theta


def normalize_fevd_rows(Theta: np.ndarray) -> np.ndarray:
    """Row-normalize FEVD so each row sums to 1."""
    row_sums = Theta.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return Theta / row_sums


# ----------------------------
# Spillover measures: TO / FROM / NET / TCI
# ----------------------------

@dataclass
class SpilloverStats:
    Theta: np.ndarray     # normalized FEVD (k,k)
    TO: np.ndarray        # (k,)
    FROM: np.ndarray      # (k,)
    NET: np.ndarray       # (k,)
    TCI: float            # scalar


def spillover_measures(Theta_norm: np.ndarray) -> SpilloverStats:
    """
    Diebold-Yilmaz style measures using normalized FEVD.

    For k variables:
      FROM_i = sum_{j != i} Theta_{ij}
      TO_j   = sum_{i != j} Theta_{ij}   (column j off-diagonal, interpreted as j -> i)
      NET_j  = TO_j - FROM_j
      TCI    = (sum_{i != j} Theta_{ij}) / k * 100
    """
    k = Theta_norm.shape[0]
    offdiag_sum = Theta_norm.sum() - np.trace(Theta_norm)

    FROM = Theta_norm.sum(axis=1) - np.diag(Theta_norm)
    TO = Theta_norm.sum(axis=0) - np.diag(Theta_norm)
    NET = TO - FROM
    TCI = (offdiag_sum / k) * 100.0

    return SpilloverStats(Theta=Theta_norm, TO=TO, FROM=FROM, NET=NET, TCI=TCI)


# ----------------------------
# One-shot connectedness for a given tau
# ----------------------------

def quantile_connectedness(
    returns_df: pd.DataFrame,
    tau: float,
    p: int = 1,
    H: int = 10
) -> SpilloverStats:
    """
    Full pipeline for one sample window:
      QVAR fit -> MA mats -> GFEVD -> normalize -> spillover measures
    """
    qfit = fit_qvar_quantreg(returns_df, tau=tau, p=p)
    k = returns_df.shape[1]
    A_list = unpack_A_mats(qfit.A, k=k, p=p)
    Psi = compute_ma_mats(A_list, H=H)
    Theta = gfevd(Psi, qfit.Sigma_u, H=H)
    Theta_n = normalize_fevd_rows(Theta)
    return spillover_measures(Theta_n)


# ----------------------------
# Rolling window dynamic connectedness
# ----------------------------

@dataclass
class RollingResult:
    tci: pd.DataFrame          # columns = taus
    net: Dict[float, pd.DataFrame]   # tau -> (index dates, columns assets)
    to_: Dict[float, pd.DataFrame]
    from_: Dict[float, pd.DataFrame]


def rolling_quantile_connectedness(
    returns_df: pd.DataFrame,
    taus: List[float] = [0.05, 0.50, 0.95],
    p: int = 1,
    H: int = 10,
    window: int = 250,
    step: int = 1
) -> RollingResult:
    """
    Compute dynamic (rolling) spillover indices.
    - window: number of observations in each rolling estimation window (e.g., 250 ~ 1 trading year)
    - step: move window by this many observations
    """
    assets = list(returns_df.columns)
    dates = returns_df.index

    tci_out = {tau: [] for tau in taus}
    net_out = {tau: [] for tau in taus}
    to_out  = {tau: [] for tau in taus}
    from_out= {tau: [] for tau in taus}
    out_dates = []

    n = len(returns_df)
    for end in range(window, n + 1, step):
        start = end - window
        wdf = returns_df.iloc[start:end]
        out_dates.append(dates[end - 1])

        for tau in taus:
            stats = quantile_connectedness(wdf, tau=tau, p=p, H=H)
            tci_out[tau].append(stats.TCI)
            net_out[tau].append(stats.NET)
            to_out[tau].append(stats.TO)
            from_out[tau].append(stats.FROM)

    idx = pd.DatetimeIndex(out_dates)
    tci_df = pd.DataFrame({tau: pd.Series(tci_out[tau], index=idx) for tau in taus})

    net_df = {tau: pd.DataFrame(np.vstack(net_out[tau]), index=idx, columns=assets) for tau in taus}
    to_df  = {tau: pd.DataFrame(np.vstack(to_out[tau]),  index=idx, columns=assets) for tau in taus}
    from_df= {tau: pd.DataFrame(np.vstack(from_out[tau]),index=idx, columns=assets) for tau in taus}

    return RollingResult(tci=tci_df, net=net_df, to_=to_df, from_=from_df)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _diamond_positions(labels):
    """
    A fixed diamond layout that works nicely for 4 assets.
    If you have exactly 4 labels, place them in a diamond.
    Otherwise fall back to a circle.
    """
    n = len(labels)
    if n == 4:
        # Top, Right, Bottom, Left (you can reorder labels before calling if desired)
        return {
            labels[0]: (0.0,  1.0),
            labels[1]: (1.0,  0.0),
            labels[2]: (0.0, -1.0),
            labels[3]: (-1.0, 0.0),
        }
    # fallback circle
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return {lab: (np.cos(a), np.sin(a)) for lab, a in zip(labels, angles)}

def _draw_arrow(ax, x0, y0, x1, y1, lw, color, alpha=0.9):
    ax.annotate(
        "",
        xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="-|>",
            lw=lw,
            color=color,
            alpha=alpha,
            shrinkA=24, shrinkB=24,     # keep arrows off the node circles
            mutation_scale=14
        ),
        zorder=2
    )

def plot_spillover_network_on_ax(
    ax,
    Theta_norm: pd.DataFrame,
    net: pd.Series | None = None,
    *,
    title: str = "",
    top_k: int = 10,
    min_weight: float | None = None,
    node_size: int = 2600,
    node_edge_lw: float = 4.0,
    node_edge_color_pos: str = "green",
    node_edge_color_neg: str = "black",
):
    """
    Draw a directed spillover network on a provided matplotlib axis.

    Theta_norm: DataFrame (rows = affected i, cols = source j). Edge j -> i weight = Theta[i,j].
    net: Series for node border color by sign (NET>0 transmitter).
    """
    assets = list(Theta_norm.columns)
    if list(Theta_norm.index) != assets:
        Theta_norm = Theta_norm.loc[assets, assets]

    # --- choose layout: force a diamond for 4 nodes ---
    pos = _diamond_positions(assets)

    # --- build edge list from off-diagonals ---
    edges = []
    for i in assets:
        for j in assets:
            if i == j:
                continue
            w = float(Theta_norm.loc[i, j])
            edges.append((j, i, w))  # sender j -> receiver i

    # keep strongest edges
    edges.sort(key=lambda t: t[2], reverse=True)
    if top_k is not None:
        edges = edges[:top_k]
    if min_weight is not None:
        edges = [e for e in edges if e[2] >= min_weight]

    # scale linewidths
    weights = np.array([w for _, _, w in edges], dtype=float)
    if len(weights) == 0:
        weights = np.array([1.0])
    w_min, w_max = weights.min(), weights.max()

    def lw_scale(w):
        if w_max == w_min:
            return 2.5
        return 1.0 + 7.0 * (w - w_min) / (w_max - w_min)

    # edge colors: color by sender NET sign (if provided)
    def edge_color(sender):
        if net is None:
            return "black"
        return "red" if net.loc[sender] > 0 else "blue"

    # --- draw nodes ---
    for a in assets:
        x, y = pos[a]
        if net is None:
            edgecol = "black"
        else:
            edgecol = node_edge_color_pos if net.loc[a] > 0 else node_edge_color_neg

        ax.scatter(
            [x], [y],
            s=node_size,
            facecolor="white",
            edgecolor=edgecol,
            linewidth=node_edge_lw,
            zorder=3
        )
        ax.text(x, y, a, ha="center", va="center", fontsize=13, zorder=4)

    # --- draw edges ---
    for sender, recv, w in edges:
        x0, y0 = pos[sender]
        x1, y1 = pos[recv]
        _draw_arrow(ax, x0, y0, x1, y1, lw=lw_scale(w), color=edge_color(sender), alpha=0.85)

    ax.set_title(title, fontsize=16)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_network_panels_from_returns(
    returns: pd.DataFrame,
    *,
    p: int = 1,
    H: int = 10,
    taus=(0.50, 0.05, 0.95),
    top_k: int = 10,
    min_weight: float | None = None,
):
    """
    Correct panel plotting: draws each network onto a subplot axis.
    Requires quantile_connectedness() from your connectedness code.
    """
    fig, axes = plt.subplots(1, len(taus), figsize=(18, 6))

    for ax, tau in zip(axes, taus):
        st = quantile_connectedness(returns, tau=tau, p=p, H=H)
        Theta = pd.DataFrame(st.Theta, index=returns.columns, columns=returns.columns)
        net = pd.Series(st.NET, index=returns.columns)

        plot_spillover_network_on_ax(
            ax,
            Theta,
            net=net,
            title=f"τ = {tau}",
            top_k=top_k,
            min_weight=min_weight
        )

    plt.tight_layout()
    return fig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

def connectedness_table(stats, asset_names):
    Theta = pd.DataFrame(
        stats.Theta * 100,
        index=asset_names,
        columns=asset_names
    )

    from_others = Theta.sum(axis=1) - np.diag(Theta)
    to_others   = Theta.sum(axis=0) - np.diag(Theta)
    net         = to_others - from_others

    table = Theta.copy()
    table["From others"] = from_others

    summary = pd.DataFrame(
        [to_others, to_others + np.diag(Theta), net],
        index=["TO", "Inc.Own", "Net"],
        columns=asset_names
    )

    return pd.concat([table, summary])



def plot_spillover_network_arc_on_ax(
    ax,
    Theta: pd.DataFrame,
    net: pd.Series | None = None,
    *,
    title: str = "",
    top_k: int = 12,
    min_weight: float | None = None,
    node_size: int = 900,
    edge_alpha: float = 0.65,
    curvature: float = 0.25,
):
    """
    Arc-style directed spillover network drawn on an existing axis.

    Theta: DataFrame (rows=affected i, cols=source j). Edge j->i weight = Theta[i,j].
    net: Series indexed by node names; node color by NET sign (gold transmitter, blue receiver).
    """
    assets = list(Theta.columns)
    Theta = Theta.loc[assets, assets]

    # Build edge list (off-diagonal)
    edges = []
    for i in assets:
        for j in assets:
            if i == j:
                continue
            w = float(Theta.loc[i, j])
            edges.append((j, i, w))  # sender j -> receiver i

    edges.sort(key=lambda t: t[2], reverse=True)
    if top_k is not None:
        edges = edges[:top_k]
    if min_weight is not None:
        edges = [e for e in edges if e[2] >= min_weight]

    # Graph
    G = nx.DiGraph()
    G.add_nodes_from(assets)
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    pos = nx.circular_layout(G)

    # Node colors
    if net is None:
        node_colors = ["lightgray" for _ in G.nodes()]
    else:
        net = net.reindex(assets)
        node_colors = ["#d4af37" if net[n] > 0 else "#2b6cb0" for n in G.nodes()]  # gold vs blue

    # Edge widths
    wts = np.array([G[u][v]["weight"] for u, v in G.edges()], dtype=float)
    if len(wts) == 0:
        wts = np.array([1.0])
    w_min, w_max = wts.min(), wts.max()

    def scale_width(w):
        if w_max == w_min:
            return 2.0
        return 0.8 + 6.0 * (w - w_min) / (w_max - w_min)

    edge_widths = [scale_width(G[u][v]["weight"]) for u, v in G.edges()]

    # Draw on provided ax
    ax.set_title(title, fontsize=16)
    ax.axis("off")

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_size,
        node_color=node_colors,
        edgecolors="black",
        linewidths=1.5
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=11,
        font_color="black"
    )

    # Curved directed edges
    for (u, v), lw in zip(G.edges(), edge_widths):
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=[(u, v)],
            width=lw,
            alpha=edge_alpha,
            edge_color="gray",
            arrows=True,
            arrowsize=18,
            connectionstyle=f"arc3,rad={curvature}",
            min_source_margin=10,
            min_target_margin=12,
        )

import numpy as np
import pandas as pd
from typing import Dict, Iterable, List, Optional, Tuple, Union

def _connectedness_panel_df(stats, names: List[str], decimals: int = 2) -> pd.DataFrame:
    """
    Build one panel DataFrame like the paper:
      - N x N Theta (in %)
      - 'From others' column
      - Rows: TO, Inc.Own, Net (with From others blank)
    """
    Theta_raw = np.asarray(stats.Theta, dtype=float) * 100.0
    n = Theta_raw.shape[0]
    if len(names) != n:
        raise ValueError(f"names length {len(names)} != Theta dim {n}. Pass the same columns used in estimation.")

    Theta = pd.DataFrame(Theta_raw, index=names, columns=names)

    diag = np.diag(Theta.values)
    from_others = Theta.sum(axis=1) - diag
    to_others   = Theta.sum(axis=0) - diag
    inc_own     = to_others + diag
    net         = to_others - from_others

    panel = Theta.copy()
    panel["From others"] = from_others

    summary = pd.DataFrame(
        [to_others, inc_own, net],
        index=["TO", "Inc.Own", "Net"],
        columns=names
    )
    summary["From others"] = np.nan

    out = pd.concat([panel, summary], axis=0)

    # Round for display
    out = out.round(decimals)
    return out


def connectedness_table_latex(
    stats_by_tau: Dict[float, object],
    names: Iterable[str],
    *,
    taus_order: Tuple[float, ...] = (0.50, 0.05, 0.95),
    panel_labels: Optional[Dict[float, str]] = None,
    caption: str = "Static return connectedness.",
    label: str = "tab:connectedness",
    note: str = "Source: Authors' estimations.",
    decimals: int = 2,
    table_env: bool = True,
    fontsize: str = "\\small",
    colsep: str = "6pt",
) -> str:
    """
    Produce LaTeX resembling the paper's Table 2a with Panels A/B/C.

    Parameters
    ----------
    stats_by_tau : dict[tau -> stats]
        Each stats must have .Theta (NxN) and .TCI (scalar, in percent terms consistent with Theta*100).
    names : iterable[str]
        Variable names (must match dimension/order used to estimate stats).
    taus_order : tuple
        Order of panels.
    panel_labels : dict
        Optional mapping tau -> panel title suffix, e.g.
            {0.50: "Quantile (median $\\tau=0.50$)", 0.05: "Quantile (extreme lower quantile $\\tau=0.05$)", ...}
    caption/label/note : strings
        LaTeX metadata.
    decimals : int
        Rounding for numbers.
    table_env : bool
        Wrap in \\begin{table}...\\end{table}.
    fontsize : str
        e.g. "\\small" or "\\footnotesize".
    colsep : str
        sets \\tabcolsep for spacing.

    Returns
    -------
    latex : str
        A LaTeX table string using booktabs.
    """
    names = list(names)
    n = len(names)

    if panel_labels is None:
        panel_labels = {
            0.50: r"Quantile (median $\tau = 0.50$)",
            0.05: r"Quantile (extreme lower quantile $\tau = 0.05$)",
            0.95: r"Quantile (extreme upper quantile $\tau = 0.95$)",
        }

    # Column spec: first column = row labels, then N columns + From others
    colspec = "l" + "r" * (n + 1)

    def fmt(x: Union[float, int, None]) -> str:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        return f"{x:.{decimals}f}"

    # Build LaTeX lines
    lines: List[str] = []
    if table_env:
        lines += [
            r"\begin{table}[!htbp]",
            r"\centering",
        ]

    lines += [
        fontsize,
        rf"\setlength{{\tabcolsep}}{{{colsep}}}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
    ]

    # Header row
    header = [""] + names + ["From others"]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    # Panels
    panel_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for k, tau in enumerate(taus_order):
        if tau not in stats_by_tau:
            raise KeyError(f"tau={tau} not found in stats_by_tau. Provided keys: {list(stats_by_tau.keys())}")

        stats = stats_by_tau[tau]
        panel_df = _connectedness_panel_df(stats, names, decimals=decimals)

        # Panel title line (spans all columns)
        panel_title = panel_labels.get(tau, rf"Quantile ($\tau={tau}$)")
        span = n + 2  # row-label + N vars + From others
        lines.append(rf"\multicolumn{{{span}}}{{l}}{{Panel {panel_letters[k]}: {panel_title}}} \\")
        # No extra midrule here (paper style is usually just spacing)

        # Main N rows
        for row in names:
            vals = [row] + [fmt(panel_df.loc[row, c]) for c in names] + [fmt(panel_df.loc[row, "From others"])]
            lines.append(" & ".join(vals) + r" \\")

        # Summary rows (TO, Inc.Own, Net)
        for row in ["TO", "Inc.Own", "Net"]:
            vals = [row] + [fmt(panel_df.loc[row, c]) for c in names] + [""]  # blank From others
            # Put TCI in the last cell of Inc.Own row? The paper places "TCI" label there,
            # and the numeric TCI in the last cell of Net row. We'll mimic that.
            if row == "Inc.Own":
                vals[-1] = r"\textit{TCI}"
            if row == "Net":
                # last cell shows TCI number
                tci_val = getattr(stats, "TCI", None)
                # If your stats.TCI is in [0,1], uncomment next line:
                # tci_val = (100.0 * tci_val) if (tci_val is not None and tci_val <= 1.0) else tci_val
                vals[-1] = fmt(tci_val)
            lines.append(" & ".join(vals) + r" \\")
        lines.append(r"\addlinespace")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\vspace{{0.5ex}}\\{{\footnotesize {note}}}",
    ]
    if table_env:
        lines.append(r"\end{table}")

    return "\n".join(lines)
