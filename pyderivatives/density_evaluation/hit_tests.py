import numpy as np
from scipy import stats
from scipy.special import expit, logit
from scipy.optimize import minimize


def hit_test(pit, lower=0.0, upper=0.05):
    pit = np.asarray(pit, dtype=float)
    pit = pit[np.isfinite(pit)]

    n = len(pit)
    if n == 0:
        return None

    expected_prob = float(upper - lower)
    hits = ((pit >= lower) & (pit <= upper)).astype(int)
    observed_prob = float(hits.mean())

    pvalue = stats.binomtest(
        int(hits.sum()),
        n,
        expected_prob,
        alternative="two-sided",
    ).pvalue

    denom = np.sqrt(expected_prob * (1.0 - expected_prob) / n)
    z_stat = (observed_prob - expected_prob) / denom if denom > 0 else np.nan

    return {
        "test": "Unconditional hit test",
        "lower": float(lower),
        "upper": float(upper),
        "expected_prob": expected_prob,
        "observed_prob": observed_prob,
        "n": int(n),
        "hits": int(hits.sum()),
        "z_stat": float(z_stat),
        "pvalue": float(pvalue),
    }


def standard_hit_tests(pit):
    return {
        "lower_1pct": hit_test(pit, 0.00, 0.01),
        "lower_5pct": hit_test(pit, 0.00, 0.05),
        "lower_10pct": hit_test(pit, 0.00, 0.10),
        "middle_80pct": hit_test(pit, 0.10, 0.90),
        "upper_10pct": hit_test(pit, 0.90, 1.00),
        "upper_5pct": hit_test(pit, 0.95, 1.00),
        "upper_1pct": hit_test(pit, 0.99, 1.00),
    }


def _patton_hit_test_from_hits(hits, expected_prob, eps=1e-10):
    hits = np.asarray(hits, dtype=float)
    hits = hits[np.isfinite(hits)]

    n_raw = len(hits)

    if n_raw <= 10:
        return {
            "test": "Patton hit regression",
            "expected_prob": float(expected_prob),
            "n_raw": int(n_raw),
            "n": 0,
            "statistic": np.nan,
            "pvalue": np.nan,
            "df": 4,
            "message": "Too few observations.",
        }

    p = float(expected_prob)
    if not (0.0 < p < 1.0):
        raise ValueError("expected_prob must be strictly between 0 and 1.")

    rows = []
    y = []

    for t in range(10, n_raw):
        lag1 = hits[t - 1]
        lag5 = hits[t - 5:t].sum()
        lag10 = hits[t - 10:t].sum()

        rows.append([1.0, lag1, lag5, lag10])
        y.append(hits[t])

    Z = np.asarray(rows, dtype=float)
    y = np.asarray(y, dtype=float)

    offset = logit(np.clip(p, eps, 1.0 - eps))

    def loglike(beta):
        eta = offset + Z @ beta
        prob = expit(eta)
        prob = np.clip(prob, eps, 1.0 - eps)

        ll = y * np.log(prob) + (1.0 - y) * np.log(1.0 - prob)
        return float(np.sum(ll))

    def nll(beta):
        return -loglike(beta)

    beta0 = np.zeros(Z.shape[1], dtype=float)
    res = minimize(nll, beta0, method="BFGS")

    ll1 = -float(res.fun)

    prob0 = np.clip(p, eps, 1.0 - eps)
    ll0 = float(np.sum(y * np.log(prob0) + (1.0 - y) * np.log(1.0 - prob0)))

    lr = -2.0 * (ll0 - ll1)
    pvalue = 1.0 - stats.chi2.cdf(lr, df=4)

    return {
        "test": "Patton hit regression",
        "expected_prob": float(p),
        "observed_prob": float(np.mean(hits)),
        "observed_prob_reg_sample": float(np.mean(y)),
        "n_raw": int(n_raw),
        "n": int(len(y)),
        "hits": int(hits.sum()),
        "hits_reg_sample": int(y.sum()),
        "statistic": float(lr),
        "pvalue": float(pvalue),
        "df": 4,
        "beta": res.x.tolist(),
        "success": bool(res.success),
        "message": str(res.message),
    }


def patton_hit_test(pit, lower=0.0, upper=0.05, eps=1e-10):
    pit = np.asarray(pit, dtype=float)
    pit = pit[np.isfinite(pit)]

    expected_prob = float(upper - lower)

    if not (0.0 < expected_prob < 1.0):
        raise ValueError("Region probability upper-lower must be between 0 and 1.")

    hits = ((pit >= lower) & (pit <= upper)).astype(float)

    out = _patton_hit_test_from_hits(hits, expected_prob, eps=eps)

    out.update({
        "lower": float(lower),
        "upper": float(upper),
    })

    return out


def patton_union_hit_test(pit, intervals, eps=1e-10):
    pit = np.asarray(pit, dtype=float)
    pit = pit[np.isfinite(pit)]

    hits = np.zeros(len(pit), dtype=bool)
    expected_prob = 0.0

    for lower, upper in intervals:
        lower = float(lower)
        upper = float(upper)

        if not (0.0 <= lower < upper <= 1.0):
            raise ValueError("Each interval must satisfy 0 <= lower < upper <= 1.")

        expected_prob += upper - lower
        hits |= (pit >= lower) & (pit <= upper)

    out = _patton_hit_test_from_hits(
        hits.astype(float),
        expected_prob,
        eps=eps,
    )

    out.update({
        "intervals": [(float(a), float(b)) for a, b in intervals],
    })

    return out


def standard_patton_hit_tests(pit):
    return {
        "lower_1pct": patton_hit_test(pit, 0.00, 0.01),
        "lower_5pct": patton_hit_test(pit, 0.00, 0.05),
        "lower_10pct": patton_hit_test(pit, 0.00, 0.10),
        "middle_80pct": patton_hit_test(pit, 0.10, 0.90),
        "upper_10pct": patton_hit_test(pit, 0.90, 1.00),
        "upper_5pct": patton_hit_test(pit, 0.95, 1.00),
        "upper_1pct": patton_hit_test(pit, 0.99, 1.00),
    }


def paper_hit_test_table(pit):
    return {
        "Panel A. Equal Density Mass": {
            "(0,0.2)": patton_hit_test(pit, 0.00, 0.20),
            "(0.2,0.4)": patton_hit_test(pit, 0.20, 0.40),
            "(0.4,0.6)": patton_hit_test(pit, 0.40, 0.60),
            "(0.6,0.8)": patton_hit_test(pit, 0.60, 0.80),
            "(0.8,1)": patton_hit_test(pit, 0.80, 1.00),
        },
        "Panel B. Unequal Density Mass": {
            "(0,0.15)": patton_hit_test(pit, 0.00, 0.15),
            "(0.15,0.35)": patton_hit_test(pit, 0.15, 0.35),
            "(0.35,0.65)": patton_hit_test(pit, 0.35, 0.65),
            "(0.65,0.85)": patton_hit_test(pit, 0.65, 0.85),
            "(0.85,1)": patton_hit_test(pit, 0.85, 1.00),
        },
        "Panel C. Tails and Center": {
            "(0,0.25)": patton_hit_test(pit, 0.00, 0.25),
            "(0.25,0.75)": patton_hit_test(pit, 0.25, 0.75),
            "(0.75,1)": patton_hit_test(pit, 0.75, 1.00),
            "(0,0.25)U(0.75,1)": patton_union_hit_test(
                pit,
                intervals=[(0.00, 0.25), (0.75, 1.00)],
            ),
        },
    }


def paper_hit_test_pvalue_table(pit):
    full = paper_hit_test_table(pit)

    out = {}

    for panel, regions in full.items():
        out[panel] = {
            region: result["pvalue"]
            for region, result in regions.items()
        }

    return out