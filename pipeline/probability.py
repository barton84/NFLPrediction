"""Cover and over/under probabilities with NFL key numbers.

Final margins pile up on 3, 7, 6, 10 and 14. We take a smooth bell curve and
reweight each whole-number margin by how often it actually happened
(2016-2025), which gives a distribution with the real key-number bumps.

The center of that distribution starts where the market says the game is a
coin flip (the line), then moves toward the model by K times the model's
edge. K is fitted on past seasons. If the model's edge historically carried
no information about who covers, K comes out near zero and every game reads
close to 50%. That is the honest answer, not a bug.

Information the market has not seen (your own manual injury entries) moves
the center at full weight, since the posted line cannot reflect it.
"""
import numpy as np

KS = np.arange(-70, 71)
SPREAD_SD = 13.3
TOTAL_SD = 13.4


def key_weights(results, sd=SPREAD_SD):
    """How much more (or less) often each final margin occurs than a smooth curve predicts."""
    res = np.asarray(results, dtype=float)
    emp = np.array([(np.abs(res) == abs(k)).mean() for k in KS])
    emp = emp / emp.sum()
    sm = np.exp(-KS ** 2 / (2 * sd ** 2)); sm /= sm.sum()
    w = np.where(KS == 0, 0.02, emp / np.maximum(sm, 1e-12))   # ties are rare (overtime)
    return np.clip(w, 0, 6)


def pmf(center, kw, sd=SPREAD_SD):
    p = np.exp(-(KS - center) ** 2 / (2 * sd ** 2)) * kw
    return p / p.sum()


def home_cover(line, center, kw, sd=SPREAD_SD):
    """line = home margin needed (spread_line convention: positive = home favored).
    Returns (P home covers, P push)."""
    p = pmf(center, kw, sd)
    return float(p[KS > line + 1e-9].sum()), float(p[np.abs(KS - line) < 1e-9].sum())


def market_center(line, kw, sd=SPREAD_SD):
    """Center where home and away cover equally often at the posted line."""
    lo, hi = line - 8, line + 8
    for _ in range(40):
        m = (lo + hi) / 2
        a, pp = home_cover(line, m, kw, sd)
        if a < 1 - a - pp:
            lo = m
        else:
            hi = m
    return (lo + hi) / 2


# ---- totals: plain discrete bell curve on whole points (key numbers are weak for totals)
TK = np.arange(0, 121)


def over_prob(line, center, sd=TOTAL_SD):
    p = np.exp(-(TK - center) ** 2 / (2 * sd ** 2)); p /= p.sum()
    return float(p[TK > line + 1e-9].sum()), float(p[np.abs(TK - line) < 1e-9].sum())


def fit_k(edges, lines, results, kw, grid=np.round(np.arange(0, 0.61, 0.02), 2), total=False):
    """Pick K that maximizes the likelihood of the actual cover outcomes."""
    best, bestll = 0.0, -1e18
    centers0 = [(l if total else market_center(l, kw)) for l in lines]
    for K in grid:
        ll = 0.0
        for e, l, r, c0 in zip(edges, lines, results, centers0):
            if total:
                a, pp = over_prob(l, c0 + K * e)
            else:
                a, pp = home_cover(l, c0 + K * e, kw)
            b = 1 - a - pp
            if r > l:
                ll += np.log(max(a, 1e-9))
            elif r < l:
                ll += np.log(max(b, 1e-9))
        if ll > bestll:
            best, bestll = float(K), ll
    return best
