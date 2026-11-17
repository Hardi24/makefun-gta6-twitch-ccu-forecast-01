"""gta6-twitch-ccu-forecast-01 entry: forecast GTA VI's launch-day Twitch
viewers, hours 2..24, as five quantiles per hour.

Model (v7, tuned on the 4 practice launches; mean pinball loss 0.1374 on
log scale, house baseline 0.3871):
  - Per-hour weighted least squares of ln(viewers_h) on ln(hour-1 viewers)
    and ln(pre-launch peak), over clean comparables only (Tier: comparable,
    excluding simultaneous-global unlocks), platform-weighted (PC games 0.35,
    console-only 1.0 — GTA VI is PS5 + Xbox Series, console-only).
  - Slopes shrunk 0.8 toward their structural targets (ln v1 -> 1, ln
    pre_peak -> 0), intercept recentered at the pool centroid.
  - Hours 5..9 (the US-unlock jump region) are fitted on unlock-aligned
    index a = h - u, where u is the US-East midnight unlock hour: 5 in
    daylight time (04:00 UTC) and 6 in standard time (05:00 UTC). GTA VI
    releases 19 Nov 2026, after DST ends, so u = 6.
  - Quantiles come from the empirical residuals of the fitted regression,
    narrowed by a 0.35x spread multiplier (calibrated on practice; the
    in-sample residual spread is far wider than the conditional spread for
    a launch of this size).
  - Size-effect saturation: the (beta_v1 - 1) term is frozen at the largest
    comparable's ln(v1), so a GTA VI hour-1 print above every past launch
    forecasts like the biggest launches instead of extrapolating the
    size trend to absurdly low evening ratios.

Standard library only, no network, deterministic. Reads every <id>.in in
the input directory and writes <id>.out to the output directory:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]
SIGMA = 0.15  # fallback random-walk spread (log units per sqrt hour)

SHRINK = 0.8        # slope shrinkage toward structural targets
SPREAD_MULT = 0.35  # residual-quantile narrowing, calibrated on practice
FEATS = ("v1", "pre_peak")

PHASE_OF = {}
for _ph, _hs in {"early": [2, 3, 4], "usjump": [5, 6, 7], "uswest": [8, 9],
                 "midday": [10, 11, 12, 13], "rise": [14, 15, 16],
                 "evening": [17, 18, 19, 20, 21], "late": [22, 23, 24]}.items():
    for _h in _hs:
        PHASE_OF[_h] = _ph


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def round2(x):
    """x rounded to 2 decimals, half up (like JavaScript Math.round)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


# ---------------------------------------------------------------- data prep

def release_hours(comp):
    """hour (1..24) -> (viewers, channels) for a comparable's release day."""
    base = parse_ts(comp["day_start"])
    out = {}
    for p in comp["hourly"]:
        t = parse_ts(p["from"])
        dh = (t - base).total_seconds() / 3600.0
        if dh == int(dh) and 0 <= dh < 24:
            out[int(dh) + 1] = (p["viewers"], p.get("channels"))
    return out


def pre_features(hourly, base):
    """Pre-release-day features from an hourly series."""
    curve = {}
    for p in hourly:
        t = parse_ts(p["from"])
        dh = (t - base).total_seconds() / 3600.0
        if dh == int(dh) and -24 <= dh < 0 and p.get("viewers"):
            curve[int(dh)] = p["viewers"]
    vals = list(curve.values())
    return {
        "pre_curve": curve,
        "pre_peak": max(vals) if vals else 0,
        "pre_last": curve.get(-1, 0),
        "pre24": sum(vals) / len(vals) if vals else 0,
    }


def comp_features(comp):
    curve = release_hours(comp)
    v1 = curve.get(1, (0, None))[0] if 1 in curve else 0
    c1 = curve.get(1, (0, None))[1] if 1 in curve else None
    g = {"name": comp["name"], "v1": v1, "c1": c1 or 0,
         "curve": {h: v for h, (v, _) in curve.items()},
         "chcurve": {h: c for h, (_, c) in curve.items() if c}}
    g.update(pre_features(comp["hourly"], parse_ts(comp["day_start"])))
    g["tier"] = tier_of(comp)
    g["platforms"] = comp.get("platforms")
    g["unlock"] = comp.get("unlock")
    g["_day_start"] = comp["day_start"]
    return g


def target_hist(case):
    h = pre_features(case.get("history_hourly") or [], parse_ts(case["day_start"]))
    h["_day_start"] = case["day_start"]
    return h


# ---------------------------------------------------------------- filtering

def tier_of(comp):
    m = re.search(r"Tier:\s*([a-z\-]+)", comp.get("unlock") or "")
    return m.group(1) if m else "unknown"


def unlock_kind(comp):
    u = (comp.get("unlock") or "").lower()
    if "simultaneous" in u or "single global" in u:
        return "simul"
    if "rolling" in u or "midnight local" in u:
        return "rolling"
    return "unknown"


def plat_weight(comp):
    """Similarity to GTA VI (PS5 + Xbox Series, console-only, rolling midnight)."""
    p = (comp.get("platforms") or "").lower()
    if "pc" in p:
        return 0.35
    return 1.0


def filter_pool(comps):
    """Clean comparables only, excluding simultaneous-global unlocks."""
    out = []
    for g in comps:
        if g.get("tier") != "comparable" or g["v1"] <= 0:
            continue
        if unlock_kind(g) == "simul":
            continue
        out.append(g)
    return out


# ---------------------------------------------------------------- DST / unlock

def us_eastern_dst(dt):
    """True if US Eastern DST is in effect on dt's date (post-2007 rules)."""
    def nth_sunday(y, m, n):
        d = date(y, m, 1)
        off = (6 - d.weekday()) % 7
        return d + timedelta(days=off + 7 * (n - 1))
    start = nth_sunday(dt.year, 3, 2)
    end = nth_sunday(dt.year, 11, 1)
    return start <= dt.date() < end


def unlock_hour(day_start):
    """1-based hour whose start is the US-East midnight unlock
    (DST: 04:00Z -> h5; EST: 05:00Z -> h6)."""
    return 5 if us_eastern_dst(parse_ts(day_start)) else 6


# ---------------------------------------------------------------- numerics

def _quantile(sv, q):
    n = len(sv)
    if n == 0:
        return 0.0
    if n == 1:
        return sv[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    return sv[lo] * (1 - (pos - lo)) + sv[hi] * (pos - lo)


def _solve(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            return None
        M[col], M[piv] = M[piv], M[col]
        for r in range(n):
            if r != col:
                f = M[r][col] / M[col][col]
                for cc in range(col, n + 1):
                    M[r][cc] -= f * M[col][cc]
    return [M[i][n] / M[i][i] for i in range(n)]


def _solve_wls(X, Y, W):
    n = len(X[0])
    A = [[sum(W[k] * X[k][i] * X[k][j] for k in range(len(X))) for j in range(n)]
         for i in range(n)]
    b = [sum(W[k] * X[k][i] * Y[k] for k in range(len(X))) for i in range(n)]
    return _solve(A, b)


def _fit_predict(rows, tx, shrink, feats):
    """Weighted OLS with shrunk slopes and recentered intercept.
    Returns (median log viewers, sorted residuals, beta) or None."""
    beta = _solve_wls([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows])
    if beta is None:
        return None
    beta = list(beta)
    for j, f in enumerate(feats, start=1):
        tgt = 1.0 if f == "v1" else 0.0
        beta[j] = tgt + shrink * (beta[j] - tgt)
    mx = [sum(r[0][j] for r in rows) / len(rows) for j in range(len(beta))]
    my = sum(r[1] for r in rows) / len(rows)
    beta[0] = my - sum(beta[j] * mx[j] for j in range(1, len(beta)))
    med = sum(b * x for b, x in zip(beta, tx))
    res = sorted(y - sum(b * x for b, x in zip(beta, xx)) for xx, y, _ in rows)
    return med, res, beta


def _fmap(v1, c1, pre_peak, pre_last):
    return {"v1": math.log(max(1, v1)),
            "pre_peak": math.log(max(1, pre_peak)),
            "pre_last": math.log(max(1, pre_last)),
            "vpc": math.log(max(1, v1)) - math.log(max(1, c1)),
            "c1": math.log(max(1, c1))}


def _fallback(p1, h):
    return [max(1.0, p1 * math.exp(z * SIGMA * math.sqrt(h - 1))) for z in Z]


# ---------------------------------------------------------------- model

def forecast(p1, c1, hist, comps):
    pool_g = filter_pool(comps)
    tfeat = _fmap(p1, c1, hist.get("pre_peak", 1), hist.get("pre_last", 1))
    # Size-effect saturation: freeze the (beta_v1 - 1) term at the largest
    # comparable's ln(v1) so huge hour-1 prints behave like the biggest
    # past launches instead of extrapolating the size trend downward.
    lv_max = None
    pool_lv = [math.log(max(1, g["v1"])) for g in pool_g if g["v1"] > 0]
    if pool_lv:
        lv_max = max(pool_lv)
    lv = math.log(max(1, p1))

    def sat(med, beta):
        if lv_max is None or "v1" not in FEATS:
            return med
        j = FEATS.index("v1") + 1
        return med + (beta[j] - 1.0) * (min(lv, lv_max) - lv)

    u_t = unlock_hour(hist["_day_start"])
    out = {}

    # --- aligned jump region: hours 5..9, aligned index a = h - u ---
    aligned_rows = {}
    for g in pool_g:
        u_g = unlock_hour(g["_day_start"])
        for h in range(5, 10):
            v = g["curve"].get(h)
            if v and v > 0:
                a = h - u_g
                gf = _fmap(g["v1"], g.get("c1", 1), g.get("pre_peak", 1), g.get("pre_last", 1))
                x = [1.0] + [gf[f] for f in FEATS]
                aligned_rows.setdefault(a, []).append((x, math.log(v), plat_weight(g)))
    for h in range(5, 10):
        a = h - u_t
        rows = aligned_rows.get(a, [])
        tx = [1.0] + [tfeat[f] for f in FEATS]
        if len(rows) < 10:
            out[h] = _fallback(p1, h)
            continue
        r = _fit_predict(rows, tx, SHRINK, FEATS)
        if r is None:
            out[h] = _fallback(p1, h)
            continue
        med, res, beta = r
        med = sat(med, beta)
        qs = [med + _quantile(res, tau) for tau in LEVELS]
        out[h] = [max(1.0, math.exp(med + (q - med) * SPREAD_MULT)) for q in qs]

    # --- other hours: per-hour OLS ---
    for h in list(range(2, 5)) + list(range(10, 25)):
        rows = []
        for g in pool_g:
            v = g["curve"].get(h)
            if v and v > 0:
                gf = _fmap(g["v1"], g.get("c1", 1), g.get("pre_peak", 1), g.get("pre_last", 1))
                x = [1.0] + [gf[f] for f in FEATS]
                rows.append((x, math.log(v), plat_weight(g)))
        tx = [1.0] + [tfeat[f] for f in FEATS]
        if len(rows) < 10:
            out[h] = _fallback(p1, h)
            continue
        r = _fit_predict(rows, tx, SHRINK, FEATS)
        if r is None:
            out[h] = _fallback(p1, h)
            continue
        med, res, beta = r
        med = sat(med, beta)
        qs = [med + _quantile(res, tau) for tau in LEVELS]
        out[h] = [max(1.0, math.exp(med + (q - med) * SPREAD_MULT)) for q in qs]
    return out


# ---------------------------------------------------------------- entry point

def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    c1 = first.get("channels") or 0
    comps = [comp_features(c) for c in case["comparables"]]
    hist = target_hist(case)
    fc = forecast(p1, c1, hist, comps)
    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        qs = []
        prev = 0.0
        for q in fc[h]:
            q = round2(min(1e9, max(1.0, q)))
            q = max(q, prev)  # keep non-decreasing after rounding
            qs.append(q)
            prev = q
        hours.append({"hour": h, "q": qs})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        fc = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(fc, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
