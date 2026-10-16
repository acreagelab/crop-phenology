"""Per-pixel land surface phenology from irregular Sentinel-2 observations.

Every pixel is processed from its own observations only (no composites, no spatial or interannual fills).

screen_block (raw uint16 DN, reflectance = DN * 1e-4):
  valid = SCL mask & B08 > 0 & B04 > 0
  snow  = valid & NDSI (B03-B11)/(B03+B11) > SNOW_NDSI & B03 > SNOW_B03 -> dropped from the fit, kept for the
          snow_at_sos bit
  VI (EVI2 or NDVI) outside VI_RANGE -> dropped; haze (B02 > HAZE_B02) -> initial weight HAZE_W
pixel (daily grid 1..ndays):
  1. floor: a Whittaker pre-fit with one upper-envelope pass on the snow-free obs; floor = BASE_PCT percentile of
     the pre-fit at the obs dates (robust to residual-cloud dips, which a percentile of the raw obs is not)
  2. obs more than DIP below the floor (residual cloud / shadow) and positive spikes (more than SPIKE above every
     obs within SPIKE_DAYS on both sides) are dropped; the rest are raised to the floor
  3. dormant pseudo-obs (weight DORMANT_W) every DORMANT_STEP d from DORMANT_GAP d before the first obs (value: floor)
     and after the last obs (value: max(floor, min of the last 3 obs), so fall stubble is not pulled to spring soil
     under winter snow); day 1 / day ndays are added when they are >= DORMANT_GAP d away from the obs
  4. weighted Whittaker (2nd differences, LDL' pentadiagonal solver), lam = LAMBDA_REF * clip(n_obs / LAMBDA_NREF,
     LAMBDA_CLIP); ENVELOPE_ITER upper-envelope passes (obs below the curve get w0 * ENVELOPE_W, the first pass
     from the pre-fit); the curve is clipped to [max(VI_RANGE[0], floor), VI_RANGE[1]] (amplitude <= 1.2)
  5. PoS = argmax in SEARCH; base_L / base_R = min of the curve before / after PoS; base = mean of both;
     no season if min(peak - base_L, peak - base_R) < MIN_AMP
  6. SoS / SoS50 = last day before PoS where the curve crosses base_L + {SOS_FRAC, MID_FRAC} * (peak - base_L);
     EoS50 / EoS = first day after PoS where it crosses base_R + {MID_FRAC, EOS_FRAC} * (peak - base_R);
     scanned inside SEARCH only, NaN + bit when not found (never clamped)
  7. LoS = EoS - SoS; AUC = trapezoid integral of max(curve - base, 0) over SoS..EoS (<= LoS * amplitude)
  8. QA: n_obs (kept obs in SEARCH), max_gap in SEARCH; gap_sos / gap_eos = largest gap between kept obs over the
     green-up ramp SoS..SoS50 / senescence ramp EoS50..EoS (bits 4 / 8 above MAX_GAP); the same on the acquisition
     dates (bits 64 / 128 above DATE_GAP, input defects); 256 snow obs within SNOW_SOS_WIN d of SoS; 512 first
     in-season obs already green; 1024 peak above the highest obs + OVERSHOOT_TOL; 1 no season; 2 n_obs < MIN_OBS;
     16 / 32 crossing outside SEARCH
Output bands: config.OUT_BANDS; encode / decode convert to / from int16 with config.OUT_SCALE and NODATA.
"""
import hashlib
import json

import numpy as np
from numba import njit

LOW_AMP, FEW_OBS, GAP_SOS, GAP_EOS, SOS_OUT, EOS_OUT = 1, 2, 4, 8, 16, 32
DATE_GAP_SOS, DATE_GAP_EOS, SNOW_SOS, GREEN_START, OVERSHOOT = 64, 128, 256, 512, 1024
N_OUT = 14  # SoS SoS50 PoS EoS50 EoS LoS AUC amplitude base n_obs max_gap gap_sos gap_eos qa
O_SOS, O_SOS50, O_POS, O_EOS50, O_EOS, O_LOS, O_AUC, O_AMP, O_BASE, O_NOBS, O_MAXGAP, O_GAPS, O_GAPE, O_QA = range(14)
WORK_ROWS = 8  # whittaker rows 0-3 (1/D, l1, l2, z); pixel rows 4-7 (y, w, kind, w0)

(P_VI, P_LO, P_HI, P_BASE_PCT, P_LAM, P_NREF, P_CLIP0, P_CLIP1, P_ENV_IT, P_ENV_W, P_S0, P_S1, P_SOS_F, P_EOS_F,
 P_MID_F, P_MIN_AMP, P_MIN_OBS, P_MAX_GAP, P_DATE_GAP, P_NDSI, P_B03, P_HAZE_B02, P_HAZE_W, P_SPIKE, P_SPIKE_D,
 P_DSTEP, P_DGAP, P_DW, P_SNOW_WIN, P_GREEN_F, P_OVER_TOL, P_DIP) = range(32)
VI_CODES = {"evi2": 0, "ndvi": 1}
HASH_KEYS = ["VI", "VI_RANGE", "BASE_PCT", "LAMBDA_REF", "LAMBDA_NREF", "LAMBDA_CLIP", "ENVELOPE_ITER", "ENVELOPE_W",
             "SEARCH", "SOS_FRAC", "EOS_FRAC", "MID_FRAC", "MIN_AMP", "MIN_OBS", "MAX_GAP", "DATE_GAP", "SNOW_NDSI",
             "SNOW_B03", "HAZE_B02", "HAZE_W", "SPIKE", "SPIKE_DAYS", "DORMANT_STEP", "DORMANT_GAP", "DORMANT_W",
             "SNOW_SOS_WIN", "GREEN_START_FRAC", "OVERSHOOT_TOL", "DIP", "OUT_BANDS", "OUT_SCALE"]
EVI_BINS, EVI_LO, EVI_STEP = 240, -0.2, 0.005
REF_BINS, REF_STEP = 500, 0.001


def ndays(year):
    return 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365


def vegetation_index(name, red, nir):
    """red / nir reflectance (0-1)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        if name == "evi2":
            return 2.5 * (nir - red) / (nir + 2.4 * red + 1.0)
        if name == "ndvi":
            return (nir - red) / (nir + red)
    raise ValueError(f"unknown VI {name}")


def params_array(cfg):
    if cfg.VI not in VI_CODES:
        raise ValueError(f"unknown VI {cfg.VI}")
    return np.array([VI_CODES[cfg.VI], cfg.VI_RANGE[0], cfg.VI_RANGE[1], cfg.BASE_PCT, cfg.LAMBDA_REF, cfg.LAMBDA_NREF,
                     cfg.LAMBDA_CLIP[0], cfg.LAMBDA_CLIP[1], cfg.ENVELOPE_ITER, cfg.ENVELOPE_W, cfg.SEARCH[0],
                     cfg.SEARCH[1], cfg.SOS_FRAC, cfg.EOS_FRAC, cfg.MID_FRAC, cfg.MIN_AMP, cfg.MIN_OBS, cfg.MAX_GAP,
                     cfg.DATE_GAP, cfg.SNOW_NDSI, cfg.SNOW_B03, cfg.HAZE_B02, cfg.HAZE_W, cfg.SPIKE, cfg.SPIKE_DAYS,
                     cfg.DORMANT_STEP, cfg.DORMANT_GAP, cfg.DORMANT_W, cfg.SNOW_SOS_WIN, cfg.GREEN_START_FRAC,
                     cfg.OVERSHOOT_TOL, cfg.DIP], np.float64)


def params_hash(cfg):
    d = {k: getattr(cfg, k) for k in HASH_KEYS}
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=list).encode()).hexdigest()


def band_indices(cfg):
    """Store positions of B04, B08, B02, B03, B11 (screen_block / date_stats order)."""
    return np.array([cfg.STORE_BANDS.index(b) for b in ("B04", "B08", "B02", "B03", "B11")], np.int64)


@njit(cache=True)
def whittaker_ref(y, w, lam):
    """Solve (W + lam * D'D) z = W y, D = 2nd-difference operator, by banded Cholesky (bandwidth 2)."""
    n = y.size
    d0 = np.empty(n)
    d1 = np.zeros(n)
    d2 = np.zeros(n)
    for i in range(n):
        c = 6.0
        if i == 0 or i == n - 1:
            c = 1.0
        elif i == 1 or i == n - 2:
            c = 5.0
        d0[i] = w[i] + lam * c
        if i >= 1:
            d1[i] = lam * (-2.0 if (i == 1 or i == n - 1) else -4.0)
        if i >= 2:
            d2[i] = lam
    l0 = np.empty(n)
    l1 = np.zeros(n)
    l2 = np.zeros(n)
    for i in range(n):
        if i >= 2:
            l2[i] = d2[i] / l0[i - 2]
        if i >= 1:
            l1[i] = (d1[i] - (l2[i] * l1[i - 1] if i >= 2 else 0.0)) / l0[i - 1]
        l0[i] = np.sqrt(d0[i] - l1[i] ** 2 - l2[i] ** 2)
    b = w * y
    z = np.empty(n)
    for i in range(n):
        s = b[i]
        if i >= 1:
            s -= l1[i] * z[i - 1]
        if i >= 2:
            s -= l2[i] * z[i - 2]
        z[i] = s / l0[i]
    for i in range(n - 1, -1, -1):
        s = z[i]
        if i + 1 < n:
            s -= l1[i + 1] * z[i + 1]
        if i + 2 < n:
            s -= l2[i + 2] * z[i + 2]
        z[i] = s / l0[i]
    return z


@njit(cache=True)
def whittaker(y, w, lam, work):
    """Same system as whittaker_ref by LDL' (pentadiagonal, n >= 4); work float64 (>= 4, n): rows 1/D, l1, l2, z.
    Returns the view work[3]; no allocations."""
    n = y.size
    iD, l1, l2, z = work[0], work[1], work[2], work[3]
    m4 = -4.0 * lam
    D0 = w[0] + lam
    iD[0] = 1.0 / D0
    l1[0] = 0.0
    l2[0] = 0.0
    z[0] = w[0] * y[0]
    l1[1] = -2.0 * lam * iD[0]
    l2[1] = 0.0
    D1 = w[1] + 5.0 * lam - l1[1] * l1[1] * D0
    iD[1] = 1.0 / D1
    z[1] = w[1] * y[1] - l1[1] * z[0]
    Dm2, Dm1 = D0, D1
    for i in range(2, n):
        if i == n - 1:
            c, e = lam, -2.0 * lam
        elif i == n - 2:
            c, e = 5.0 * lam, m4
        else:
            c, e = 6.0 * lam, m4
        a2 = lam * iD[i - 2]
        a1 = (e - a2 * l1[i - 1] * Dm2) * iD[i - 1]
        l2[i] = a2
        l1[i] = a1
        Di = w[i] + c - a1 * a1 * Dm1 - a2 * a2 * Dm2
        iD[i] = 1.0 / Di
        z[i] = w[i] * y[i] - a1 * z[i - 1] - a2 * z[i - 2]
        Dm2, Dm1 = Dm1, Di
    z[n - 1] = z[n - 1] * iD[n - 1]
    z[n - 2] = z[n - 2] * iD[n - 2] - l1[n - 1] * z[n - 1]
    for i in range(n - 3, -1, -1):
        z[i] = z[i] * iD[i] - l1[i + 1] * z[i + 1] - l2[i + 2] * z[i + 2]
    return z


@njit(cache=True)
def screen_block(raw, valid, bidx, params):
    """raw uint16 (T,h,w,10), valid (T,h,w), bidx = band_indices -> vi f4 (P,T) NaN = not usable, w f4, snow u1."""
    T, H, W = raw.shape[0], raw.shape[1], raw.shape[2]
    P = H * W
    ir, inr, ib, ig, isw = bidx[0], bidx[1], bidx[2], bidx[3], bidx[4]
    is_ndvi = params[P_VI] == 1
    lo, hi = params[P_LO], params[P_HI]
    ndsi_t, b03_t, haze_t, haze_w = params[P_NDSI], params[P_B03] * 1e4, params[P_HAZE_B02] * 1e4, params[P_HAZE_W]
    vi = np.full((P, T), np.nan, np.float32)
    wt = np.zeros((P, T), np.float32)
    snow = np.zeros((P, T), np.uint8)
    for t in range(T):
        for r in range(H):
            for c in range(W):
                if not valid[t, r, c]:
                    continue
                red, nir = float(raw[t, r, c, ir]), float(raw[t, r, c, inr])
                if red == 0.0 or nir == 0.0:
                    continue
                p = r * W + c
                g, s = float(raw[t, r, c, ig]), float(raw[t, r, c, isw])
                if g > b03_t and g - s > ndsi_t * (g + s):
                    snow[p, t] = 1
                    continue
                v = (nir - red) / (nir + red) if is_ndvi else 2.5 * (nir - red) / (nir + 2.4 * red + 1e4)
                if v < lo or v > hi:
                    continue
                vi[p, t] = v
                wt[p, t] = haze_w if float(raw[t, r, c, ib]) > haze_t else 1.0
    return vi, wt, snow


@njit(cache=True)
def date_stats(raw, valid, bidx, params, counts, h_vi, h_b02, h_b04):
    """Accumulate per-date counts (T,4: n_px, valid, snow, haze) and histograms of EVI2 (T,EVI_BINS) and B02 / B04
    reflectance (T,REF_BINS) over valid non-snow obs, with the screen_block rules."""
    T, H, W = raw.shape[0], raw.shape[1], raw.shape[2]
    ir, inr, ib, ig, isw = bidx[0], bidx[1], bidx[2], bidx[3], bidx[4]
    ndsi_t, b03_t, haze_t = params[P_NDSI], params[P_B03] * 1e4, params[P_HAZE_B02] * 1e4
    inv_vi, inv_ref = 1.0 / EVI_STEP, 1e-4 / REF_STEP
    for t in range(T):
        nval = nsnow = nhaze = 0
        for r in range(H):
            for c in range(W):
                if not valid[t, r, c]:
                    continue
                red, nir = float(raw[t, r, c, ir]), float(raw[t, r, c, inr])
                if red == 0.0 or nir == 0.0:
                    continue
                nval += 1
                g, s = float(raw[t, r, c, ig]), float(raw[t, r, c, isw])
                if g > b03_t and g - s > ndsi_t * (g + s):
                    nsnow += 1
                    continue
                b = float(raw[t, r, c, ib])
                if b > haze_t:
                    nhaze += 1
                v = 2.5 * (nir - red) / (nir + 2.4 * red + 1e4)
                k = int((min(max(v, EVI_LO), 1.0) - EVI_LO) * inv_vi)
                h_vi[t, min(k, EVI_BINS - 1)] += 1
                h_b02[t, min(int(b * inv_ref), REF_BINS - 1)] += 1
                h_b04[t, min(int(red * inv_ref), REF_BINS - 1)] += 1
        counts[t, 0] += H * W
        counts[t, 1] += nval
        counts[t, 2] += nsnow
        counts[t, 3] += nhaze


def hist_quantile(h, q, lo, step):
    """Quantile per row of histograms h (T,B) at bin centres; NaN for empty rows."""
    n = h.sum(1)
    cum = np.cumsum(h, 1)
    k = (cum < (q * n)[:, None]).sum(1)
    return np.where(n > 0, lo + (np.minimum(k, h.shape[1] - 1) + 0.5) * step, np.nan)


@njit(cache=True)
def _ramp_gap(arr, n, d0, d1, nd):
    """Largest gap between consecutive arr days (padded by 1 and nd) over the intervals overlapping [d0, d1]."""
    prev, g = 1, 0
    for k in range(n):
        d = arr[k]
        if d <= d0:
            prev = d
        elif d <= d1:
            g = max(g, d - prev)
            prev = d
        else:
            return max(g, d - prev)
    return max(g, nd - prev)


@njit(cache=True)
def pixel(vi, w0, snow, doys, acq, nd, params, work):
    """One pixel: vi f8 (T,) NaN = invalid, w0 f8 (T,), snow u1 (T,), doys i8 (T,) sorted, acq i8 (A,) sorted
    acquisition DOYs, nd days, work f8 (WORK_ROWS, nd). Returns f8 (N_OUT,) in config.OUT_BANDS order."""
    out = np.full(N_OUT, np.nan)
    T = vi.size
    s0, s1 = int(params[P_S0]), int(params[P_S1])
    ok = np.empty(T, np.bool_)
    for k in range(T):
        ok[k] = np.isfinite(vi[k]) and snow[k] == 0
    idx = np.flatnonzero(ok)
    n0 = idx.size
    floor = 0.0
    below = np.zeros(T, np.bool_)
    if n0 >= 3:
        y, w, wd = work[4], work[5], work[7]
        for i in range(nd):
            y[i] = 0.0
            w[i] = 0.0
        for k in idx:
            y[doys[k] - 1] = vi[k]
            w[doys[k] - 1] = w0[k]
            wd[doys[k] - 1] = w0[k]
        lam0 = params[P_LAM] * min(max(n0 / params[P_NREF], params[P_CLIP0]), params[P_CLIP1])
        z = whittaker(y, w, lam0, work)
        for k in idx:
            i = doys[k] - 1
            w[i] = wd[i] * (params[P_ENV_W] if y[i] < z[i] else 1.0)
        z = whittaker(y, w, lam0, work)
        zo = np.empty(n0)
        for j in range(n0):
            zo[j] = z[doys[idx[j]] - 1]
            below[idx[j]] = vi[idx[j]] < zo[j]
        floor = np.percentile(zo, params[P_BASE_PCT])
        for k in idx:
            if vi[k] < floor - params[P_DIP]:
                ok[k] = False
        idx = np.flatnonzero(ok)
        n0 = idx.size
    sd = params[P_SPIKE_D]
    for j in range(1, n0 - 1):
        k = idx[j]
        ml, mr = -1e9, -1e9
        jj = j - 1
        while jj >= 0 and doys[k] - doys[idx[jj]] <= sd:
            ml = max(ml, vi[idx[jj]])
            jj -= 1
        jj = j + 1
        while jj < n0 and doys[idx[jj]] - doys[k] <= sd:
            mr = max(mr, vi[idx[jj]])
            jj += 1
        if ml > -1e9 and mr > -1e9 and vi[k] - max(ml, mr) > params[P_SPIKE]:
            ok[k] = False
    vd = np.empty(T, np.int64)
    vv = np.empty(T)
    nv = 0
    n_obs = 0
    last = s0
    mg = 0
    for k in range(T):
        if ok[k]:
            d = doys[k]
            vd[nv] = d
            vv[nv] = vi[k]
            nv += 1
            if s0 <= d <= s1:
                n_obs += 1
                mg = max(mg, d - last)
                last = d
    mg = max(mg, s1 - last)
    out[O_NOBS] = n_obs
    out[O_MAXGAP] = mg
    qa = 0
    if n_obs < params[P_MIN_OBS]:
        qa |= FEW_OBS
    if nv < 3:
        out[O_QA] = qa | LOW_AMP
        return out
    y, w, kind, wd = work[4], work[5], work[6], work[7]
    for i in range(nd):
        y[i] = 0.0
        w[i] = 0.0
        kind[i] = 0.0
    ymax = -1e9
    for k in range(T):
        if ok[k]:
            i = doys[k] - 1
            y[i] = max(vi[k], floor)
            ymax = max(ymax, y[i])
            wd[i] = w0[k]
            w[i] = w0[k] * (params[P_ENV_W] if below[k] else 1.0)
            kind[i] = 1.0
    step, gap, dw = int(params[P_DSTEP]), int(params[P_DGAP]), params[P_DW]
    first, lastd = vd[0], vd[nv - 1]
    d = first - gap
    while d >= 1:
        y[d - 1] = floor
        w[d - 1] = dw
        kind[d - 1] = 2.0
        d -= step
    yr = vv[nv - 1]
    for j in range(max(0, nv - 3), nv):
        yr = min(yr, vv[j])
    yr = max(yr, floor)
    d = lastd + gap
    while d <= nd:
        y[d - 1] = yr
        w[d - 1] = dw
        kind[d - 1] = 2.0
        d += step
    if kind[0] == 0.0 and first - 1 >= gap:
        y[0] = floor
        w[0] = dw
        kind[0] = 2.0
    if kind[nd - 1] == 0.0 and nd - lastd >= gap:
        y[nd - 1] = yr
        w[nd - 1] = dw
        kind[nd - 1] = 2.0
    lam = params[P_LAM] * min(max(n_obs / params[P_NREF], params[P_CLIP0]), params[P_CLIP1])
    z = whittaker(y, w, lam, work)
    for _ in range(int(params[P_ENV_IT]) - 1):
        for i in range(nd):
            if kind[i] == 1.0:
                w[i] = wd[i] * (params[P_ENV_W] if y[i] < z[i] else 1.0)
        z = whittaker(y, w, lam, work)
    lo, hi = max(params[P_LO], floor), params[P_HI]
    for i in range(nd):
        z[i] = min(max(z[i], lo), hi)
    a, b = s0 - 1, min(s1, nd) - 1
    p = a
    for i in range(a, b + 1):
        if z[i] > z[p]:
            p = i
    peak = z[p]
    bl = peak
    for i in range(p + 1):
        bl = min(bl, z[i])
    br = peak
    for i in range(p, nd):
        br = min(br, z[i])
    base = 0.5 * (bl + br)
    out[O_AMP] = peak - base
    out[O_BASE] = base
    if peak > ymax + params[P_OVER_TOL]:
        qa |= OVERSHOOT
    if min(peak - bl, peak - br) < params[P_MIN_AMP]:
        out[O_QA] = qa | LOW_AMP
        return out
    out[O_POS] = p + 1
    ampl, ampr = peak - bl, peak - br
    ts, ts50 = bl + params[P_SOS_F] * ampl, bl + params[P_MID_F] * ampl
    te, te50 = br + params[P_EOS_F] * ampr, br + params[P_MID_F] * ampr
    si = si50 = ei = ei50 = -1
    for i in range(p, a - 1, -1):
        if si50 < 0 and z[i] < ts50:
            si50 = i + 1
        if z[i] < ts:
            si = i + 1
            break
    for i in range(p, b + 1):
        if ei50 < 0 and z[i] < te50:
            ei50 = i - 1
        if z[i] < te:
            ei = i - 1
            break
    if si < 0:
        qa |= SOS_OUT
    if ei < 0:
        qa |= EOS_OUT
    if si50 >= 0:
        out[O_SOS50] = si50 + 1
    if ei50 >= 0:
        out[O_EOS50] = ei50 + 1
    max_gap_ok, date_gap = params[P_MAX_GAP], params[P_DATE_GAP]
    if si >= 0:
        sos = si + 1
        out[O_SOS] = sos
        s50 = si50 + 1 if si50 >= 0 else sos
        g = _ramp_gap(vd, nv, sos, s50, nd)
        out[O_GAPS] = g
        if g > max_gap_ok:
            qa |= GAP_SOS
        if _ramp_gap(acq, acq.size, sos, s50, nd) > date_gap:
            qa |= DATE_GAP_SOS
        lsnow = -1
        for k in range(T):
            if snow[k] and doys[k] < p + 1:
                lsnow = doys[k]
        if lsnow >= 0 and lsnow >= sos - params[P_SNOW_WIN]:
            qa |= SNOW_SOS
    if ei >= 0:
        eos = ei + 1
        out[O_EOS] = eos
        e50 = ei50 + 1 if ei50 >= 0 else eos
        g = _ramp_gap(vd, nv, e50, eos, nd)
        out[O_GAPE] = g
        if g > max_gap_ok:
            qa |= GAP_EOS
        if _ramp_gap(acq, acq.size, e50, eos, nd) > date_gap:
            qa |= DATE_GAP_EOS
    if si >= 0 and ei >= 0:
        out[O_LOS] = ei - si
        s = 0.0
        prev = max(z[si] - base, 0.0)
        for i in range(si + 1, ei + 1):
            cur = max(z[i] - base, 0.0)
            s += 0.5 * (prev + cur)
            prev = cur
        out[O_AUC] = s
    for j in range(nv):
        if vd[j] >= s0:
            if vd[j] <= s1 and vd[j] < p + 1 and max(vv[j], floor) >= bl + params[P_GREEN_F] * ampl:
                qa |= GREEN_START
            break
    out[O_QA] = qa
    return out


@njit(cache=True)
def block(vi, w, snow, doys, acq, nd, params):
    """vi f4 (P,T) NaN = invalid, w f4 (P,T), snow u1 (P,T), doys / acq i8 sorted -> f4 (N_OUT, P)."""
    P, T = vi.shape
    res = np.empty((N_OUT, P), np.float32)
    work = np.zeros((WORK_ROWS, nd))
    v = np.empty(T)
    ww = np.empty(T)
    for j in range(P):
        for k in range(T):
            v[k] = vi[j, k]
            ww[k] = w[j, k]
        o = pixel(v, ww, snow[j], doys, acq, nd, params, work)
        for i in range(N_OUT):
            res[i, j] = o[i]
    return res


def _check_doys(doys, nd):
    d = np.asarray(doys)
    if d.size and (d.min() < 1 or d.max() > nd):
        raise ValueError(f"DOY outside 1..{nd}")
    if np.unique(d).size != d.size:
        raise ValueError("duplicate DOYs")


def phenology(vi, doys, year, cfg, weights=None, snow=None, acq_doys=None):
    """vi (T, y, x) NaN-masked, doys (T,) int; optional weights / snow (T, y, x), acq_doys (A,) (default doys).
    Returns (N_OUT, y, x) float32."""
    nd = ndays(year)
    doys = np.asarray(doys).astype(np.int64)
    _check_doys(doys, nd)
    acq = np.sort(np.asarray(doys if acq_doys is None else acq_doys).astype(np.int64))
    _check_doys(acq, nd)
    T, H, W = vi.shape
    lo, hi = cfg.VI_RANGE
    v = np.where((vi >= lo) & (vi <= hi), vi, np.nan).reshape(T, H * W)
    w = np.ones_like(v) if weights is None else np.asarray(weights, np.float32).reshape(T, H * W)
    s = np.zeros(v.shape, np.uint8) if snow is None else np.asarray(snow).astype(np.uint8).reshape(T, H * W)
    o = np.argsort(doys, kind="stable")
    f = lambda a, t: np.ascontiguousarray(a[o].T, dtype=t)  # noqa: E731
    res = block(f(v, np.float32), f(w, np.float32), f(s, np.uint8), doys[o], acq, nd, params_array(cfg))
    return res.reshape(N_OUT, H, W)


def _scales(cfg):
    return np.array([cfg.OUT_SCALE.get(b, 1.0) for b in cfg.OUT_BANDS], np.float64)


def encode(res, cfg):
    """float (N_OUT, ...) NaN = nodata -> int16 with config.OUT_SCALE (stored = round(physical / scale))."""
    sc = _scales(cfg).reshape((-1,) + (1,) * (res.ndim - 1))
    with np.errstate(invalid="ignore"):
        q = np.clip(np.rint(res / sc), -32767, 32767)
    return np.where(np.isfinite(q), q, cfg.NODATA).astype(np.int16)


def decode(arr, cfg, band=None):
    """int16 (N_OUT, ...) or a single band (with its name) -> float32 physical values, NaN = nodata."""
    a = np.asarray(arr)
    if band is not None:
        sc = cfg.OUT_SCALE.get(band, 1.0)
    else:
        sc = _scales(cfg).reshape((-1,) + (1,) * (a.ndim - 1))
    return np.where(a == cfg.NODATA, np.nan, a * sc).astype(np.float32)
