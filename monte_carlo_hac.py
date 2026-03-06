"""
Monte Carlo Evidence for Variable-Window HAC Estimators  —  V2 (patched)
=========================================================================
Reproduces and extends the simulations described in the paper
"Bulk-Boundary Decomposition for Adaptive Windows in Spectral/HAC Estimation".

PATCHES APPLIED (round 1 — Patches 1–4)
  Patch 1 — t0 fixed; Patch 2 — verdict; Patch 3 — pos controls; Patch 4 — W_hat curves

FIXES APPLIED (round 2 — Fixes 1–6)
  Fix 1–2 — certified/sign consistency + stable CSV column order
  Fix 3   — R column in MC tables; Fix 4 — ROOT-relative paths
  Fix 5–6 — MC6 W_hat_AXN guard; MC5 "refinedcheck" label

EDITORIAL FIXES (round 3 — Fixes 7–12)
  Fix 7  — run_parameter_map: refined_pts (not certified_pts); stable_neg (not is_cert);
            print message: "NOT certified; certified points reported in Appendix A"
  Fix 8  — fig_MC4 legend: "refined-stable negative (NOT certified)" / "unstable under refinement"
  Fix 9  — fig_MC5 per-panel: "stable-neg" / "unstable" (no "robust" language)
  Fix 10 — fig_MC6 suptitle: "sign certified at t0 only"; per-panel says same;
            dynamic group list from actual data in cert_df
  Fix 11 — Group D rebuilt as neighbourhood-positive check (all 9 neighbours > 0);
            E3s docstring updated to Route-A1 mitigation framing (no "confirms mechanism")
  Fix 12 — df_to_latex helper; Table_MC4_E3s.tex + Table_A1_certified.tex auto-exported

DATA-GENERATING PROCESSES
  DGP1: AR(1)-GARCH(1,1)
  DGP2: Markov-switching AR(1)

ESTIMATORS
  E0   Fixed hard-cutoff rectangular baseline
  E1   Bartlett fixed-bandwidth (Newey-West style)
  E2   Andrews-style QS adaptive bandwidth
  E3h  State-dependent HARD-CUTOFF moving truncation
  E3s  State-dependent SMOOTH-KERNEL moving bandwidth (Bartlett)  [NEW in V2]
       Diagnostic/mitigation: near-zero negativity vs E3h supports Route A1
       (smooth kernel mitigates boundary-discontinuity pathology in finite samples)
"""

import numpy as np
import pandas as pd
from scipy.special import expit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── NumPy compatibility shim (trapz renamed in NumPy 2.0) ────────────────────
if hasattr(np, "trapezoid"):
    _trapezoid = np.trapezoid
else:
    _trapezoid = np.trapz

# ── Output directory (Fix 4: ROOT-relative so cache is stable in Spyder/Windows)
ROOT  = Path(__file__).resolve().parent
OUT   = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
CACHE = OUT / "cache"
CACHE.mkdir(exist_ok=True)

# ── Global settings ───────────────────────────────────────────────────────────
SEED_BASE  = 12345
T_LIST     = [250, 500, 1000]
R_DICT     = {250: 2000, 500: 2000, 1000: 1000}
OMEGA_GRID = np.linspace(0.0, np.pi, 257)
BETA_TRUE  = np.array([1.0, 0.0, 0.0])
C0         = 1.5

EST_LIST   = ["E0", "E1", "E2", "E3h", "E3s"]

_EPS  = 1e-12
_PI   = np.pi
_2PI  = 2.0 * np.pi


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA-GENERATING PROCESSES
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_DGP1(T, seed):
    """AR(1)-GARCH(1,1) errors."""
    rng = np.random.default_rng(seed)
    phi, omega_g, a, b = 0.6, 0.05, 0.10, 0.85
    u      = np.zeros(T + 500)
    eps    = np.zeros(T + 500)
    sigma2 = np.ones(T + 500) * omega_g / (1 - a - b)
    z      = rng.standard_normal(T + 500)
    for t in range(1, T + 500):
        sigma2[t] = omega_g + a * eps[t-1]**2 + b * sigma2[t-1]
        eps[t]    = np.sqrt(sigma2[t]) * z[t]
        u[t]      = phi * u[t-1] + eps[t]
    return u[-T:]


def simulate_DGP2(T, seed):
    """Markov-switching AR(1) errors."""
    rng = np.random.default_rng(seed)
    p00, p11 = 0.97, 0.90
    phi  = np.array([0.2, 0.8])
    sig  = np.array([1.0, 3.0])
    s    = np.zeros(T + 500, dtype=int)
    u    = np.zeros(T + 500)
    z    = rng.standard_normal(T + 500)
    for t in range(1, T + 500):
        p_stay = p11 if s[t-1] == 1 else p00
        s[t]   = s[t-1] if rng.random() < p_stay else 1 - s[t-1]
        u[t]   = phi[s[t]] * u[t-1] + sig[s[t]] * z[t]
    return u[-T:]


def simulate_regressors(T, seed):
    """Exogenous AR(1) regressors; returns (T,3) design matrix [1, x1, x2]."""
    rng = np.random.default_rng(seed)
    rho = 0.3
    x1 = np.zeros(T); x2 = np.zeros(T)
    e1, e2 = rng.standard_normal((2, T))
    for t in range(1, T):
        x1[t] = rho * x1[t-1] + e1[t]
        x2[t] = rho * x2[t-1] + e2[t]
    return np.column_stack([np.ones(T), x1, x2])


# ═══════════════════════════════════════════════════════════════════════════════
# 2. OLS + SANDWICH
# ═══════════════════════════════════════════════════════════════════════════════

def ols(y, X):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    uhat = y - X @ beta
    return beta, uhat


def sandwich_var(X, Omega_hat, T):
    """(X'X/T)^{-1} Omega (X'X/T)^{-1} / T"""
    XX  = X.T @ X / T
    XXi = np.linalg.inv(XX)
    return XXi @ Omega_hat @ XXi / T


def symmetrize(A):
    return 0.5 * (A + A.T)


def max_asymmetry(A):
    return float(np.max(np.abs(A - A.T)))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. KERNEL WEIGHTS
# ═══════════════════════════════════════════════════════════════════════════════

def bartlett_weights(m):
    k = np.arange(m + 1)
    return 1.0 - k / (m + 1)


def qs_kernel(x):
    x = np.asarray(x, dtype=float)
    scalar = (x.ndim == 0)
    x = np.atleast_1d(x)
    y = 6 * np.pi * x / 5
    w = np.ones_like(x)
    mask = np.abs(x) >= 1e-10
    y_m = y[mask]
    w[mask] = 3 / y_m**2 * (np.sin(y_m) / y_m - np.cos(y_m))
    return float(w[0]) if scalar else w


# ═══════════════════════════════════════════════════════════════════════════════
# 4. STATE RULE (bandwidth path m_t)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_state_mt(uhat, T, m_min=2, m_max=None, kappa=1.0, eps=1e-8):
    if m_max is None:
        m_max = max(m_min + 1, int(np.floor(2 * T**(1/3))))
    nu    = np.log(eps + uhat**2)
    med   = np.median(nu)
    mad   = np.median(np.abs(nu - med)) + 1e-12
    nu_z  = (nu - med) / mad
    g     = expit(kappa * nu_z)
    mt    = (m_min + np.floor((m_max - m_min) * g)).astype(int)
    return mt, m_min, m_max


# ═══════════════════════════════════════════════════════════════════════════════
# 5. HAC ESTIMATORS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_autocov(scores, max_lag):
    T, d = scores.shape
    Gamma = []
    for k in range(max_lag + 1):
        if k == 0:
            G = scores.T @ scores / T
        else:
            G = scores[k:].T @ scores[:T-k] / T
        Gamma.append(G)
    return Gamma


def HAC_E1(scores, T, c0=C0):
    m0   = max(1, int(np.floor(c0 * T**(1/3))))
    Gam  = compute_autocov(scores, m0)
    wts  = bartlett_weights(m0)
    Omega = wts[0] * Gam[0]
    for k in range(1, m0 + 1):
        Omega += wts[k] * (Gam[k] + Gam[k].T)
    return Omega


def HAC_E2(scores, T):
    d = scores.shape[1]
    rhos = []
    for j in range(d):
        v   = scores[:, j]
        rho = np.dot(v[1:], v[:-1]) / (np.dot(v[:-1], v[:-1]) + 1e-12)
        rho = np.clip(rho, -0.99, 0.99)
        rhos.append(rho)
    num = 0.0; den = 0.0
    for j, rho in enumerate(rhos):
        sig2  = np.var(scores[:, j])
        sig4  = sig2**2
        num  += 4 * rho**2 * sig4 / (1 - rho)**8
        den  += sig4 / (1 - rho)**4
    alpha2  = num / (den + 1e-12)
    b_hat   = max(1.0, 1.3221 * (alpha2 * T)**(0.2))
    max_lag = min(int(np.floor(b_hat)) + 1, T - 1)
    Gam     = compute_autocov(scores, max_lag)
    Omega   = qs_kernel(0) * Gam[0]
    for k in range(1, max_lag + 1):
        w      = qs_kernel(k / b_hat)
        Omega += w * (Gam[k] + Gam[k].T)
    return Omega


def HAC_E0(scores, T, c0=C0):
    m0   = max(1, int(np.floor(c0 * T**(1/3))))
    Gam  = compute_autocov(scores, m0)
    Omega = Gam[0]
    for k in range(1, m0 + 1):
        Omega += (Gam[k] + Gam[k].T)
    return Omega


def HAC_E3h(scores, uhat, T):
    mt, m_min, m_max = compute_state_mt(uhat, T, m_min=2, m_max=None, kappa=1.0, eps=1e-8)
    d     = scores.shape[1]
    Omega = np.zeros((d, d))
    for k in range(-m_max, m_max + 1):
        valid_t = np.where(mt >= abs(k))[0]
        tk      = valid_t - k
        in_range = (tk >= 0) & (tk < T)
        valid_t  = valid_t[in_range]
        tk       = tk[in_range]
        if len(valid_t) == 0:
            continue
        S = scores[valid_t].T @ scores[tk]
        Omega += S
    Omega /= T
    return Omega


def PSD_project(Omega):
    Omega = symmetrize(Omega)
    vals, vecs = np.linalg.eigh(Omega)
    vals_clipped = np.maximum(vals, 0.0)
    return vecs @ np.diag(vals_clipped) @ vecs.T


def HAC_E3s(scores, uhat, T):
    """
    State-dependent SMOOTH-KERNEL (Bartlett) moving-bandwidth HAC (E3s).  [NEW V2]

    Same state rule m_t as E3h, but uses Bartlett weights K_B(k/(m_t+1)).

    Purpose (Route A1): diagnostic / mitigation check.
      - If E3h exhibits negativity/non-PSD failures while E3s does not, this
        supports that the pathology is tied to the interaction of state-dependent
        truncation and boundary discontinuity, and that smooth kernel weighting
        mitigates it in finite samples.
      - A near-zero negativity rate for E3s (relative to E3h) is a positive
        finding: it shows smooth kernel weighting provides mitigation evidence,
        not that the moving-boundary mechanism is absent.
    """
    mt, m_min, m_max = compute_state_mt(uhat, T, m_min=2, m_max=None, kappa=1.0, eps=1e-8)
    d     = scores.shape[1]
    Omega = np.zeros((d, d))
    for k in range(-m_max, m_max + 1):
        absk    = abs(k)
        valid_t = np.where(mt >= absk)[0]
        tk      = valid_t - k
        in_range = (tk >= 0) & (tk < T)
        valid_t  = valid_t[in_range]
        tk       = tk[in_range]
        if len(valid_t) == 0:
            continue
        wk = 1.0 - absk / (mt[valid_t].astype(float) + 1.0)
        S  = (scores[valid_t] * wk[:, None]).T @ scores[tk]
        Omega += S
    Omega /= T
    return Omega


# ═══════════════════════════════════════════════════════════════════════════════
# 6. SPECTRAL ESTIMATORS (univariate)
# ═══════════════════════════════════════════════════════════════════════════════

def spectrum_E1(v, T, c0=C0):
    m0  = max(1, int(np.floor(c0 * T**(1/3))))
    gam = np.array([np.dot(v[:T-k], v[k:]) / T for k in range(m0 + 1)])
    wts = bartlett_weights(m0)
    ks  = np.arange(1, m0 + 1)
    fhat = np.array([
        (wts[0] * gam[0] + 2 * np.sum(wts[1:] * gam[1:] * np.cos(om * ks))) / (2 * np.pi)
        for om in OMEGA_GRID
    ])
    return fhat


def spectrum_E2(v, T):
    rho  = np.dot(v[1:], v[:-1]) / (np.dot(v[:-1], v[:-1]) + 1e-12)
    rho  = np.clip(rho, -0.99, 0.99)
    sig2 = np.var(v)
    alpha2 = 4 * rho**2 * sig2**2 / (1-rho)**8 / (sig2**2/(1-rho)**4 + 1e-12)
    b    = max(1.0, 1.3221 * (alpha2 * T)**0.2)
    m    = min(int(np.floor(b)) + 1, T - 1)
    gam  = np.array([np.dot(v[:T-k], v[k:]) / T for k in range(m + 1)])
    ks   = np.arange(1, m + 1)
    wks  = qs_kernel(ks / b)
    fhat = np.array([
        (qs_kernel(0) * gam[0] + 2 * np.sum(wks * gam[1:] * np.cos(om * ks))) / (2 * np.pi)
        for om in OMEGA_GRID
    ])
    return fhat


def spectrum_E0(v, T, c0=C0):
    m0  = max(1, int(np.floor(c0 * T**(1/3))))
    gam = np.array([np.dot(v[:T-k], v[k:]) / T for k in range(m0 + 1)])
    ks  = np.arange(1, m0 + 1)
    fhat = np.array([
        (gam[0] + 2 * np.sum(gam[1:] * np.cos(om * ks))) / (2 * np.pi)
        for om in OMEGA_GRID
    ])
    return fhat


def spectrum_E3h(v, uhat, T):
    mt, m_min, m_max = compute_state_mt(uhat, T, m_min=2, m_max=None, kappa=1.0, eps=1e-8)
    gam = np.zeros(m_max + 1)
    gam[0] = float(np.dot(v, v) / T)
    for k in range(1, m_max + 1):
        valid_t = np.where(mt >= k)[0]
        tk      = valid_t - k
        in_rng  = (tk >= 0) & (tk < T)
        valid_t = valid_t[in_rng]
        tk      = tk[in_rng]
        if len(valid_t) == 0:
            continue
        gam[k] = float(np.dot(v[valid_t], v[tk]) / T)
    ks = np.arange(1, m_max + 1)
    fhat = np.array([
        (gam[0] + 2 * np.sum(gam[1:] * np.cos(om * ks))) / (2 * np.pi)
        for om in OMEGA_GRID
    ])
    return fhat


def spectrum_E3s(v, uhat, T):
    mt, m_min, m_max = compute_state_mt(uhat, T, m_min=2, m_max=None, kappa=1.0, eps=1e-8)
    gam = np.zeros(m_max + 1)
    gam[0] = float(np.dot(v, v) / T)
    for k in range(1, m_max + 1):
        valid_t = np.where(mt >= k)[0]
        tk      = valid_t - k
        in_rng  = (tk >= 0) & (tk < T)
        valid_t = valid_t[in_rng]
        tk      = tk[in_rng]
        if len(valid_t) == 0:
            continue
        wk = 1.0 - k / (mt[valid_t].astype(float) + 1.0)
        gam[k] = float(np.dot(v[valid_t] * wk, v[tk]) / T)
    ks = np.arange(1, m_max + 1)
    fhat = np.array([
        (gam[0] + 2 * np.sum(gam[1:] * np.cos(om * ks))) / (2 * np.pi)
        for om in OMEGA_GRID
    ])
    return fhat


# ═══════════════════════════════════════════════════════════════════════════════
# 7. MONTE CARLO ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_monte_carlo():
    results = {}

    for dgp_name in ["DGP1", "DGP2"]:
        simulate_dgp = simulate_DGP1 if dgp_name == "DGP1" else simulate_DGP2
        dgp_id       = 1 if dgp_name == "DGP1" else 2
        results[dgp_name] = {}

        for T in T_LIST:
            R = R_DICT[T]
            print(f"\n{'='*60}")
            print(f"  {dgp_name}  |  T={T}  |  R={R}")
            print(f"{'='*60}")

            store = {est: {
                "lam_min":   [],
                "neg_eig":   [],
                "fmin":      [],
                "neg_spec":  [],
                "cov95":     [],
                "size5":     [],
                "power_loc": [],
                "inadm_se":  [],
                "cov95_psd": [],
                "size5_psd": [],
                "asym":      [],
                "_lam_min":  [],
                "_fmin":     [],
                "neg_eig_mass": [],
                "dist_to_psd":  [],
                "f0":           [],
                "fmin_over_f0": [],
                "spec_neg_mass":[],
                "neg_spec_flag":[],
                "neg_eig_flag": [],
                "size5_flag":   [],
                "cov95_flag":   [],
                "power_flag":   [],
            } for est in EST_LIST}

            for r in range(R):
                if (r + 1) % 500 == 0:
                    print(f"    rep {r+1}/{R}")

                seed_u = SEED_BASE + 10_000 * dgp_id + 100 * (T_LIST.index(T)+1) + r
                seed_x = seed_u + 50_000

                u = simulate_dgp(T, seed_u)
                X = simulate_regressors(T, seed_x)
                y = X @ BETA_TRUE + u

                beta_hat, uhat = ols(y, X)
                scores = X * uhat[:, None]

                delta = 1.0
                y_alt = X @ np.array([1.0, delta/np.sqrt(T), 0.0]) + u
                bhat_alt, uhat_alt = ols(y_alt, X)
                scores_alt = X * uhat_alt[:, None]

                estimators = {
                    "E0":  (HAC_E0(scores, T),        HAC_E0(scores_alt, T)),
                    "E1":  (HAC_E1(scores, T),        HAC_E1(scores_alt, T)),
                    "E2":  (HAC_E2(scores, T),        HAC_E2(scores_alt, T)),
                    "E3h": (HAC_E3h(scores, uhat, T), HAC_E3h(scores_alt, uhat_alt, T)),
                    "E3s": (HAC_E3s(scores, uhat, T), HAC_E3s(scores_alt, uhat_alt, T)),
                }

                for est_name, (Om, Om_alt) in estimators.items():
                    s = store[est_name]

                    Om     = symmetrize(Om)
                    Om_alt = symmetrize(Om_alt)
                    s["asym"].append(max_asymmetry(Om))

                    vals = np.linalg.eigvalsh(Om)
                    lmin = float(vals.min())
                    neg_mass_eig = float(np.sum(np.maximum(-vals, 0.0)))
                    s["neg_eig_mass"].append(neg_mass_eig)
                    s["neg_eig_flag"].append(int(lmin < 0))
                    s["lam_min"].append(lmin)
                    s["neg_eig"].append(int(lmin < 0))
                    s["_lam_min"].append(lmin)

                    if est_name == "E0":
                        fhat = spectrum_E0(u, T)
                    elif est_name == "E1":
                        fhat = spectrum_E1(u, T)
                    elif est_name == "E2":
                        fhat = spectrum_E2(u, T)
                    elif est_name == "E3h":
                        fhat = spectrum_E3h(u, uhat, T)
                    else:
                        fhat = spectrum_E3s(u, uhat, T)

                    fmin = float(fhat.min())
                    s["fmin"].append(fmin)
                    s["neg_spec"].append(int(fmin < 0))
                    s["_fmin"].append(fmin)

                    f0 = float(fhat[0])
                    s["f0"].append(f0)
                    s["fmin_over_f0"].append(float(fmin / max(f0, 1e-30)))
                    s["spec_neg_mass"].append(
                        float(_trapezoid(np.maximum(-fhat, 0.0), OMEGA_GRID))
                    )
                    s["neg_spec_flag"].append(int(fmin < 0))

                    Vb = sandwich_var(X, Om, T)
                    diag_ok = (Vb[1, 1] > 0)
                    s["inadm_se"].append(int(not diag_ok))

                    tstat = beta_hat[1] / (np.sqrt(max(Vb[1, 1], 1e-30)))
                    s["size5"].append(int(abs(tstat) > 1.96))

                    ci_lo = beta_hat[1] - 1.96 * np.sqrt(max(Vb[1, 1], 0.0))
                    ci_hi = beta_hat[1] + 1.96 * np.sqrt(max(Vb[1, 1], 0.0))
                    s["cov95"].append(int(ci_lo <= 0.0 <= ci_hi))

                    Vb_alt = sandwich_var(X, Om_alt, T)
                    ts_alt = bhat_alt[1] / (np.sqrt(max(Vb_alt[1, 1], 1e-30)))
                    s["power_loc"].append(int(abs(ts_alt) > 1.96))

                    s["size5_flag"].append(s["size5"][-1])
                    s["cov95_flag"].append(s["cov95"][-1])
                    s["power_flag"].append(s["power_loc"][-1])

                    Om_psd = PSD_project(Om)
                    Vb_psd = sandwich_var(X, Om_psd, T)
                    tstat_psd = beta_hat[1] / (np.sqrt(max(Vb_psd[1, 1], 1e-30)))
                    s["size5_psd"].append(int(abs(tstat_psd) > 1.96))
                    ci_lo_p = beta_hat[1] - 1.96 * np.sqrt(max(Vb_psd[1, 1], 0.0))
                    ci_hi_p = beta_hat[1] + 1.96 * np.sqrt(max(Vb_psd[1, 1], 0.0))
                    s["cov95_psd"].append(int(ci_lo_p <= 0.0 <= ci_hi_p))
                    dist_psd = float(np.linalg.norm(Om - Om_psd, ord="fro"))
                    s["dist_to_psd"].append(dist_psd)

            summary = {}

            def cond_mean(arr, mask):
                return float(np.mean(arr[mask])) if mask.any() else np.nan

            for est_name in EST_LIST:
                sv = store[est_name]

                neg_spec = np.array(sv["neg_spec_flag"], dtype=bool)
                neg_eig  = np.array(sv["neg_eig_flag"],  dtype=bool)
                size_arr = np.array(sv["size5_flag"],  dtype=float)
                cov_arr  = np.array(sv["cov95_flag"],  dtype=float)
                pow_arr  = np.array(sv["power_flag"],  dtype=float)

                summary[est_name] = {
                    "Pr(lmin<0)":      float(np.mean(sv["neg_eig"])),
                    "E[lmin]":         float(np.mean(sv["lam_min"])),
                    "Pr(fmin<0)":      float(np.mean(sv["neg_spec"])),
                    "E[fmin]":         float(np.mean(sv["fmin"])),
                    "Coverage95":      float(np.mean(sv["cov95"])),
                    "Size5":           float(np.mean(sv["size5"])),
                    "Power(local)":    float(np.mean(sv["power_loc"])),
                    "Inadm_SE_rate":   float(np.mean(sv["inadm_se"])),
                    "Coverage95_PSD":  float(np.mean(sv["cov95_psd"])),
                    "Size5_PSD":       float(np.mean(sv["size5_psd"])),
                    "Asym_maxmean":    float(np.mean(sv["asym"])),
                    "_lam_min":        np.array(sv["_lam_min"], dtype=float),
                    "_fmin":           np.array(sv["_fmin"],    dtype=float),
                    "N_neg_spec":      int(np.sum(neg_spec)),
                    "N_neg_eig":       int(np.sum(neg_eig)),
                    "Size5|neg_spec":  cond_mean(size_arr, neg_spec),
                    "Cov95|neg_spec":  cond_mean(cov_arr,  neg_spec),
                    "Power|neg_spec":  cond_mean(pow_arr,  neg_spec),
                    "Size5|neg_eig":   cond_mean(size_arr, neg_eig),
                    "Cov95|neg_eig":   cond_mean(cov_arr,  neg_eig),
                    "Power|neg_eig":   cond_mean(pow_arr,  neg_eig),
                    "E[neg_eig_mass]": float(np.mean(sv["neg_eig_mass"])),
                    "E[dist_to_psd]":  float(np.mean(sv["dist_to_psd"])),
                    "E[f0]":           float(np.mean(sv["f0"])),
                    "E[fmin/f0]":      float(np.mean(sv["fmin_over_f0"])),
                    "E[spec_neg_mass]":float(np.mean(sv["spec_neg_mass"])),
                }

            results[dgp_name][T] = summary

            print(f"\n  {'Estimator':<6} {'Pr(lmin<0)':>12} {'E[lmin]':>10} "
                  f"{'Pr(f<0)':>10} {'Cov95':>8} {'Size5':>8} {'Size5_PSD':>10} {'Asym~0':>8}")
            print("  " + "-" * 86)
            for est_name in EST_LIST:
                sv = summary[est_name]
                print(f"  {est_name:<6} {sv['Pr(lmin<0)']:>12.4f} {sv['E[lmin]']:>10.4f} "
                      f"{sv['Pr(fmin<0)']:>10.4f} {sv['Coverage95']:>8.4f} "
                      f"{sv['Size5']:>8.4f} {sv['Size5_PSD']:>10.4f} {sv['Asym_maxmean']:>8.2e}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 8. PARAMETER MAP FOR H_{alpha,beta}
# ═══════════════════════════════════════════════════════════════════════════════

def w_weight(a):
    a = np.asarray(a, dtype=float)
    z = a / _2PI
    return (1.0 / (8.0 * _PI)) * np.log(np.maximum(z, 1.0 + 1e-16)) * (z ** 1.5)


def gauss_legendre_integrate(f, a0, a1, n=64):
    xs, ws = np.polynomial.legendre.leggauss(n)
    mid  = 0.5 * (a0 + a1)
    half = 0.5 * (a1 - a0)
    a    = mid + half * xs
    return half * np.sum(ws * f(a))


def precompute_I012_on_xgrid(x_grid, tol_a=1e-14, n_quad=64):
    I0 = np.zeros_like(x_grid, dtype=float)
    I1 = np.zeros_like(x_grid, dtype=float)
    I2 = np.zeros_like(x_grid, dtype=float)

    for i, x in enumerate(x_grid):
        cx   = np.cosh(x)
        a0   = _2PI * np.exp(abs(x))
        tail = max(5.0, np.log(1.0 / tol_a) / max(cx, 1.0))
        a1   = a0 + tail

        def base(a):
            return w_weight(a) * np.exp(-a * cx)

        I0[i] = gauss_legendre_integrate(lambda a: base(a),        a0, a1, n=n_quad)
        I1[i] = gauss_legendre_integrate(lambda a: a * base(a),    a0, a1, n=n_quad)
        I2[i] = gauss_legendre_integrate(lambda a: (a**2)*base(a), a0, a1, n=n_quad)

    return I0, I1, I2


def W_approx_paper(alpha, beta, t_grid,
                   x_max=3.0, n_x=260,
                   tol_a=1e-14, n_quad=64):
    x_grid = np.linspace(0.0, x_max, n_x)
    dx = x_grid[1] - x_grid[0]

    wx = np.ones_like(x_grid)
    wx[0]  = 0.5
    wx[-1] = 0.5
    wx *= dx

    I0, I1, I2 = precompute_I012_on_xgrid(x_grid, tol_a=tol_a, n_quad=n_quad)
    cosh_x = np.cosh(x_grid)

    Hx = I2 + beta * I0 - alpha * cosh_x * I1

    t_grid = np.asarray(t_grid, dtype=float)
    C      = np.cos(np.outer(t_grid, x_grid))
    W_vals = 2.0 * (C @ (Hx * wx))

    meta = {
        "x_grid": x_grid,
        "wx": wx,
        "I0": I0, "I1": I1, "I2": I2,
        "cosh_x": cosh_x,
        "x_max": x_max, "n_x": n_x,
        "tol_a": tol_a, "n_quad": n_quad,
    }
    return W_vals, meta


def run_parameter_map(N_alpha=30, N_beta=30, t_max=30.0, dt=0.1,
                      x_max=3.0, n_x=220,
                      tol_a=1e-14, n_quad=48,
                      eps_scr=1e-10, K_cert=10):
    print("\n" + "="*60)
    print("  PARAMETER MAP FOR H_{alpha,beta}  (paper definitions)")
    print("="*60)

    alpha_grid = np.linspace(0.05, 2*np.pi - 0.05, N_alpha)
    beta_grid  = np.logspace(-1, 1, N_beta)
    t_grid     = np.arange(0.0, t_max + 1e-12, dt)

    screen_map = np.full((N_alpha, N_beta), np.inf, dtype=float)
    tstar_map  = np.zeros((N_alpha, N_beta), dtype=float)

    for i, alpha in enumerate(alpha_grid):
        for j, beta in enumerate(beta_grid):
            W_vals, _ = W_approx_paper(alpha, beta, t_grid,
                                       x_max=x_max, n_x=n_x,
                                       tol_a=tol_a, n_quad=n_quad)
            mmin  = float(np.min(W_vals))
            tstar = float(t_grid[int(np.argmin(W_vals))])
            screen_map[i, j] = mmin
            tstar_map[i, j]  = tstar

    neg_idx = list(zip(*np.where(screen_map < -eps_scr)))
    refined_pts = []   # NOT certified; refined stability check only
    if len(neg_idx) > 0:
        sorted_idx = sorted(neg_idx, key=lambda ij: screen_map[ij[0], ij[1]])
        pick = sorted_idx[:min(K_cert, len(sorted_idx))]

        x_max2  = max(x_max, 3.5)
        n_x2    = int(np.ceil(1.5 * n_x))
        n_quad2 = max(n_quad, 64)
        tol_a2  = min(tol_a, 1e-16)

        for (i, j) in pick:
            alpha = float(alpha_grid[i])
            beta  = float(beta_grid[j])
            tstar = float(tstar_map[i, j])

            t_fine = np.linspace(max(0.0, tstar - 1.0), tstar + 1.0, 801)
            W_coarse, _ = W_approx_paper(alpha, beta, t_fine,
                                          x_max=x_max, n_x=n_x,
                                          tol_a=tol_a, n_quad=n_quad)
            W_ref, _ = W_approx_paper(alpha, beta, t_fine,
                                       x_max=x_max2, n_x=n_x2,
                                       tol_a=tol_a2, n_quad=n_quad2)

            m1 = float(np.min(W_coarse))
            m2 = float(np.min(W_ref))
            err_est = abs(m2 - m1) + 1e-8
            stable_neg = (m2 + err_est < 0.0)   # stable under grid refinement, NOT certified

            refined_pts.append((alpha, beta, m2, bool(stable_neg), float(err_est)))

        alpha_ref, beta_ref = 6.0, 9.0
        t_ref  = np.arange(0.0, t_max + 1e-12, dt)
        W_scr, _ = W_approx_paper(alpha_ref, beta_ref, t_ref,
                                   x_max=x_max, n_x=n_x,
                                   tol_a=tol_a, n_quad=n_quad)
        tstar_ref = float(t_ref[int(np.argmin(W_scr))])

        t_fine = np.linspace(max(0.0, tstar_ref - 1.0), tstar_ref + 1.0, 801)
        W1, _ = W_approx_paper(alpha_ref, beta_ref, t_fine,
                                x_max=x_max, n_x=n_x,
                                tol_a=tol_a, n_quad=n_quad)
        x_max2  = max(x_max, 3.5)
        n_x2    = int(np.ceil(1.5 * n_x))
        n_quad2 = max(n_quad, 64)
        tol_a2  = min(tol_a, 1e-16)
        W2, _ = W_approx_paper(alpha_ref, beta_ref, t_fine,
                                x_max=x_max2, n_x=n_x2,
                                tol_a=tol_a2, n_quad=n_quad2)
        m1 = float(np.min(W1))
        m2 = float(np.min(W2))
        err_est = abs(m2 - m1) + 1e-8
        stable_ref = (m2 + err_est < 0.0)
        refined_pts.append((alpha_ref, beta_ref, m2, bool(stable_ref), float(err_est)))

        n_stable = sum(int(p[3]) for p in refined_pts)
        print(f"  Refined stability checks: {n_stable} / {len(refined_pts)} stable negatives "
              f"(NOT certified; certified points reported in Appendix A)")

    return alpha_grid, beta_grid, screen_map, refined_pts


# ═══════════════════════════════════════════════════════════════════════════════
# 9. FIGURES AND TABLES (MC.1 – MC.5)
# ═══════════════════════════════════════════════════════════════════════════════

COLORS = {"E0": "#9467bd", "E1": "#1f77b4", "E2": "#ff7f0e",
          "E3h": "#2ca02c", "E3s": "#d62728"}


def make_table_MC1(results):
    rows = []
    for dgp in ["DGP1", "DGP2"]:
        for T in T_LIST:
            for est in EST_LIST:
                sv = results[dgp][T][est]
                rows.append({
                    "DGP": dgp, "T": T, "R": R_DICT[T], "Estimator": est,
                    "Pr(fmin<0)":       sv["Pr(fmin<0)"],
                    "E[fmin]":          sv["E[fmin]"],
                    "Pr(lmin<0)":       sv["Pr(lmin<0)"],
                    "E[lmin]":          sv["E[lmin]"],
                    "Asym_maxmean":     sv["Asym_maxmean"],
                    "E[neg_eig_mass]":  sv["E[neg_eig_mass]"],
                    "E[dist_to_psd]":   sv["E[dist_to_psd]"],
                    "E[f0]":            sv["E[f0]"],
                    "E[fmin/f0]":       sv["E[fmin/f0]"],
                    "E[spec_neg_mass]": sv["E[spec_neg_mass]"],
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "Table_MC1.csv", index=False)
    print("\nTable MC.1 saved.")
    return df


def make_table_MC2(results):
    rows = []
    for dgp in ["DGP1", "DGP2"]:
        for T in T_LIST:
            for est in EST_LIST:
                sv = results[dgp][T][est]
                rows.append({
                    "DGP": dgp, "T": T, "R": R_DICT[T], "Estimator": est,
                    "Coverage95":     sv["Coverage95"],
                    "Size5":          sv["Size5"],
                    "Power(local)":   sv["Power(local)"],
                    "Inadm_SE_rate":  sv["Inadm_SE_rate"],
                    "N_neg_spec":     sv["N_neg_spec"],
                    "N_neg_eig":      sv["N_neg_eig"],
                    "Size5|neg_spec": sv["Size5|neg_spec"],
                    "Cov95|neg_spec": sv["Cov95|neg_spec"],
                    "Size5|neg_eig":  sv["Size5|neg_eig"],
                    "Cov95|neg_eig":  sv["Cov95|neg_eig"],
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "Table_MC2.csv", index=False)
    print("Table MC.2 saved.")
    return df


def make_table_MC3(results):
    rows = []
    for dgp in ["DGP1", "DGP2"]:
        for T in T_LIST:
            for est in EST_LIST:
                sv = results[dgp][T][est]
                rows.append({
                    "DGP": dgp, "T": T, "R": R_DICT[T], "Estimator": est,
                    "Size5":          sv["Size5"],
                    "Size5_PSD":      sv["Size5_PSD"],
                    "DeltaSize":      sv["Size5_PSD"] - sv["Size5"],
                    "Coverage95":     sv["Coverage95"],
                    "Coverage95_PSD": sv["Coverage95_PSD"],
                    "DeltaCov":       sv["Coverage95_PSD"] - sv["Coverage95"],
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "Table_MC3.csv", index=False)
    print("Table MC.3 saved.")
    return df


def fig_MC1_negativity_frequency(results):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)
    for ax, dgp in zip(axes, ["DGP1", "DGP2"]):
        x = np.arange(len(T_LIST))
        width = 0.20
        for k, est in enumerate(EST_LIST):
            vals = [results[dgp][T][est]["Pr(fmin<0)"] for T in T_LIST]
            ax.bar(x + k*width, vals, width, label=est, color=COLORS[est], alpha=0.85)
        ax.set_xticks(x + 1.5*width)
        ax.set_xticklabels([f"T={T}" for T in T_LIST])
        ax.set_ylabel("Pr(min spectral estimate < 0)")
        ax.set_title(f"{dgp}")
        ax.legend()
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_ylim(bottom=0)
    fig.suptitle("Figure MC.1 – Frequency of Negative Spectral Estimates\n"
                 "by Estimator, DGP, and Sample Size T", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "Figure_MC1.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Figure_MC1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Figure MC.1 saved.")


def fig_MC2_severity(results, T_plot=None):
    if T_plot is None:
        T_plot = T_LIST[-1]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    titles_spec = ["DGP1 – Min Spectrum", "DGP2 – Min Spectrum"]
    titles_eig  = ["DGP1 – Min Eigenvalue", "DGP2 – Min Eigenvalue"]
    for col, dgp in enumerate(["DGP1", "DGP2"]):
        for est in EST_LIST:
            sv = results[dgp][T_plot][est]
            axes[0, col].hist(sv["_fmin"], bins=60, alpha=0.5, label=est,
                              color=COLORS[est], density=True)
            axes[1, col].hist(sv["_lam_min"], bins=60, alpha=0.5, label=est,
                              color=COLORS[est], density=True)
        for row in range(2):
            axes[row, col].axvline(0, color="red", linewidth=1.5, linestyle="--", label="0")
            axes[row, col].legend(fontsize=8)
        axes[0, col].set_xlabel("min_ω f̂(ω)"); axes[0, col].set_title(titles_spec[col])
        axes[1, col].set_xlabel("λ_min(Ω̂)");   axes[1, col].set_title(titles_eig[col])
    fig.suptitle(f"Figure MC.2 – Severity Distributions (T={T_plot})", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / f"Figure_MC2_T{T_plot}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"Figure_MC2_T{T_plot}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure MC.2 (T={T_plot}) saved.")


def fig_MC3_inference(results):
    metrics = ["Coverage95", "Size5", "Power(local)"]
    labels  = ["95% CI Coverage", "Size at 5% (H₀)", "Power (local alt, δ=1)"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for row, dgp in enumerate(["DGP1", "DGP2"]):
        for col, (met, lbl) in enumerate(zip(metrics, labels)):
            ax = axes[row, col]
            x  = np.arange(len(T_LIST))
            width = 0.18
            for k, est in enumerate(EST_LIST):
                vals = [results[dgp][T][est][met] for T in T_LIST]
                ax.bar(x + k*width, vals, width, label=est, color=COLORS[est], alpha=0.85)
            if met == "Size5":
                vals_psd = [results[dgp][T]["E3h"]["Size5_PSD"] for T in T_LIST]
                ax.plot(x + 2*width, vals_psd, "k^--", label="E3h+PSD", markersize=7)
            if met == "Coverage95":
                ax.axhline(0.95, color="black", linewidth=1.2, linestyle="--", label="Nominal 0.95")
            if met == "Size5":
                ax.axhline(0.05, color="red", linewidth=1.2, linestyle="--", label="Nominal 0.05")
            ax.set_xticks(x + 1.5*width)
            ax.set_xticklabels([f"T={T}" for T in T_LIST])
            ax.set_title(f"{dgp} – {lbl}")
            ax.legend(fontsize=7)
            ax.set_ylim(0, 1)
    fig.suptitle("Figure MC.3 – Inference Diagnostics (Coverage, Size, Power)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "Figure_MC3.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Figure_MC3.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Figure MC.3 saved.")


def fig_MC4_parameter_map(alpha_grid, beta_grid, screen_map, refined_pts):
    fig, ax = plt.subplots(figsize=(9, 7))
    vmin = float(np.min(screen_map))
    vmax = float(np.max(screen_map))
    im = ax.pcolormesh(
        beta_grid, alpha_grid, screen_map,
        cmap="RdBu_r",
        norm=mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=max(1e-6, vmax))
    )
    plt.colorbar(im, ax=ax, label="min_t W_{α,β}(t)  (screened)")
    ax.contour(beta_grid, alpha_grid, screen_map, levels=[0],
               colors="black", linewidths=1.5)

    for alpha, beta, Wmin_ref, stable_neg, err_est in refined_pts:
        marker = "o" if stable_neg else "x"
        color  = "lime" if stable_neg else "yellow"
        ax.scatter(beta, alpha, c=color, s=80, marker=marker, zorder=5,
                   edgecolors="black", linewidths=0.5)

    ax.set_xscale("log")
    ax.set_xlabel("β  (log scale)")
    ax.set_ylabel("α")
    ax.set_title(
        "Figure MC.4 – Parameter Map for Sign Changes in W_{α,β}(t)\n"
        "Heatmap: screened minimum; ● = refined-stable negative (NOT certified); "
        "× = unstable under refinement"
    )
    fig.tight_layout()
    fig.savefig(OUT / "Figure_MC4.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Figure_MC4.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Figure MC.4 saved.")


def fig_MC5_certified_curves(refined_pts):
    """Figure MC.5 – Representative W_{α,β}(t) curves (illustrative refined check, NOT certified)."""
    if not refined_pts:
        print("Figure MC.5 skipped (no refined candidates).")
        return

    t_plot = np.linspace(0, 30, 800)
    n_show = min(6, len(refined_pts))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    for idx in range(n_show):
        alpha, beta, Wmin_ref, stable_neg, err_est = refined_pts[idx]
        Wvals, _ = W_approx_paper(alpha, beta, t_plot,
                                   x_max=3.5, n_x=260,
                                   tol_a=1e-16, n_quad=64)
        ax = axes[idx]
        ax.plot(t_plot, Wvals, color="steelblue", linewidth=1.5)
        ax.axhline(0, color="red", linewidth=1.2, linestyle="--")
        ax.set_xlabel("t"); ax.set_ylabel("W_{α,β}(t)")
        # Fix 9: no "robust" language; use "stable-neg" / "unstable"
        panel_label = "stable-neg (refined check)" if stable_neg else "unstable under refinement"
        ax.set_title(f"α={alpha:.3f}, β={beta:.4f}\nmin≈{Wmin_ref:.2e}  ({panel_label})")

    for ax in axes[n_show:]:
        ax.axis("off")

    fig.suptitle("Figure MC.5 – Representative W_{α,β}(t) Curves\n"
                 "(illustrative refined check, not certified)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "Figure_MC5_refinedcheck.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Figure_MC5_refinedcheck.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Figure MC.5 (Figure_MC5_refinedcheck — illustrative only, not certified) saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. PAPER ANALYTICS: APPENDIX-A CERTIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def w_prime(a):
    a = np.asarray(a, dtype=float)
    z = a / _2PI
    return (1.0 / (16.0 * _PI**2)) * np.sqrt(np.maximum(z, 1e-300)) * (1.0 + 1.5 * np.log(np.maximum(z, 1e-300)))


def rho_ab(u, alpha, beta):
    u = np.asarray(u, dtype=float)
    term = (4.0 * _PI**2 - alpha * _PI) * np.exp(2.0 * u) + (beta - alpha * _PI)
    return 0.25 * u * np.exp(2.5 * u) * term * np.exp(-_PI * (np.exp(2.0 * u) + 1.0))


def P_ab(a, x, alpha, beta):
    return a**2 + beta - alpha * a * np.cosh(x)


def F_ab(a, x, alpha, beta):
    return w_weight(a) * P_ab(a, x, alpha, beta) * np.exp(-a * np.cosh(x))


def Fx_ab(a, x, alpha, beta):
    sx = np.sinh(x)
    cx = np.cosh(x)
    P  = P_ab(a, x, alpha, beta)
    return -a * sx * w_weight(a) * (alpha + P) * np.exp(-a * cx)


def Fa_ab(a, x, alpha, beta):
    cx = np.cosh(x)
    P  = P_ab(a, x, alpha, beta)
    wa = w_weight(a)
    wpa = w_prime(a)
    dP_da = 2.0 * a - alpha * cx
    return np.exp(-a * cx) * (wpa * P + wa * dP_da - cx * wa * P)


def compute_M0_M1_M2_Bpartial(alpha, beta, X, A, n_quad_a=512, n_u_grid=20001):
    def integrand_M0(a):
        return w_weight(a) * (a**2 + beta + alpha * a * np.cosh(X)) * np.exp(-a)
    M0 = float(gauss_legendre_integrate(integrand_M0, _2PI, A, n=n_quad_a))

    u_grid = np.linspace(0.0, X, n_u_grid)
    rho_vals = rho_ab(u_grid, alpha, beta)
    rho_max  = float(np.max(rho_vals))

    def integrand_M1(a):
        return w_weight(a) * a * (a**2 + beta + alpha * a * np.cosh(X) + alpha) * np.exp(-a)
    I_M1 = float(gauss_legendre_integrate(integrand_M1, _2PI, A, n=n_quad_a))
    M1 = rho_max + I_M1

    a_b = _2PI * np.exp(u_grid)
    Fx = Fx_ab(a_b, u_grid, alpha, beta)
    Fa = Fa_ab(a_b, u_grid, alpha, beta)
    expr_B = rho_vals + 2.0 * a_b * np.abs(Fx) + (a_b**2) * np.abs(Fa)
    Bpartial = float(np.max(expr_B))

    def integrand_M2(a):
        inner = (a + a**2) * (a**2 + beta + alpha * a * np.cosh(X) + alpha) + alpha * a**2
        return w_weight(a) * inner * np.exp(-a)
    I_M2 = float(gauss_legendre_integrate(integrand_M2, _2PI, A, n=n_quad_a))
    M2 = Bpartial + I_M2

    return M0, M1, M2, Bpartial


def eps_trap_L43(alpha, beta, t, X, A, N, n_quad_a=512, n_u_grid=20001):
    h = X / N
    M0, M1, M2, Bp = compute_M0_M1_M2_Bpartial(alpha, beta, X, A, n_quad_a=n_quad_a, n_u_grid=n_u_grid)
    C2 = M2 + 2.0 * abs(t) * M1 + (t**2) * M0
    return float((X * h**2 / 6.0) * C2), dict(M0=M0, M1=M1, M2=M2, Bpartial=Bp, C2=C2)


def C_ab(alpha, beta):
    _2pi = 2.0 * np.pi
    return (1.0 / (8.0 * np.pi)) * (_2pi ** (-1.5)) * (1.0 + alpha / _2pi + beta / (_2pi**2))


def eps_A_majorant(X, A, alpha, beta):
    C = C_ab(alpha, beta)
    poly = A**4 + 4*A**3 + 12*A**2 + 24*A + 24
    return float(2.0 * X * C * np.exp(-A) * poly)


def eps_X_majorant(X, alpha, beta):
    C = C_ab(alpha, beta)
    pi = np.pi
    eX = np.exp(X)
    bracket = ((4*pi)**4 / 8.0) * eX + (4*pi)**3 + 6*(4*pi)**2 * np.exp(-X) + 24*(4*pi) * np.exp(-2*X) + 48*np.exp(-3*X)
    return float((C / pi) * np.exp(-pi * np.exp(2*X)) * bracket)


def _cache_key_I012A(X, A, N, n_quad):
    def f(x): return str(x).replace(".", "p")
    return f"I012A_X{f(X)}_A{f(A)}_N{N}_q{n_quad}.npz"


def load_or_compute_I012A_bases(X, A, N, n_quad=64, block=2048):
    path = (CACHE / _cache_key_I012A(X, A, N, n_quad)).resolve()
    if path.exists():
        z = np.load(path)
        return z["x_grid"], z["cosh_x"], z["I0"], z["I1"], z["I2"]

    print(f"\n  [cache miss] computing bases I0/I1/I2 with n_quad={n_quad} ...")
    x_grid = np.linspace(0.0, X, N + 1, dtype=float)
    cosh_x = np.cosh(x_grid).astype(float)

    I0 = np.zeros(N + 1, dtype=float)
    I1 = np.zeros(N + 1, dtype=float)
    I2 = np.zeros(N + 1, dtype=float)

    xs, ws = np.polynomial.legendre.leggauss(n_quad)
    xs = xs.astype(float); ws = ws.astype(float)

    for start in range(0, N + 1, block):
        end = min(N + 1, start + block)
        xb = x_grid[start:end]
        cb = cosh_x[start:end]

        a0 = _2PI * np.exp(xb)
        mask = (a0 < A)
        if not np.any(mask):
            continue

        a0m = a0[mask]
        cbm = cb[mask]

        mid  = 0.5 * (A + a0m)
        half = 0.5 * (A - a0m)

        a = mid[:, None] + half[:, None] * xs[None, :]
        w = half[:, None] * ws[None, :]

        exp_term = np.exp(-a * cbm[:, None])
        base = w_weight(a) * exp_term * w

        I0b = np.sum(base, axis=1)
        I1b = np.sum(base * a, axis=1)
        I2b = np.sum(base * (a**2), axis=1)

        I0[start:end][mask] = I0b
        I1[start:end][mask] = I1b
        I2[start:end][mask] = I2b

        if (start // block) % 10 == 0:
            print(f"    block {start}:{end} / {N+1}")

    np.savez_compressed(path, x_grid=x_grid, cosh_x=cosh_x, I0=I0, I1=I1, I2=I2)
    print(f"  [saved] {path}")
    return x_grid, cosh_x, I0, I1, I2


def W_hat_AXN(alpha, beta, t, X, N, x_grid, cosh_x, I0, I1, I2):
    r"""
    Compute \hat W_{A,X,N}(t) using the Lemma 4.3 trapezoid rule:
      \hat W = 2h[ 1/2 H^A(0) + sum_{j=1}^{N-1} H^A(jh) cos(t jh)
                 + 1/2 H^A(X) cos(tX) ]
    with H^A(x) = I2(x) + beta I0(x) - alpha cosh(x) I1(x).
    """
    H = I2 + beta * I0 - alpha * cosh_x * I1
    h = X / N
    c = np.cos(t * x_grid)
    return float(2.0 * h * (0.5 * H[0] + np.dot(H[1:-1], c[1:-1]) + 0.5 * H[-1] * c[-1]))


def W_hat_AXN_tgrid(alpha, beta, t_arr, X, N, x_grid, cosh_x, I0, I1, I2):
    r"""
    Vectorised version: compute \hat W_{A,X,N}(t) for an array of t values.
    Returns array of shape (len(t_arr),).
    """
    H = I2 + beta * I0 - alpha * cosh_x * I1   # (N+1,)
    h = X / N
    # C[i,j] = cos(t_arr[i] * x_grid[j])
    C = np.cos(np.outer(t_arr, x_grid))          # (Nt, N+1)
    # trapezoid weights
    wt = np.ones(N + 1)
    wt[0]  = 0.5
    wt[-1] = 0.5
    wt *= h
    return 2.0 * (C @ (H * wt))                  # (Nt,)


def eps_inner_from_bases(alpha, beta, X, cosh_x, I0c, I1c, I2c, I0f, I1f, I2f):
    dH = (I2f - I2c) + beta * (I0f - I0c) - alpha * cosh_x * (I1f - I1c)
    return float(2.0 * X * np.max(np.abs(dH)))


# ─── PATCH 1 & PATCH 2: certify_point_appendixA_fast ────────────────────────
def certify_point_appendixA_fast(alpha, beta, t0_init,
                                 X, A, N,
                                 x_grid, cosh_x,
                                 I0c, I1c, I2c,
                                 I0f, I1f, I2f,
                                 n_quad_a=512, n_u_grid=20001):
    """
    Certify the sign of W_{α,β}(t0) following Appendix A / Lemma 4.3.

    PATCH 1: t0 is fixed to t0_init (no scan).  The caller is responsible for
             passing the correct anchor value (e.g. 19.8798 for group A, or the
             screened argmin for groups B/C/D).

    PATCH 2: Correct verdict logic distinguishing NEGATIVE CERTIFIED,
             POSITIVE CERTIFIED, and INCONCLUSIVE.
    """
    # ── PATCH 1: fix t0, no scan ──────────────────────────────────────────────
    t0    = float(t0_init)
    W_hat = W_hat_AXN(alpha, beta, t0, X, N, x_grid, cosh_x, I0f, I1f, I2f)

    # Truncation bounds (Lemma 4.1)
    epsA = eps_A_majorant(X, A, alpha, beta)
    epsX = eps_X_majorant(X, alpha, beta)

    # Trapezoid bound (Lemma 4.3)
    epsT, metaT = eps_trap_L43(alpha, beta, t0, X, A, N,
                                n_quad_a=n_quad_a, n_u_grid=n_u_grid)

    # Inner quadrature discrepancy (explicit, deterministic)
    epsIn = eps_inner_from_bases(alpha, beta, X, cosh_x,
                                 I0c, I1c, I2c, I0f, I1f, I2f)

    epsTot = epsA + epsX + epsT + epsIn

    # ── PATCH 2: correct verdict + Fix 1: certified/sign consistency ────────
    neg_cert = bool(W_hat + epsTot < 0.0)
    pos_cert = bool(W_hat - epsTot > 0.0)
    certified = bool(neg_cert or pos_cert)
    sign      = "NEG" if neg_cert else ("POS" if pos_cert else "NA")

    if neg_cert:
        verdict = "NEGATIVE CERTIFIED"
    elif pos_cert:
        verdict = "POSITIVE CERTIFIED"
    else:
        verdict = "INCONCLUSIVE"

    return dict(
        alpha=alpha, beta=beta, t0=t0,
        W_hat=W_hat,
        eps_total=epsTot,
        eps_A=epsA, eps_X=epsX, eps_trap=epsT, eps_inner=epsIn,
        M0=metaT["M0"], M1=metaT["M1"], M2=metaT["M2"], Bpartial=metaT["Bpartial"], C2=metaT["C2"],
        neg_cert=neg_cert, pos_cert=pos_cert,
        certified=certified,
        sign=sign,
        verdict=verdict,
    )


# ─── PATCH 3 & PATCH 4 driver ────────────────────────────────────────────────
def run_multipoint_certification_appendixA(screen_map, alpha_grid, beta_grid,
                                           X=3.0, A=200.0, N=131072):
    """
    Sparse multi-point certification driver.

    Groups:
      A: (6,9) anchor + 2 nearby  — uses fixed t0=19.8798 (Appendix A)
      B: 2 strongest negatives from screening
      C: 2 boundary points (neighbourhood changes sign)
      D: 2 positive controls  [PATCH 3: require pos_cert==True]

    PATCH 3: positive controls selected from screen_map > 0.1 with actual
             pos_cert check; fall through to next best if pos_cert fails.
    """
    # Precompute bases once (coarse n_quad=32, fine n_quad=64)
    x_grid, cosh_x, I0c, I1c, I2c = load_or_compute_I012A_bases(X, A, N, n_quad=32, block=2048)
    _,      _,      I0f, I1f, I2f = load_or_compute_I012A_bases(X, A, N, n_quad=64, block=2048)

    print("\n" + "="*60)
    print("  MULTI-POINT CERTIFICATION (Appendix-A compatible)")
    print("="*60)

    # Quick screener for t0
    t_coarse = np.arange(0.0, 31.0, 0.1)
    def _screen_t0(a, b):
        W, _ = W_approx_paper(a, b, t_coarse, x_max=3.0, n_x=220, tol_a=1e-14, n_quad=48)
        return float(t_coarse[int(np.argmin(W))])

    candidates = []  # (group, alpha, beta, t0_init)

    # ── Group A: anchor (6,9) + 2 nearby; use fixed t0=19.8798 ──────────────
    for a, b in [(6.0, 9.0), (5.7, 9.0), (6.0, 8.0)]:
        candidates.append(("A-near(6,9)", a, b, 19.8798))

    # ── Group B: 2 strongest negatives in screening ──────────────────────────
    flat_neg = sorted(
        [(screen_map[i, j], i, j)
         for i in range(len(alpha_grid))
         for j in range(len(beta_grid))
         if screen_map[i, j] < 0]
    )
    for _, i, j in flat_neg[:2]:
        a = float(alpha_grid[i]); b = float(beta_grid[j])
        candidates.append(("B-strong-neg", a, b, _screen_t0(a, b)))

    # ── Group C: boundary (neighbourhood has sign change) ────────────────────
    boundary = []
    for i in range(1, len(alpha_grid)-1):
        for j in range(1, len(beta_grid)-1):
            blk = screen_map[i-1:i+2, j-1:j+2]
            if blk.min() < 0 and blk.max() > 0:
                boundary.append((abs(screen_map[i, j]), i, j))
    boundary.sort()
    for _, i, j in boundary[:2]:
        a = float(alpha_grid[i]); b = float(beta_grid[j])
        candidates.append(("C-boundary", a, b, _screen_t0(a, b)))

    # ── Group D: positive controls — all 9 neighbours positive, away from boundary ──
    # Fix 11: select points where entire 3×3 neighbourhood is positive AND value > 1e-8.
    # This guarantees "strong positive, away from boundary" sanity-check points.
    pos_ctrl = []
    for i in range(1, len(alpha_grid)-1):
        for j in range(1, len(beta_grid)-1):
            blk = screen_map[i-1:i+2, j-1:j+2]
            if (blk.min() > 0.0) and (screen_map[i, j] > 1e-8):
                pos_ctrl.append((screen_map[i, j], i, j))

    pos_ctrl.sort(reverse=True)   # strongest positives first

    pos_controls_added = 0
    pos_control_candidates = []
    for _, i, j in pos_ctrl:
        a = float(alpha_grid[i]); b = float(beta_grid[j])
        t0 = _screen_t0(a, b)
        res_pre = certify_point_appendixA_fast(
            a, b, t0, X, A, N, x_grid, cosh_x,
            I0c, I1c, I2c, I0f, I1f, I2f
        )
        if res_pre["pos_cert"]:
            pos_control_candidates.append(("D-positive", a, b, t0))
            pos_controls_added += 1
            if pos_controls_added >= 2:
                break

    # Fallback: if fewer than 2 pos_cert found, add best available without pos_cert check
    if pos_controls_added < 2:
        for _, i, j in pos_ctrl:
            a = float(alpha_grid[i]); b = float(beta_grid[j])
            already = any(c[1] == a and c[2] == b for c in pos_control_candidates)
            if not already:
                pos_control_candidates.append(("D-positive(fallback)", a, b, _screen_t0(a, b)))
                if len(pos_control_candidates) >= 2:
                    break

    candidates.extend(pos_control_candidates)

    # ── Run certification for all candidates ─────────────────────────────────
    rows = []
    for group, a, b, t0 in candidates:
        print(f"\n  Certifying {group}: alpha={a:.4f}, beta={b:.4f}, t0={t0:.4f}")
        res = certify_point_appendixA_fast(
            a, b, t0, X, A, N,
            x_grid, cosh_x,
            I0c, I1c, I2c, I0f, I1f, I2f
        )
        res["group"] = group
        rows.append(res)
        print(f"    verdict={res['verdict']}  W_hat={res['W_hat']:.3e}  "
              f"eps_total={res['eps_total']:.2e}  "
              f"neg_cert={res['neg_cert']}  pos_cert={res['pos_cert']}")

    df = pd.DataFrame(rows)
    # Fix 2: stable column order for LaTeX/pgfplotstable reproducibility
    cols_ordered = [
        "group", "alpha", "beta", "t0",
        "W_hat",
        "eps_A", "eps_X", "eps_trap", "eps_inner", "eps_total",
        "neg_cert", "pos_cert", "certified", "sign", "verdict",
        "M0", "M1", "M2", "Bpartial", "C2",
    ]
    df = df.reindex(columns=[c for c in cols_ordered if c in df.columns])
    df.to_csv(OUT / "Table_certified_signchanges_APPENDIXA.csv", index=False)
    print(f"\n  Saved: outputs/Table_certified_signchanges_APPENDIXA.csv")
    n_neg = int(df["neg_cert"].sum())
    n_pos = int(df["pos_cert"].sum())
    print(f"  Certified negatives : {n_neg} / {len(df)}")
    print(f"  Certified positives : {n_pos} / {len(df)}")

    # ── Cambio 2B: inline LaTeX export for compact paper-facing Table A1 ─────
    # Opción 3.1: format W_hat and eps_total as scientific notation strings
    df_A1 = df[["group", "alpha", "beta", "t0",
                "W_hat", "eps_total", "sign", "verdict"]].copy()
    df_A1["W_hat"]     = df_A1["W_hat"].map(lambda x: f"{x:.2e}")
    df_A1["eps_total"] = df_A1["eps_total"].map(lambda x: f"{x:.2e}")

    df_A1 = df_A1.rename(columns={
        "group":     "Group",
        "alpha":     r"$\alpha$",
        "beta":      r"$\beta$",
        "t0":        r"$t_0$",
        "W_hat":     r"$\widehat{W}(t_0)$",
        "eps_total": r"$\varepsilon_{\mathrm{tot}}$",
        "sign":      r"Sign at $t_0$",
        "verdict":   "Verdict",
    })

    df_to_latex(
        df_A1,
        OUT / "Table_A1_certified.tex",
        caption=(
            r"Certified sign at $t_0$ for selected $(\alpha,\beta)$ points "
            r"(Appendix~A procedure). "
            r"Certification holds \emph{only} at $t_0$; "
            r"$\pm\varepsilon_{\mathrm{tot}}$ whisker shown in Figure~MC.6."
        ),
        label="tab:appA_certified",
        col_rename=None,   # already renamed above
        float_fmt="%.4g",
        escape=False,
        table_env=True,
    )
    return df


# ─── PATCH 4: fig_MC6 using W_hat_AXN (certified estimator) ─────────────────
def fig_MC6_certified_multipoint(cert_df,
                                 X=3.0, A=200.0, N=131072,
                                 t_half_window=3.0, n_t_plot=600):
    """
    Figure MC.6  —  Multi-point certification (certified W_hat curves).

    PATCH 4: Uses W_hat_AXN_tgrid (the certified Appendix-A estimator) instead
    of the approximate W_approx_paper.  Shows W_hat(t) around each t0 together
    with ±ε_total whiskers at t0.  Renames old illustrative figure to MC6_refinedcheck.
    """
    if cert_df is None or len(cert_df) == 0:
        print("Figure MC.6 skipped (no certification data).")
        return

    # Load (or recompute) the fine bases used for plotting
    x_grid, cosh_x, I0f, I1f, I2f = load_or_compute_I012A_bases(X, A, N, n_quad=64, block=2048)

    # Show up to 6 negative / 2 positive — use W_hat (canonical column)
    neg_rows = cert_df[cert_df["W_hat"] < 0].head(6)
    pos_rows = cert_df[cert_df["W_hat"] >= 0].head(2)
    show_rows = pd.concat([neg_rows, pos_rows], ignore_index=True)
    n_show = len(show_rows)

    if n_show == 0:
        print("Figure MC.6 skipped (no data to show).")
        return

    ncols = 3
    nrows = int(np.ceil(n_show / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows))
    axes = np.array(axes).ravel()

    for plot_idx, (_, row) in enumerate(show_rows.iterrows()):
        alpha   = float(row["alpha"])
        beta    = float(row["beta"])
        t0      = float(row["t0"])
        eps     = float(row["eps_total"])
        W_t0    = float(row["W_hat"])
        group   = str(row["group"])
        verdict = str(row["verdict"])

        # Build t-grid around t0 and compute W_hat_AXN
        t_lo   = max(0.0, t0 - t_half_window)
        t_hi   = t0 + t_half_window
        t_plot = np.linspace(t_lo, t_hi, n_t_plot)
        # ── W_hat_AXN_tgrid: MUST use certified estimator, never W_approx_paper ──
        W_plot = W_hat_AXN_tgrid(alpha, beta, t_plot, X, N, x_grid, cosh_x, I0f, I1f, I2f)

        ax = axes[plot_idx]
        ax.plot(t_plot, W_plot, color="steelblue", linewidth=1.6, label="Ŵ(t)")
        ax.axhline(0, color="red", linewidth=1.2, linestyle="--")
        ax.axvline(t0, color="gray", linewidth=0.8, linestyle=":", label=f"t₀={t0:.2f}")

        # Error whiskers at t0 as a shaded band around the point estimate
        ax.fill_between(
            [t0 - 0.05 * t_half_window, t0 + 0.05 * t_half_window],
            W_t0 - eps, W_t0 + eps,
            color="orange", alpha=0.65, label=f"±ε={eps:.1e}"
        )
        # Marker at (t0, W_hat(t0))
        color_pt = "green" if row.get("neg_cert", False) else (
                   "blue"  if row.get("pos_cert", False) else "orange")
        ax.scatter([t0], [W_t0], color=color_pt, s=60, zorder=5)

        ax.set_xlabel("t")
        ax.set_ylabel("Ŵ_{A,X,N}(t)")
        short = verdict.replace(" CERTIFIED", "✓").replace("INCONCLUSIVE", "?")
        ax.set_title(
            f"[{group}]  α={alpha:.2f}, β={beta:.3f}\n"
            f"sign certified at t₀={t0:.2f}  →  {short}\n"
            f"Ŵ(t₀)={W_t0:.2e},  ε={eps:.1e}",
            fontsize=8
        )
        ax.legend(fontsize=7)

    for ax in axes[n_show:]:
        ax.axis("off")

    # Fix 10: dynamic group list from actual data; title clarifies certification scope
    present_groups = ", ".join(sorted(cert_df["group"].unique().tolist()))
    fig.suptitle(
        "Figure MC.6 – Sign certified at t\u2080 for selected (\u03b1,\u03b2) points\n"
        "Curves: \u0174\u207f_{A,X,N}(t) (certified estimator); "
        "certification applies only at t\u2080 (\u00b1\u03b5_total whisker shown at t\u2080).\n"
        f"Groups present: {present_groups}",
        fontsize=10
    )
    fig.tight_layout()
    fig.savefig(OUT / "Figure_MC6_certified.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Figure_MC6_certified.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Figure MC.6 (certified, W_hat_AXN curves) saved.")


def make_table_MC4_E3s(results):
    """Table MC.4 – E3s comparative diagnostics."""
    rows = []
    for dgp in ["DGP1", "DGP2"]:
        for T in T_LIST:
            for est in ["E1", "E3h", "E3s"]:
                sv = results[dgp][T][est]
                rows.append({
                    "DGP": dgp, "T": T, "R": R_DICT[T], "Estimator": est,
                    "Pr(fmin<0)":      sv["Pr(fmin<0)"],
                    "E[fmin]":         sv["E[fmin]"],
                    "Pr(lmin<0)":      sv["Pr(lmin<0)"],
                    "E[lmin]":         sv["E[lmin]"],
                    "Coverage95":      sv["Coverage95"],
                    "Size5":           sv["Size5"],
                    "Size5_PSD":       sv["Size5_PSD"],
                    "Power(local)":    sv["Power(local)"],
                    "E[neg_eig_mass]": sv["E[neg_eig_mass]"],
                    "E[dist_to_psd]":  sv["E[dist_to_psd]"],
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "Table_MC4_E3s.csv", index=False)
    print("Table MC.4 (E3s) saved.")

    # ── LaTeX export (Cambio 2A + Opción 3.1) ────────────────────────────────
    col_rename_mc4 = {
        "DGP":            "DGP",
        "T":              r"$T$",
        "R":              r"$R$",
        "Estimator":      "Estimator",
        "Pr(fmin<0)":     r"$\Pr(\min_\omega \hat f < 0)$",
        "E[fmin]":        r"$\mathbb{E}[\min_\omega \hat f]$",
        "Pr(lmin<0)":     r"$\Pr(\lambda_{\min}(\hat\Omega) < 0)$",
        "E[lmin]":        r"$\mathbb{E}[\lambda_{\min}(\hat\Omega)]$",
        "Coverage95":     r"Coverage (95\%)",
        "Size5":          r"Size (5\%)",
        "Size5_PSD":      r"Size (5\%), PSD-proj.",
        "Power(local)":   r"Power ($\delta=1$)",
        "E[neg_eig_mass]":r"$\mathbb{E}[\mathrm{neg.eig.mass}]$",
        "E[dist_to_psd]": r"$\mathbb{E}[\mathrm{dist.PSD}]$",
    }
    # Opción 3.1: format ultra-small columns as "x.xxe±yy" strings before export
    df_tex = df.copy()
    for c in ["E[neg_eig_mass]", "E[dist_to_psd]"]:
        if c in df_tex.columns:
            df_tex[c] = df_tex[c].map(lambda x: f"{x:.2e}")

    df_to_latex(
        df_tex,
        OUT / "Table_MC4_E3s.tex",
        caption=(
            r"Monte Carlo diagnostics: fixed Bartlett (E1) vs.\ moving hard-cutoff (E3h) "
            r"vs.\ moving smooth-kernel (E3s). "
            r"Near-zero negativity for E3s supports Route~A1: "
            r"smooth kernel weighting provides mitigation evidence (NOT a certified result)."
        ),
        label="tab:mc4_diag",
        col_rename=col_rename_mc4,
        float_fmt="%.4g",
        escape=False,
        table_env=True,
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 12. LATEX EXPORT HELPER  (Fix 12)
# ═══════════════════════════════════════════════════════════════════════════════

def df_to_latex(df, path, caption, label,
                col_rename=None,
                float_fmt="%.4g",
                escape=False,
                table_env=True):
    """
    Export a DataFrame as a paper-ready LaTeX table.

    Generates a proper table float (\\begin{table} … \\end{table}) with
    \\caption and \\label OUTSIDE the tabular environment.  The old approach
    of injecting \\caption after \\toprule was structurally invalid LaTeX and
    caused compilation errors.

    Parameters
    ----------
    df        : DataFrame to export.
    path      : Path object for the output .tex file.
    caption   : Table caption string (may contain raw LaTeX).
    label     : LaTeX label string (e.g. "tab:mc4_e3s").
    col_rename: dict mapping original column names → display names (LaTeX ok).
    float_fmt : printf-style format for float cells (default "%.4g").
    escape    : passed to df.to_latex(); set False to allow math in headers/cells.
    table_env : if True (default), wraps tabular in \\begin{table}…\\end{table}.
    """
    df_out = df.copy()
    if col_rename is not None:
        df_out = df_out.rename(columns=col_rename)

    # Pandas produces a complete tabular block
    tab = df_out.to_latex(
        index=False,
        escape=escape,
        float_format=float_fmt,
        longtable=False,
    )

    if table_env:
        tex = (
            "\\begin{table}[!htbp]\n"
            "\\centering\n"
            f"\\caption{{{caption}}}\n"
            f"\\label{{{label}}}\n"
            f"{tab}\n"
            "\\end{table}\n"
        )
    else:
        tex = tab

    with open(path, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"  LaTeX table saved: {path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# 13. ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  MONTE CARLO V2 (patched + fixed + editorial): Variable-Window HAC")
    print("  DGPs: AR(1)-GARCH  |  Markov-switching")
    print("  Estimators: E0, E1, E2, E3h, E3s")
    print("  Patches 1-4 + Fixes 1-12 + Changes 1-4 applied")
    print("="*60)

    # ── Monte Carlo ───────────────────────────────────────────────────────────
    results = run_monte_carlo()

    # ── Tables (CSV + inline LaTeX) ───────────────────────────────────────────
    # MC4 also auto-exports Table_MC4_E3s.tex (Cambio 2A)
    df1 = make_table_MC1(results)
    df2 = make_table_MC2(results)
    df3 = make_table_MC3(results)
    df4 = make_table_MC4_E3s(results)

    # ── Figures MC.1–3 ───────────────────────────────────────────────────────
    fig_MC1_negativity_frequency(results)
    fig_MC2_severity(results, T_plot=250)
    fig_MC2_severity(results, T_plot=1000)
    fig_MC3_inference(results)

    # ── Parameter map (refined_pts, NOT certified) ────────────────────────────
    alpha_grid, beta_grid, screen_map, refined_pts = run_parameter_map(
        N_alpha=30, N_beta=30, t_max=30.0, dt=0.1,
        x_max=3.0, n_x=220,
        tol_a=1e-14, n_quad=48,
        eps_scr=1e-10, K_cert=10
    )
    fig_MC4_parameter_map(alpha_grid, beta_grid, screen_map, refined_pts)
    fig_MC5_certified_curves(refined_pts)   # illustrative refined check, NOT certified

    # ── Multi-point Appendix-A certification ──────────────────────────────────
    # Also auto-exports Table_A1_certified.tex (Cambio 2B)
    cert_df = run_multipoint_certification_appendixA(
        screen_map, alpha_grid, beta_grid,
        X=3.0, A=200.0, N=131072
    )

    # MC6: sign certified at t0 only; Ŵ_{A,X,N}(t) curves with ±ε whiskers
    fig_MC6_certified_multipoint(cert_df, X=3.0, A=200.0, N=131072)

    print("\n" + "="*60)
    print("  ALL DONE.  Outputs saved to ./outputs/")
    print("  CSV outputs:")
    print("    Table_MC1.csv / MC2.csv / MC3.csv  (with R column)")
    print("    Table_MC4_E3s.csv                  (with R column)")
    print("    Table_certified_signchanges_APPENDIXA.csv")
    print("      -> sign=NEG/POS/NA, certified=True/False, stable col order")
    print("  LaTeX outputs (proper table float, caption outside tabular):")
    print("    Table_MC4_E3s.tex    -> tab:mc4_diag")
    print("    Table_A1_certified.tex -> tab:appA_certified")
    print("  Figures:")
    print("    Figure_MC6_certified.pdf/.png  (Ŵ_{A,X,N}; sign certified at t0 ONLY)")
    print("    Figure_MC5_refinedcheck.pdf/.png (illustrative, NOT certified)")
    print("    Figure_MC4.pdf/.png             (refined-stable, NOT certified)")
    print("="*60)