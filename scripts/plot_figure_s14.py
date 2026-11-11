#!/usr/bin/env python3
"""Figure S14 and Table S1: multimodal reproducibility benchmark comparing
automated high-throughput (HT) synthesis and characterization against manually
prepared reference films (McCarthy et al. 2026), simulated single-crystal X-ray
diffraction (SCXRD) powder patterns, and manually prepared DMF control films
across three bismuth-iodide compositions:
  - (PEAOH)BiI4 (1:1, X/M = 4.0)
  - (beta-Ala-H)2BiI5 (2:1, X/M = 5.0)
  - (beta-Ala-tBu)3Bi2I9 (3:2, X/M = 4.5)

Input:
  data/figs14_cohort_films.csv
  data/figs14_uvvis_spectra.csv
  data/figs14_uvvis_scans.csv
  data/figs14_diffraction_profiles.csv
  data/figs14_scxrd_simulated_peaks.csv
  data/figs14_rga_spectra.csv
  data/figs14_rga_runs.csv

Output (written to figures/):
  figures/figure_s14.{svg,pdf,png}
  figures/table_s1.{csv,tex}
  figures/figs14_peaks_all.csv
  figures/figs14_sensitivity.csv
  figures/figs14_text_values.csv
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
from matplotlib import font_manager
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

warnings.filterwarnings("ignore", category=RuntimeWarning)

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / ".." / "data"
OUTPUT_DIR = HERE / ".." / "figures"

RASTER_DPI = 300

# ---------------------------------------------------------------- analysis parameters
UVVIS_CFG = {
    "deriv_savgol_window_nm": 15,
    "deriv_savgol_polyorder": 3,
    "cosine_window_nm": (400.0, 650.0),
    "exciton_local_max_window_nm": (465.0, 565.0),
    "exciton_local_max_min_prominence": 0.002,
    "exciton_local_max_min_distance_nm": 10,
    "exciton_d2_fallback_window_nm": (472.0, 555.0),
    "exciton_d2_min_prominence": 1.0e-5,
    "exciton_d2_min_distance_nm": 8,
    "valid_film_min_abs_exciton_bgsub": 0.03,
    "valid_film_max_abs_exciton_raw": 3.5,
}

DIFFRACTION_WINDOWS = {
    "PEAOH": {
        "q1": (0.57, 0.64),
        "secondary (~0.96-0.97)": (0.93, 0.99),
    },
    "beta-Ala-H": {
        "low-q": (0.39, 0.45),
        "(002)": (0.535, 0.575),
        "(-101)": (0.605, 0.650),
        "(002)_narrow": (0.542, 0.560),
        "(-101)_narrow": (0.615, 0.635),
    },
    "beta-Ala-tBu": {
        "q1": (0.45, 0.51),
        "second-order (~0.95)": (0.935, 0.965),
    },
}

XRD_CO_KA1_A = 1.78897
XRD_CO_KA_WEIGHTED_A = 1.79026

RGA_ORGANIC_RANGE = (12, 160)
RGA_EXCLUDED_MZ = {18, 32}
RGA_DIAGNOSTIC_20_IONS = [
    2, 15, 16, 17, 26, 27, 28, 29, 39, 41, 42, 43, 44, 55, 56, 57, 58, 127, 141, 142
]

# ---------------------------------------------------------------- style
PUBLICATION_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": 8.5,
    "axes.titlesize": 10.2,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "axes.labelweight": "bold",
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.0,
    "axes.linewidth": 0.85,
    "xtick.major.width": 0.85,
    "ytick.major.width": 0.85,
    "xtick.major.size": 3.4,
    "ytick.major.size": 3.4,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
}

CATION_COLORS = {
    "PEAOH": "#1F4E79",        # Deep Navy Blue
    "beta-Ala-H": "#B85D19",   # Warm Rust Orange
    "beta-Ala-tBu": "#1E7145", # Forest Teal Green
}

CATION_DISPLAY = {
    "PEAOH": "PEAOH",
    "beta-Ala-H": r"$\beta$-Ala-H",
    "beta-Ala-tBu": r"$\beta$-Ala-$^{t}$Bu",
}


def apply_publication_style() -> None:
    """Raises if Arial is unavailable, matching Figures 3 and 4."""
    mpl.rcParams.update(PUBLICATION_STYLE)
    font_manager.findfont("Arial", fallback_to_default=False)


def rhu(val: float | str, quant: str = "0.001") -> str:
    """Round stored full-precision value using Decimal(..., ROUND_HALF_UP)."""
    return str(Decimal(str(val)).quantize(Decimal(quant), rounding=ROUND_HALF_UP))


# ---------------------------------------------------------------- UV-vis metrics
def analyze_bgsub_spectrum(wl_grid: np.ndarray, bgsub_grid: np.ndarray) -> dict:
    d_win = int(UVVIS_CFG["deriv_savgol_window_nm"])
    d_poly = int(UVVIS_CFG["deriv_savgol_polyorder"])
    d2_grid = savgol_filter(bgsub_grid, window_length=d_win, polyorder=d_poly, deriv=2, delta=1.0)

    exc_lo, exc_hi = UVVIS_CFG["exciton_local_max_window_nm"]
    exc_mask = (wl_grid >= exc_lo) & (wl_grid <= exc_hi)
    wl_exc = wl_grid[exc_mask]
    abs_exc = bgsub_grid[exc_mask]

    peaks, props = find_peaks(
        abs_exc,
        prominence=float(UVVIS_CFG["exciton_local_max_min_prominence"]),
        distance=int(UVVIS_CFG["exciton_local_max_min_distance_nm"]),
    )
    if len(peaks) > 0:
        best_idx = peaks[int(np.argmax(props["prominences"]))]
        lambda_peak_max = float(wl_exc[best_idx])
        has_local_max = True
    else:
        lambda_peak_max = float("nan")
        has_local_max = False

    d2_lo, d2_hi = UVVIS_CFG["exciton_d2_fallback_window_nm"]
    d2_mask = (wl_grid >= d2_lo) & (wl_grid <= d2_hi)
    wl_d2 = wl_grid[d2_mask]
    d2_sub = d2_grid[d2_mask]
    neg_peaks, _ = find_peaks(
        -d2_sub,
        prominence=float(UVVIS_CFG["exciton_d2_min_prominence"]),
        distance=int(UVVIS_CFG["exciton_d2_min_distance_nm"]),
    )
    if len(neg_peaks) > 0:
        best_d2_idx = neg_peaks[int(np.argmin(d2_sub[neg_peaks]))]
        lambda_d2_min = float(wl_d2[best_d2_idx])
    else:
        lambda_d2_min = float(wl_d2[int(np.argmin(d2_sub))])

    lambda_exciton = lambda_peak_max if has_local_max else lambda_d2_min
    idx_exc = int(np.argmin(np.abs(wl_grid - lambda_exciton)))
    abs_exciton_bgsub = float(bgsub_grid[idx_exc])

    norm_denom = (
        abs_exciton_bgsub
        if abs_exciton_bgsub > 0.02
        else float(np.nanmax(bgsub_grid[(wl_grid >= 400.0) & (wl_grid <= 650.0)]))
    )
    if norm_denom <= 1e-6:
        norm_denom = 1.0
    norm_grid = bgsub_grid / norm_denom

    return {
        "bgsub_grid": bgsub_grid,
        "norm_grid": norm_grid,
        "has_local_max": has_local_max,
        "lambda_peak_max_nm": lambda_peak_max,
        "lambda_d2_min_nm": lambda_d2_min,
        "lambda_exciton_nm": lambda_exciton,
        "abs_exciton_bgsub": abs_exciton_bgsub,
    }


def cosine_uv(bgsub_a: np.ndarray, bgsub_b: np.ndarray, wl_grid: np.ndarray, wl_min: float, wl_max: float) -> float:
    mask = (wl_grid >= wl_min) & (wl_grid <= wl_max)
    ya = np.clip(bgsub_a[mask], 0.0, None)
    yb = np.clip(bgsub_b[mask], 0.0, None)
    na = float(np.linalg.norm(ya))
    nb = float(np.linalg.norm(yb))
    return float(np.dot(ya, yb) / (na * nb)) if (na > 1e-9 and nb > 1e-9) else float("nan")


def compute_uvvis_metrics(uv_spectra: pd.DataFrame, uv_scans: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    wl = uv_spectra["wavelength_nm"].to_numpy(dtype=float)
    cos_lo, cos_hi = UVVIS_CFG["cosine_window_nm"]

    ref_solvents = {"PEAOH": "DMF", "beta-Ala-H": "MeOH", "beta-Ala-tBu": "MeOH"}
    ref_specs = {}
    dmf_specs = {}
    for comp in ["PEAOH", "beta-Ala-H", "beta-Ala-tBu"]:
        ctag = comp.replace("-", "_")
        ref_specs[comp] = analyze_bgsub_spectrum(wl, uv_spectra[f"Reference_{ctag}_bgsub"].to_numpy(dtype=float))
        dmf_specs[comp] = analyze_bgsub_spectrum(wl, uv_spectra[f"DMF_control_{ctag}_bgsub"].to_numpy(dtype=float))

    # Recompute cohort normalized envelopes and summary metrics directly from stored spectra
    ht_norm_by_mfid = {}
    for _, r in uv_scans[uv_scans["included_in_100_120C_cohort"] == "y"].iterrows():
        mfid = str(r["mfid"])
        col = f"HT_{mfid}_bgsub"
        if col in uv_spectra.columns:
            proc = analyze_bgsub_spectrum(wl, uv_spectra[col].to_numpy(dtype=float))
            ht_norm_by_mfid[mfid] = proc["norm_grid"]

    envelopes = {"wavelength_nm": wl}
    summary_rows = []
    for comp in ["PEAOH", "beta-Ala-H", "beta-Ala-tBu"]:
        ctag = comp.replace("-", "_")
        sub_c = uv_scans[(uv_scans["composition"] == comp) & (uv_scans["included_in_uvvis_cohort"] == True)].copy()
        # Recompute per-film cosine similarities directly from bgsub spectra
        cos_ref_vals = [
            cosine_uv(uv_spectra[f"HT_{m}_bgsub"].to_numpy(dtype=float), ref_specs[comp]["bgsub_grid"], wl, cos_lo, cos_hi)
            for m in sub_c["mfid"]
        ]
        cos_dmf_vals = [
            cosine_uv(uv_spectra[f"HT_{m}_bgsub"].to_numpy(dtype=float), dmf_specs[comp]["bgsub_grid"], wl, cos_lo, cos_hi)
            for m in sub_c["mfid"]
        ]
        norm_stack = np.vstack([ht_norm_by_mfid[m] for m in sub_c["mfid"]])
        envelopes[f"Reference_{ctag}_norm"] = ref_specs[comp]["norm_grid"]
        envelopes[f"DMF_control_{ctag}_norm"] = dmf_specs[comp]["norm_grid"]
        envelopes[f"HT_{ctag}_norm_mean"] = np.nanmean(norm_stack, axis=0)
        envelopes[f"HT_{ctag}_norm_std"] = np.nanstd(norm_stack, axis=0)

        summary_rows.append({
            "composition": comp,
            "n_films": len(sub_c),
            "n_films_8of10_positions": int((sub_c["n_retained_positions"] == 8).sum()),
            "n_films_3of5_positions": int((sub_c["n_retained_positions"] == 3).sum()),
            "ht_lambda_exc_mean_nm": float(sub_c["lambda_exciton_nm"].mean()),
            "ht_lambda_exc_sd_nm": float(sub_c["lambda_exciton_nm"].std(ddof=1)),
            "reference_lambda_exc_nm": float(ref_specs[comp]["lambda_exciton_nm"]),
            "reference_solvent": ref_solvents[comp],
            "dmf_control_lambda_exc_nm": float(dmf_specs[comp]["lambda_exciton_nm"]),
            "dmf_control_solvent": "DMF",
            "cos_theta_vs_reference_mean": float(np.mean(cos_ref_vals)),
            "cos_theta_vs_dmf_control_mean": float(np.mean(cos_dmf_vals)),
        })

    return pd.DataFrame(summary_rows).set_index("composition"), envelopes


# ---------------------------------------------------------------- diffraction metrics
def parabolic_peak_with_bin(q_grid: np.ndarray, y: np.ndarray, q_lo: float, q_hi: float) -> tuple[float, float]:
    """3-point parabolic refinement around local argmax on unsmoothed 0.005 A^-1 binned profile."""
    m = (q_grid >= q_lo) & (q_grid <= q_hi) & np.isfinite(y)
    idxs = np.where(m)[0]
    best_i = idxs[int(np.argmax(y[idxs]))]
    q_bin = float(q_grid[best_i])
    q_ref = q_bin
    if 0 < best_i < len(q_grid) - 1:
        y0, y1, y2 = float(y[best_i - 1]), float(y[best_i]), float(y[best_i + 1])
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-12 and y1 >= y0 and y1 >= y2:
            delta = 0.5 * (y0 - y2) / denom
            if abs(delta) <= 1.0:
                dq = float(q_grid[best_i + 1] - q_grid[best_i])
                q_ref = q_bin + delta * dq
    return q_bin, q_ref


def compute_diffraction_metrics(diff_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    q_grid = diff_df["q_invA"].to_numpy(dtype=float)

    # Parse individual HT scan columns: HT_scan_{ctag}_{mfid}_{ang_tag}
    scan_cols_by_comp: dict[str, list[tuple[str, float, np.ndarray]]] = {
        "PEAOH": [],
        "beta-Ala-H": [],
        "beta-Ala-tBu": [],
    }
    ctag_to_comp = {
        "PEAOH": "PEAOH",
        "beta_Ala_H": "beta-Ala-H",
        "beta_Ala_tBu": "beta-Ala-tBu",
    }
    for col in diff_df.columns:
        if not col.startswith("HT_scan_"):
            continue
        rem = col[len("HT_scan_"):]
        for ctag, comp_name in ctag_to_comp.items():
            if rem.startswith(ctag + "_"):
                tail = rem[len(ctag) + 1:]
                mfid, ang_raw = tail.split("_", 1)
                ang_deg = float(ang_raw.replace("deg", "").replace("p", "."))
                scan_cols_by_comp[comp_name].append((mfid, ang_deg, diff_df[col].to_numpy(dtype=float)))
                break

    refl_specs = [
        ("PEAOH", "PEAOH", "q1", DIFFRACTION_WINDOWS["PEAOH"]["q1"], "(-3,0,2) at 0.61315 A^-1 (100 K SCXRD simulation)"),
        ("PEAOH", "PEAOH", "secondary (~0.96-0.97)", DIFFRACTION_WINDOWS["PEAOH"]["secondary (~0.96-0.97)"], "(2,0,-2)/(-3,1,1) multiplet (~0.970 A^-1)"),
        ("beta-Ala-H", "beta_Ala_H", "low-q", DIFFRACTION_WINDOWS["beta-Ala-H"]["low-q"], "DMF-promoted low-q lamellar reflection (absent in MeOH SCXRD)"),
        ("beta-Ala-H", "beta_Ala_H", "(002)", DIFFRACTION_WINDOWS["beta-Ala-H"]["(002)"], "(0,0,2) simulated Bragg q = 0.55238 A^-1 (298 K)"),
        ("beta-Ala-H", "beta_Ala_H", "(-101)", DIFFRACTION_WINDOWS["beta-Ala-H"]["(-101)"], "(-1,0,1) simulated Bragg q = 0.62724 A^-1 (298 K)"),
        ("beta-Ala-tBu", "beta_Ala_tBu", "q1", DIFFRACTION_WINDOWS["beta-Ala-tBu"]["q1"], "(1,0,-1) simulated Bragg q = 0.46641 A^-1 (d = 13.4714 A, I_rel = 38.67%, 273 K)"),
        ("beta-Ala-tBu", "beta_Ala_tBu", "second-order (~0.95)", DIFFRACTION_WINDOWS["beta-Ala-tBu"]["second-order (~0.95)"], "2nd-order harmonic of q1 (broadened simulated peak at 0.96128 A^-1)"),
    ]

    peak_rows = []
    one_per_film_rows = []

    for comp_lbl, ctag, refl_lbl, (q_lo, q_hi), sim_note in refl_specs:
        col_ht = f"HT_{ctag}_mean"
        y_ht = diff_df[col_ht].to_numpy(dtype=float)
        qb_ht, qr_ht = parabolic_peak_with_bin(q_grid, y_ht, q_lo, q_hi)

        scans = scan_cols_by_comp[comp_lbl]
        uniq_films = sorted({m for m, _, _ in scans})
        film_peaks = []
        film_peaks_narrow = []
        one_pat_profiles = []
        shoulder_film_notes = []

        if comp_lbl == "beta-Ala-H" and refl_lbl == "(002)":
            nar_lo, nar_hi = DIFFRACTION_WINDOWS["beta-Ala-H"]["(002)_narrow"]
        elif comp_lbl == "beta-Ala-H" and refl_lbl == "(-101)":
            nar_lo, nar_hi = DIFFRACTION_WINDOWS["beta-Ala-H"]["(-101)_narrow"]
        else:
            nar_lo, nar_hi = q_lo, q_hi

        for fn in uniq_films:
            f_scans = [(ang, prof) for m, ang, prof in scans if m == fn]
            f_prof = np.nanmean(np.vstack([prof for _, prof in f_scans]), axis=0)
            _, fq_ref = parabolic_peak_with_bin(q_grid, f_prof, q_lo, q_hi)
            _, fq_nar = parabolic_peak_with_bin(q_grid, f_prof, nar_lo, nar_hi)
            film_peaks.append(fq_ref)
            film_peaks_narrow.append(fq_nar)
            if comp_lbl == "beta-Ala-H" and refl_lbl == "(002)" and fq_ref > 0.562:
                shoulder_film_notes.append(f"{fn} ({fq_ref:.7f} A^-1)")
            best_prof = min(f_scans, key=lambda item: abs(item[0] - 0.3))[1]
            one_pat_profiles.append(best_prof)

        film_sd_wide = float(np.std(film_peaks, ddof=1)) if len(film_peaks) > 1 else 0.0
        film_sd_narrow = float(np.std(film_peaks_narrow, ddof=1)) if len(film_peaks_narrow) > 1 else 0.0

        pat_peaks = [parabolic_peak_with_bin(q_grid, prof, q_lo, q_hi)[1] for _, _, prof in scans]
        pattern_sd = float(np.std(pat_peaks, ddof=1)) if len(pat_peaks) > 1 else 0.0

        y_one = np.nanmean(np.vstack(one_pat_profiles), axis=0)
        _, qr_one = parabolic_peak_with_bin(q_grid, y_one, q_lo, q_hi)

        qr_ht_3dec = rhu(f"{qr_ht:.10f}", "0.001")
        qr_one_3dec = rhu(f"{qr_one:.10f}", "0.001")
        shift_full = qr_one - qr_ht
        abs_shift_full = abs(shift_full)
        abs_shift_3dec = abs(Decimal(qr_one_3dec) - Decimal(qr_ht_3dec))

        one_per_film_rows.append({
            "composition": comp_lbl,
            "reflection": refl_lbl,
            "n_patterns_all": len(scans),
            "n_films_one": len(uniq_films),
            "all_patterns_q_full": f"{qr_ht:.7f}",
            "all_patterns_q_5dec": rhu(f"{qr_ht:.10f}", "0.00001"),
            "all_patterns_q_3dec": qr_ht_3dec,
            "one_per_film_q_full": f"{qr_one:.7f}",
            "one_per_film_q_5dec": rhu(f"{qr_one:.10f}", "0.00001"),
            "one_per_film_q_3dec": qr_one_3dec,
            "shift_full_invA": f"{shift_full:+.7f}",
            "shift_5dec_invA": f"{float(rhu(f'{shift_full:.10f}', '0.00001')):+.5f}",
            "abs_shift_full_invA": f"{abs_shift_full:.7f}",
            "abs_shift_5dec_invA": rhu(f"{abs_shift_full:.10f}", "0.00001"),
            "abs_shift_3dec_invA": f"{abs_shift_3dec:.3f}",
        })

        ht_note = f"N={len(scans)} patterns across {len(uniq_films)} films (TF000004 excluded); column {col_ht}"
        if comp_lbl == "beta-Ala-H" and refl_lbl in ("(002)", "(-101)"):
            ht_note += (
                f"; per-film refined SD = {film_sd_narrow:.7f} A^-1 ({rhu(f'{film_sd_narrow:.10f}', '0.001')}) on narrow window [{nar_lo}, {nar_hi}] "
                f"and {film_sd_wide:.7f} A^-1 ({rhu(f'{film_sd_wide:.10f}', '0.001')}) on wide window [{q_lo}, {q_hi}]"
            )
            if shoulder_film_notes:
                ht_note += f" (films carrying ~0.568 A^-1 shoulder: {', '.join(shoulder_film_notes)})"

        peak_rows.append({
            "composition": comp_lbl,
            "reflection": refl_lbl,
            "source": "HT GIWAXS",
            "q_bin_center": f"{qb_ht:.4f}",
            "q_refined_full": f"{qr_ht:.7f}",
            "q_refined_5dec": rhu(f"{qr_ht:.10f}", "0.00001"),
            "q_3dec": qr_ht_3dec,
            "d_spacing_A_from_full": f"{2.0 * np.pi / qr_ht:.7f}",
            "d_spacing_A_from_5dec": rhu(f"{2.0 * np.pi / qr_ht:.10f}", "0.0001"),
            "film_to_film_SD": f"{film_sd_wide:.7f}",
            "film_to_film_SD_3dec": rhu(f"{film_sd_wide:.10f}", "0.001"),
            "film_to_film_SD_narrow": f"{film_sd_narrow:.7f}" if (comp_lbl == "beta-Ala-H" and refl_lbl in ("(002)", "(-101)")) else "",
            "film_to_film_SD_narrow_3dec": rhu(f"{film_sd_narrow:.10f}", "0.001") if (comp_lbl == "beta-Ala-H" and refl_lbl in ("(002)", "(-101)")) else "",
            "pattern_to_pattern_SD": f"{pattern_sd:.7f}",
            "search_window_invA": f"[{q_lo}, {q_hi}]",
            "narrow_window_invA": f"[{nar_lo}, {nar_hi}]" if (comp_lbl == "beta-Ala-H" and refl_lbl in ("(002)", "(-101)")) else "",
            "notes": ht_note,
        })

        # Reference GIWAXS, Reference XRD, and SCXRD (simulated)
        other_sources = [
            ("Reference GIWAXS", f"Reference_GIWAXS_{ctag}_norm", f"Reference GIWAXS on Si; column Reference_GIWAXS_{ctag}_norm"),
            ("Reference XRD", f"Reference_XRD_{ctag}_norm", f"Co K-alpha1 (1.78897 A) on 0.005 A^-1 grid; column Reference_XRD_{ctag}_norm"),
            ("SCXRD (simulated)", f"SCXRD_sim_{ctag}_norm", sim_note),
        ]
        for src_name, src_col, src_note in other_sources:
            if comp_lbl == "beta-Ala-H" and refl_lbl == "low-q":
                peak_rows.append({
                    "composition": comp_lbl,
                    "reflection": refl_lbl,
                    "source": src_name,
                    "q_bin_center": "N/A",
                    "q_refined_full": "N/A",
                    "q_refined_5dec": "N/A",
                    "q_3dec": "N/A",
                    "d_spacing_A_from_full": "N/A",
                    "d_spacing_A_from_5dec": "N/A",
                    "film_to_film_SD": "",
                    "film_to_film_SD_3dec": "",
                    "film_to_film_SD_narrow": "",
                    "film_to_film_SD_narrow_3dec": "",
                    "pattern_to_pattern_SD": "",
                    "search_window_invA": f"[{q_lo}, {q_hi}]",
                    "narrow_window_invA": "",
                    "notes": f"DMF-promoted reflection absent in MeOH {src_name}",
                })
            else:
                y_s = diff_df[src_col].to_numpy(dtype=float)
                qb_s, qr_s = parabolic_peak_with_bin(q_grid, y_s, q_lo, q_hi)
                peak_rows.append({
                    "composition": comp_lbl,
                    "reflection": refl_lbl,
                    "source": src_name,
                    "q_bin_center": f"{qb_s:.4f}",
                    "q_refined_full": f"{qr_s:.7f}",
                    "q_refined_5dec": rhu(f"{qr_s:.10f}", "0.00001"),
                    "q_3dec": rhu(f"{qr_s:.10f}", "0.001"),
                    "d_spacing_A_from_full": f"{2.0 * np.pi / qr_s:.7f}",
                    "d_spacing_A_from_5dec": rhu(f"{2.0 * np.pi / qr_s:.10f}", "0.0001"),
                    "film_to_film_SD": "",
                    "film_to_film_SD_3dec": "",
                    "film_to_film_SD_narrow": "",
                    "film_to_film_SD_narrow_3dec": "",
                    "pattern_to_pattern_SD": "",
                    "search_window_invA": f"[{q_lo}, {q_hi}]",
                    "narrow_window_invA": "",
                    "notes": src_note,
                })

    return pd.DataFrame(peak_rows), pd.DataFrame(one_per_film_rows)


# ---------------------------------------------------------------- RGA metrics
def cosine_vec(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    return float(np.dot(a, b) / (na * nb)) if (na > 0 and nb > 0) else float("nan")


def compute_rga_metrics(
    rga_spectra: pd.DataFrame, rga_runs: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    org_lo, org_hi = RGA_ORGANIC_RANGE
    org_mask = np.array([(org_lo <= mz <= org_hi) and (mz not in RGA_EXCLUDED_MZ) for mz in range(1, 301)], dtype=bool)
    diag_idx = [m - 1 for m in RGA_DIAGNOSTIC_20_IONS]

    ref_rows = rga_runs[rga_runs["source"] == "Reference"].set_index("composition")
    ht_rows = rga_runs[rga_runs["source"] == "HT"].copy()

    ref_clip_by_comp = {}
    ref_org_by_comp = {}
    for comp in ["PEAOH", "beta-Ala-H", "beta-Ala-tBu"]:
        ctag = comp.replace("-", "_")
        p_torr = rga_spectra[f"Reference_{ctag}_torr"].to_numpy(dtype=float)
        p_clip = np.clip(p_torr, 0.0, None)
        ref_clip_by_comp[comp] = p_clip
        ref_org_by_comp[comp] = float(np.sum(p_clip[org_mask]))

    per_run_records = []
    for _, r in ht_rows.iterrows():
        rkey = str(r["run_key"])
        comp = str(r["composition"])
        delta_ms = rga_spectra[f"{rkey}_torr"].to_numpy(dtype=float)
        delta_clip = np.clip(delta_ms, 0.0, None)
        org_torr = float(np.sum(delta_clip[org_mask]))
        cos_self = cosine_vec(delta_clip[diag_idx], ref_clip_by_comp[comp][diag_idx])

        per_run_records.append({
            "run_key": rkey,
            "mfid": str(r["mfid"]),
            "rga_session": str(r["rga_session"]),
            "composition": comp,
            "am_ratio": str(r["am_ratio"]),
            "anneal_c": r["anneal_c"],
            "exposure_duration_s": float(r["exposure_duration_s"]),
            "in_primary_area_cohort": bool(r["in_primary_area_cohort"]),
            "included_in_100_120C_cohort": bool(r["included_in_100_120C_cohort"]),
            "organic_outgassing_torr": org_torr,
            "organic_outgassing_1e9_torr": org_torr * 1e9,
            "mz_28_torr": float(delta_clip[27]),
            "mz_44_torr": float(delta_clip[43]),
            "mz_58_signed_torr": float(delta_ms[57]),
            "mz_73_signed_torr": float(delta_ms[72]),
            "cos_20ion_self": cos_self,
        })
    df_per_run = pd.DataFrame(per_run_records)

    ensemble_spectra: dict[str, np.ndarray] = {"mz": rga_spectra["mz"].to_numpy(dtype=int)}
    summary_rows = []
    for comp in ["PEAOH", "beta-Ala-H", "beta-Ala-tBu"]:
        ctag = comp.replace("-", "_")
        ensemble_spectra[f"Reference_{ctag}_torr"] = rga_spectra[f"Reference_{ctag}_torr"].to_numpy(dtype=float)

        sub_100_120 = df_per_run[(df_per_run["composition"] == comp) & (df_per_run["included_in_100_120C_cohort"] == True)]
        sub_all_t = df_per_run[(df_per_run["composition"] == comp) & (df_per_run["in_primary_area_cohort"] == True)]

        mat_100_120 = np.vstack([rga_spectra[f"{rk}_torr"].to_numpy(dtype=float) for rk in sub_100_120["run_key"]])
        mean_100_120 = np.mean(mat_100_120, axis=0)
        std_100_120 = np.std(mat_100_120, axis=0)
        ensemble_spectra[f"HT_{ctag}_100_120C_torr"] = mean_100_120

        mean_clip = np.clip(mean_100_120, 0.0, None)
        l_clip = ref_clip_by_comp[comp]
        cos_20_100_120 = cosine_vec(mean_clip[diag_idx], l_clip[diag_idx])

        mat_all_t = np.vstack([rga_spectra[f"{rk}_torr"].to_numpy(dtype=float) for rk in sub_all_t["run_key"]])
        mean_all_t_clip = np.clip(np.mean(mat_all_t, axis=0), 0.0, None)
        cos_20_all_t = cosine_vec(mean_all_t_clip[diag_idx], l_clip[diag_idx])

        ht_org_run_mean = float(sub_100_120["organic_outgassing_torr"].mean())
        ht_org_run_sd = float(sub_100_120["organic_outgassing_torr"].std(ddof=1)) if len(sub_100_120) > 1 else 0.0
        ht_org_spec_sum = float(np.sum(mean_clip[org_mask]))
        ref_org = ref_org_by_comp[comp]

        mz28_ht = float(mean_clip[27])
        mz44_ht = float(mean_clip[43])
        mz28_ref = float(l_clip[27])
        mz44_ref = float(l_clip[43])
        ratio_44_28 = mz44_ht / mz28_ht if mz28_ht > 0 else float("nan")
        ref_ratio_44_28 = mz44_ref / mz28_ref if mz28_ref > 0 else float("nan")

        diff_org_run_mean = ht_org_run_mean - ref_org
        diff_org_spec_sum = ht_org_spec_sum - ref_org
        diff_44 = mz44_ht - mz44_ref
        share_44_pct = 100.0 * diff_44 / diff_org_run_mean if abs(diff_org_run_mean) > 1e-15 else float("nan")
        share_44_spec_pct = 100.0 * diff_44 / diff_org_spec_sum if abs(diff_org_spec_sum) > 1e-15 else float("nan")

        rms_noise_60_120 = float(np.sqrt(np.mean((mean_100_120[59:120] * 1e9) ** 2)))
        sd_noise_60_120 = float(np.std(mean_100_120[59:120] * 1e9, ddof=1))

        ref_meta = ref_rows.loc[comp]
        summary_rows.append({
            "composition": comp,
            "n_runs_100_120C": len(sub_100_120),
            "mfids_100_120C": ";".join(sub_100_120["mfid"].tolist()),
            "ht_organic_outgassing_1e9_torr_mean": ht_org_run_mean * 1e9,
            "ht_organic_outgassing_1e9_torr_sd": ht_org_run_sd * 1e9,
            "ht_organic_outgassing_from_mean_spec_1e9_torr": ht_org_spec_sum * 1e9,
            "reference_organic_outgassing_1e9_torr": ref_org * 1e9,
            "cos_theta_20ion_100_120C": cos_20_100_120,
            "cos_theta_20ion_all_temps": cos_20_all_t,
            "ht_mz44_1e9_torr": mz44_ht * 1e9,
            "ht_mz28_1e9_torr": mz28_ht * 1e9,
            "ht_mz44_to_mz28_ratio": ratio_44_28,
            "reference_mz44_1e9_torr": mz44_ref * 1e9,
            "reference_mz28_1e9_torr": mz28_ref * 1e9,
            "reference_mz44_to_mz28_ratio": ref_ratio_44_28,
            "delta_mz44_share_of_delta_organic_pct": share_44_pct,
            "delta_mz44_share_from_mean_spec_pct": share_44_spec_pct,
            "ht_mz58_signed_1e9_torr": float(mean_100_120[57] * 1e9),
            "ht_mz58_std_1e9_torr": float(std_100_120[57] * 1e9),
            "ht_mz73_signed_1e9_torr": float(mean_100_120[72] * 1e9),
            "ht_mz73_std_1e9_torr": float(std_100_120[72] * 1e9),
            "ht_rms_noise_mz60_120_1e9_torr": rms_noise_60_120,
            "ht_sd_noise_mz60_120_1e9_torr": sd_noise_60_120,
            "ref_t_first_sweep_s": float(ref_meta["t_first_sweep_s"]),
            "ref_t_fourth_sweep_s": float(ref_meta["t_fourth_sweep_s"]),
            "ref_t_span_starts_s": float(ref_meta["t_span_starts_s"]),
            "ref_t_next_sweep_s": float(ref_meta["t_next_sweep_s"]),
            "ref_t_four_sweep_window_s": float(ref_meta["t_four_sweep_window_s"]),
        })

    return pd.DataFrame(summary_rows).set_index("composition"), df_per_run, ensemble_spectra


# ---------------------------------------------------------------- sensitivity & tables
def build_sensitivity_table(
    cohort_df: pd.DataFrame, uv_scans: pd.DataFrame, rga_sum: pd.DataFrame
) -> pd.DataFrame:
    cohort_uv = uv_scans[uv_scans["included_in_uvvis_cohort"] == True].copy()
    rows = []

    def add_uv_row(
        analysis_group: str,
        variant_name: str,
        comp: str,
        lam_vals: pd.Series,
        cref_vals: pd.Series,
        cdmf_vals: pd.Series,
        base_df: pd.DataFrame,
        notes: str,
    ) -> None:
        lam_m = float(lam_vals.mean())
        lam_sd = float(lam_vals.std(ddof=1)) if len(lam_vals) > 1 else 0.0
        cref_m = float(cref_vals.mean())
        cdmf_m = float(cdmf_vals.mean())

        b_lam_m = float(base_df["lambda_exciton_nm"].mean())
        b_cref_m = float(base_df["cos_theta_vs_reference"].mean())
        b_cdmf_m = float(base_df["cos_theta_vs_dmf_control"].mean())

        lam_m_1d = rhu(f"{lam_m:.10f}", "0.1")
        lam_sd_1d = rhu(f"{lam_sd:.10f}", "0.1")
        cref_3d = rhu(f"{cref_m:.10f}", "0.001")
        cdmf_3d = rhu(f"{cdmf_m:.10f}", "0.001")

        rows.append({
            "analysis_group": analysis_group,
            "variant": variant_name,
            "composition": comp,
            "n_films": len(lam_vals),
            "lambda_exc_mean_nm_full": f"{lam_m:.7f}",
            "lambda_exc_sd_nm_full": f"{lam_sd:.7f}",
            "lambda_exc_mean_nm_4dec": rhu(f"{lam_m:.10f}", "0.0001"),
            "lambda_exc_sd_nm_4dec": rhu(f"{lam_sd:.10f}", "0.0001"),
            "lambda_exc_reported_1dec": f"{lam_m_1d} +/- {lam_sd_1d}",
            "cos_theta_vs_reference_full": f"{cref_m:.7f}",
            "cos_theta_vs_reference_4dec": rhu(f"{cref_m:.10f}", "0.0001"),
            "cos_theta_vs_reference_3dec": cref_3d,
            "cos_theta_vs_dmf_control_full": f"{cdmf_m:.7f}",
            "cos_theta_vs_dmf_control_4dec": rhu(f"{cdmf_m:.10f}", "0.0001"),
            "cos_theta_vs_dmf_control_3dec": cdmf_3d,
            "delta_lambda_exc_vs_baseline_nm": f"{lam_m - b_lam_m:+.7f}",
            "delta_cos_ref_vs_baseline": f"{cref_m - b_cref_m:+.7f}",
            "delta_cos_dmf_vs_baseline": f"{cdmf_m - b_cdmf_m:+.7f}",
            "rga_ht_organic_outgassing_full_1e9_torr": "",
            "rga_ht_organic_outgassing_2dec_1e9_torr": "",
            "rga_reference_organic_outgassing_full_1e9_torr": "",
            "rga_reference_organic_outgassing_2dec_1e9_torr": "",
            "rga_delta_mz44_share_full_pct": "",
            "rga_delta_mz44_share_1dec_pct": "",
            "notes": notes,
        })

    # 1. Baseline full cohort
    for comp in ["PEAOH", "beta-Ala-H", "beta-Ala-tBu"]:
        sub = cohort_uv[cohort_uv["composition"] == comp]
        add_uv_row(
            "1_baseline_cohort",
            "full_100_120C_cohort",
            comp,
            sub["lambda_exciton_nm"],
            sub["cos_theta_vs_reference"],
            sub["cos_theta_vs_dmf_control"],
            sub,
            "Baseline 100-120 C cohort (41 cohort films: 18/19/4; 39 valid UV-vis: 17/18/4; central 8 of 10 on ITO [32 films], central 3 of 5 on Dec 2025 glass [7 films])",
        )

    # 2. beta-Ala-H excluding TF000326
    bala_base = cohort_uv[cohort_uv["composition"] == "beta-Ala-H"]
    bala_no326 = bala_base[bala_base["mfid"] != "TF000326"]
    add_uv_row(
        "2_outlier_exclusion",
        "beta_Ala_H_excluding_TF000326",
        "beta-Ala-H",
        bala_no326["lambda_exciton_nm"],
        bala_no326["cos_theta_vs_reference"],
        bala_no326["cos_theta_vs_dmf_control"],
        bala_base,
        "Excludes TF000326 (N=17; TF000326 has local maximum at 472.0 nm and 2nd-derivative minimum at 478.0 nm)",
    )

    # 3. PMMA subsets
    for comp in ["PEAOH", "beta-Ala-H", "beta-Ala-tBu"]:
        base_c = cohort_uv[cohort_uv["composition"] == comp]
        unc_c = base_c[base_c["pmma_coated"] == "n"]
        coa_c = base_c[base_c["pmma_coated"] == "y"]
        add_uv_row(
            "3_pmma_sensitivity",
            "pmma_uncoated_only",
            comp,
            unc_c["lambda_exciton_nm"],
            unc_c["cos_theta_vs_reference"],
            unc_c["cos_theta_vs_dmf_control"],
            base_c,
            f"Uncoated films only (N={len(unc_c)}); excludes {len(coa_c)} PMMA-coated films ({';'.join(coa_c['mfid'].tolist())})",
        )
        if len(coa_c) > 0:
            add_uv_row(
                "3_pmma_sensitivity",
                "pmma_coated_only",
                comp,
                coa_c["lambda_exciton_nm"],
                coa_c["cos_theta_vs_reference"],
                coa_c["cos_theta_vs_dmf_control"],
                base_c,
                f"PMMA-coated films only (N={len(coa_c)}: {';'.join(coa_c['mfid'].tolist())})",
            )

    # 4. Strict non-null heater_sv_temp subset
    for comp in ["PEAOH", "beta-Ala-H", "beta-Ala-tBu"]:
        base_c = cohort_uv[cohort_uv["composition"] == comp]
        strict_c = base_c[base_c["temperature_source_field"] == "heater_sv_temp"]
        n_coh_strict = int(
            (
                (cohort_df["composition"] == comp)
                & (cohort_df["included"] == "y")
                & (cohort_df["temperature_source_field"] == "heater_sv_temp")
            ).sum()
        )
        add_uv_row(
            "4_temperature_fallback_sensitivity",
            "strict_nonnull_heater_sv_temp_only",
            comp,
            strict_c["lambda_exciton_nm"],
            strict_c["cos_theta_vs_reference"],
            strict_c["cos_theta_vs_dmf_control"],
            base_c,
            f"Strict non-null heater_sv_temp cohort ({n_coh_strict} cohort films, {len(strict_c)} valid UV-vis films)",
        )
    strict_bala_no326 = bala_base[
        (bala_base["temperature_source_field"] == "heater_sv_temp") & (bala_base["mfid"] != "TF000326")
    ]
    add_uv_row(
        "4_temperature_fallback_sensitivity",
        "strict_nonnull_heater_sv_temp_excl_TF000326",
        "beta-Ala-H",
        strict_bala_no326["lambda_exciton_nm"],
        strict_bala_no326["cos_theta_vs_reference"],
        strict_bala_no326["cos_theta_vs_dmf_control"],
        bala_base,
        "Strict non-null heater_sv_temp cohort for beta-Ala-H excluding TF000326 (N=15)",
    )

    # 5. Spatial aggregation variants
    spatial_variants = [
        ("current_slice_median_1_to_m1", None, "Default slice abs_2d[1:-1, :] with pixel-wise median"),
        ("all_spectra_median", "all_spectra_median", "All stored spatial positions abs_2d[:, :] with pixel-wise median"),
        ("all_spectra_mean", "all_spectra_mean", "All stored spatial positions abs_2d[:, :] with pixel-wise mean"),
        ("central_six_median", "central_six_median", "Central 6 of 10 positions on ITO (central 3 of 5 on glass) with pixel-wise median"),
        ("current_slice_mean_1_to_m1", "current_slice_mean_1_to_m1", "Slice abs_2d[1:-1, :] with pixel-wise mean"),
    ]
    for var_name, col_pref, var_desc in spatial_variants:
        for comp in ["PEAOH", "beta-Ala-H", "beta-Ala-tBu"]:
            base_c = cohort_uv[cohort_uv["composition"] == comp]
            if col_pref is None:
                add_uv_row(
                    "5_uvvis_spatial_aggregation",
                    var_name,
                    comp,
                    base_c["lambda_exciton_nm"],
                    base_c["cos_theta_vs_reference"],
                    base_c["cos_theta_vs_dmf_control"],
                    base_c,
                    var_desc,
                )
            else:
                add_uv_row(
                    "5_uvvis_spatial_aggregation",
                    var_name,
                    comp,
                    base_c[f"{col_pref}_lambda_exc_nm"],
                    base_c[f"{col_pref}_cos_ref"],
                    base_c[f"{col_pref}_cos_dmf"],
                    base_c,
                    var_desc,
                )

    # 6. Cosine similarity wavelength window sensitivity (400-650 nm vs 380-620 nm)
    for w_label, cref_col, cdmf_col, w_desc in [
        ("cosine_window_400_650nm", "cos_theta_vs_reference", "cos_theta_vs_dmf_control", "UV-vis cosine similarity evaluated over [400, 650] nm"),
        ("cosine_window_380_620nm", "cos_theta_vs_reference_380_620nm", "cos_theta_vs_dmf_control_380_620nm", "UV-vis cosine similarity evaluated over [380, 620] nm"),
    ]:
        for comp in ["PEAOH", "beta-Ala-H", "beta-Ala-tBu"]:
            base_c = cohort_uv[cohort_uv["composition"] == comp]
            add_uv_row(
                "6_uvvis_cosine_window_sensitivity",
                w_label,
                comp,
                base_c["lambda_exciton_nm"],
                base_c[cref_col],
                base_c[cdmf_col],
                base_c,
                w_desc,
            )

    # 7. RGA organic outgassing and m/z 44 share under both definitions
    for def_name, org_col, share_col, def_desc in [
        (
            "per_run_clip_and_sum_then_mean",
            "ht_organic_outgassing_1e9_torr_mean",
            "delta_mz44_share_of_delta_organic_pct",
            "Definition (i): Clip negative channels to zero and sum m/z 12-160 (excl. 18, 32) per run, then average across runs",
        ),
        (
            "ensemble_mean_spectrum_clip_and_sum",
            "ht_organic_outgassing_from_mean_spec_1e9_torr",
            "delta_mz44_share_from_mean_spec_pct",
            "Definition (ii): Average unclipped background-subtracted spectra across runs first, then clip negative channels to zero and sum m/z 12-160 (excl. 18, 32)",
        ),
    ]:
        for comp in ["PEAOH", "beta-Ala-H", "beta-Ala-tBu"]:
            r_row = rga_sum.loc[comp]
            ht_org = float(r_row[org_col])
            ref_org = float(r_row["reference_organic_outgassing_1e9_torr"])
            sh_44 = float(r_row[share_col])
            cos_rga = float(r_row["cos_theta_20ion_100_120C"])
            rows.append({
                "analysis_group": "7_rga_aggregation_definition",
                "variant": def_name,
                "composition": comp,
                "n_films": int(r_row["n_runs_100_120C"]),
                "lambda_exc_mean_nm_full": "",
                "lambda_exc_sd_nm_full": "",
                "lambda_exc_mean_nm_4dec": "",
                "lambda_exc_sd_nm_4dec": "",
                "lambda_exc_reported_1dec": "",
                "cos_theta_vs_reference_full": f"{cos_rga:.7f}",
                "cos_theta_vs_reference_4dec": rhu(f"{cos_rga:.10f}", "0.0001"),
                "cos_theta_vs_reference_3dec": rhu(f"{cos_rga:.10f}", "0.001"),
                "cos_theta_vs_dmf_control_full": "",
                "cos_theta_vs_dmf_control_4dec": "",
                "cos_theta_vs_dmf_control_3dec": "",
                "delta_lambda_exc_vs_baseline_nm": "",
                "delta_cos_ref_vs_baseline": "",
                "delta_cos_dmf_vs_baseline": "",
                "rga_ht_organic_outgassing_full_1e9_torr": f"{ht_org:.7f}",
                "rga_ht_organic_outgassing_2dec_1e9_torr": rhu(f"{ht_org:.10f}", "0.01"),
                "rga_reference_organic_outgassing_full_1e9_torr": f"{ref_org:.7f}",
                "rga_reference_organic_outgassing_2dec_1e9_torr": rhu(f"{ref_org:.10f}", "0.01"),
                "rga_delta_mz44_share_full_pct": f"{sh_44:.7f}",
                "rga_delta_mz44_share_1dec_pct": rhu(f"{sh_44:.10f}", "0.1"),
                "notes": def_desc,
            })

    return pd.DataFrame(rows)


def build_table_s1(
    cohort_df: pd.DataFrame,
    uv_sum: pd.DataFrame,
    peaks_df: pd.DataFrame,
    one_df: pd.DataFrame,
    rga_sum: pd.DataFrame,
    sens_df: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    def get_peak(comp: str, refl: str, src: str) -> pd.Series:
        sub = peaks_df[
            (peaks_df["composition"] == comp)
            & (peaks_df["reflection"] == refl)
            & (peaks_df["source"] == src)
        ]
        return sub.iloc[0]

    comps = [
        ("PEAOH", "(PEAOH)BiI4 (1:1)", "4.0", "q1", "1D edge-sharing chains", "DMF / DMF"),
        ("beta-Ala-H", "(beta-Ala-H)2BiI5 (2:1)", "5.0", "(002) and (-101)", "1D cis-corner-sharing chains", "DMF / MeOH"),
        ("beta-Ala-tBu", "(beta-Ala-tBu)3Bi2I9 (3:2)", "4.5", "q1 (1,0,-1)", "0D bioctahedral [Bi2I9]^3- dimers", "DMF / MeOH"),
    ]

    table_rows = []
    for comp, label, xm, refl_desc, motif, solv_pair in comps:
        uv_r = uv_sum.loc[comp]
        rga_r = rga_sum.loc[comp]

        sub_coh = cohort_df[(cohort_df["composition"] == comp) & (cohort_df["included"] == "y")]
        sub_all = cohort_df[
            (cohort_df["composition"] == comp)
            & (
                (cohort_df["included"] == "y")
                | (cohort_df["exclusion_reason"].str.startswith("temperature_outside_100_120C"))
            )
        ]
        n_all_temp = len(sub_all)
        n_cohort = len(sub_coh)
        n_uv = int(uv_r["n_films"])
        n_rga = int(rga_r["n_runs_100_120C"])

        if comp in ("PEAOH", "beta-Ala-tBu"):
            p_ht = get_peak(comp, "q1", "HT GIWAXS")
            p_rg = get_peak(comp, "q1", "Reference GIWAXS")
            p_rx = get_peak(comp, "q1", "Reference XRD")
            p_sim = get_peak(comp, "q1", "SCXRD (simulated)")
            one_r = one_df[(one_df["composition"] == comp) & (one_df["reflection"] == "q1")].iloc[0]
            n_giwaxs_pat = int(one_r["n_patterns_all"])
            n_giwaxs_films = int(one_r["n_films_one"])

            q_ht_3 = rhu(p_ht["q_refined_full"], "0.001")
            q_rg_3 = rhu(p_rg["q_refined_full"], "0.001")
            q_rx_3 = rhu(p_rx["q_refined_full"], "0.001")
            q_sim_3 = rhu(p_sim["q_refined_full"], "0.001")

            q_ht_disp = q_ht_3
            q_ht_with_sd_disp = f"{q_ht_3} +/- {rhu(p_ht['film_to_film_SD'], '0.001')}"
            q_ht_7dec = f"{float(p_ht['q_refined_full']):.7f}"
            q_rg_disp = q_rg_3
            q_rg_7dec = f"{float(p_rg['q_refined_full']):.7f}"
            q_rx_disp = q_rx_3
            q_rx_7dec = f"{float(p_rx['q_refined_full']):.7f}"
            q_sim_disp = q_sim_3
            q_sim_7dec = f"{float(p_sim['q_refined_full']):.7f}"

            abs_dq_rg_tab = f"{abs(Decimal(q_ht_3) - Decimal(q_rg_3)):.3f}"
            abs_dq_rx_tab = f"{abs(Decimal(q_ht_3) - Decimal(q_rx_3)):.3f}"
            abs_dq_tab_pair = f"{abs_dq_rg_tab} / {abs_dq_rx_tab}"
            abs_dq_fn_c_pair = ""

            dq_rg_7dec = f"{float(p_ht['q_refined_full']) - float(p_rg['q_refined_full']):+.7f}"
            dq_rx_7dec = f"{float(p_ht['q_refined_full']) - float(p_rx['q_refined_full']):+.7f}"
        else:
            p_ht_1 = get_peak("beta-Ala-H", "(002)", "HT GIWAXS")
            p_ht_2 = get_peak("beta-Ala-H", "(-101)", "HT GIWAXS")
            p_rg_1 = get_peak("beta-Ala-H", "(002)", "Reference GIWAXS")
            p_rg_2 = get_peak("beta-Ala-H", "(-101)", "Reference GIWAXS")
            p_rx_1 = get_peak("beta-Ala-H", "(002)", "Reference XRD")
            p_rx_2 = get_peak("beta-Ala-H", "(-101)", "Reference XRD")
            p_sim_1 = get_peak("beta-Ala-H", "(002)", "SCXRD (simulated)")
            p_sim_2 = get_peak("beta-Ala-H", "(-101)", "SCXRD (simulated)")
            one_r = one_df[(one_df["composition"] == "beta-Ala-H") & (one_df["reflection"] == "(002)")].iloc[0]
            n_giwaxs_pat = int(one_r["n_patterns_all"])
            n_giwaxs_films = int(one_r["n_films_one"])

            q_ht_1_3 = rhu(p_ht_1["q_refined_full"], "0.001")
            q_ht_2_3 = rhu(p_ht_2["q_refined_full"], "0.001")
            q_rg_1_3 = rhu(p_rg_1["q_refined_full"], "0.001")
            q_rg_2_3 = rhu(p_rg_2["q_refined_full"], "0.001")
            q_rx_1_3 = rhu(p_rx_1["q_refined_full"], "0.001")
            q_rx_2_3 = rhu(p_rx_2["q_refined_full"], "0.001")
            q_sim_1_3 = rhu(p_sim_1["q_refined_full"], "0.001")
            q_sim_2_3 = rhu(p_sim_2["q_refined_full"], "0.001")

            q_ht_disp = f"{q_ht_1_3}, {q_ht_2_3}"
            q_ht_with_sd_disp = (
                f"{q_ht_1_3} +/- {rhu(p_ht_1['film_to_film_SD'], '0.001')} / "
                f"{q_ht_2_3} +/- {rhu(p_ht_2['film_to_film_SD'], '0.001')}"
            )
            q_ht_7dec = f"{float(p_ht_1['q_refined_full']):.7f}, {float(p_ht_2['q_refined_full']):.7f}"
            q_rg_disp = f"{q_rg_1_3}, {q_rg_2_3}"
            q_rg_7dec = f"{float(p_rg_1['q_refined_full']):.7f}, {float(p_rg_2['q_refined_full']):.7f}"
            q_rx_disp = f"{q_rx_1_3}, {q_rx_2_3}"
            q_rx_7dec = f"{float(p_rx_1['q_refined_full']):.7f}, {float(p_rx_2['q_refined_full']):.7f}"
            q_sim_disp = f"{q_sim_1_3}, {q_sim_2_3}"
            q_sim_7dec = f"{float(p_sim_1['q_refined_full']):.7f}, {float(p_sim_2['q_refined_full']):.7f}"

            abs_dq_rg_1 = f"{abs(Decimal(q_ht_1_3) - Decimal(q_rg_1_3)):.3f}"
            abs_dq_rx_1 = f"{abs(Decimal(q_ht_1_3) - Decimal(q_rx_1_3)):.3f}"
            abs_dq_rg_2 = f"{abs(Decimal(q_ht_2_3) - Decimal(q_rg_2_3)):.3f}"
            abs_dq_rx_2 = f"{abs(Decimal(q_ht_2_3) - Decimal(q_rx_2_3)):.3f}"
            abs_dq_rg_tab = f"{abs_dq_rg_1} (002), {abs_dq_rg_2} (-101)"
            abs_dq_rx_tab = f"{abs_dq_rx_1} (002), {abs_dq_rx_2} (-101)"
            abs_dq_tab_pair = f"{abs_dq_rg_1} / {abs_dq_rx_1}"
            abs_dq_fn_c_pair = f"{abs_dq_rg_2} / {abs_dq_rx_2}"

            dq_rg_7dec = (
                f"{float(p_ht_1['q_refined_full']) - float(p_rg_1['q_refined_full']):+.7f}, "
                f"{float(p_ht_2['q_refined_full']) - float(p_rg_2['q_refined_full']):+.7f}"
            )
            dq_rx_7dec = (
                f"{float(p_ht_1['q_refined_full']) - float(p_rx_1['q_refined_full']):+.7f}, "
                f"{float(p_ht_2['q_refined_full']) - float(p_rx_2['q_refined_full']):+.7f}"
            )

        lam_m_full = f"{float(uv_r['ht_lambda_exc_mean_nm']):.7f}"
        lam_sd_full = f"{float(uv_r['ht_lambda_exc_sd_nm']):.7f}"
        lam_m_1d = rhu(lam_m_full, "0.1")
        lam_sd_1d = rhu(lam_sd_full, "0.1")

        ref_lam_full = f"{float(uv_r['reference_lambda_exc_nm']):.7f}"
        dmf_lam_full = f"{float(uv_r['dmf_control_lambda_exc_nm']):.7f}"
        cref_full = f"{float(uv_r['cos_theta_vs_reference_mean']):.7f}"
        cdmf_full = f"{float(uv_r['cos_theta_vs_dmf_control_mean']):.7f}"

        ht_org_full = f"{float(rga_r['ht_organic_outgassing_1e9_torr_mean']):.7f}"
        ref_org_full = f"{float(rga_r['reference_organic_outgassing_1e9_torr']):.7f}"
        rga_cos_full = f"{float(rga_r['cos_theta_20ion_100_120C']):.7f}"

        table_rows.append({
            "composition_key": comp,
            "composition_label": label,
            "nominal_X_over_M": xm,
            "casting_solvent_ht_over_reference": solv_pair,
            "structural_motif": motif,
            "n_all_temp_films": n_all_temp,
            "n_cohort_100_120C_films": n_cohort,
            "n_uvvis_films": n_uv,
            "n_giwaxs_patterns": n_giwaxs_pat,
            "n_giwaxs_films": n_giwaxs_films,
            "n_rga_runs": n_rga,
            "uvvis_ht_lambda_exc_display_nm": f"{lam_m_1d} +/- {lam_sd_1d}",
            "uvvis_ht_lambda_exc_mean_unrounded_nm": lam_m_full,
            "uvvis_ht_lambda_exc_sd_unrounded_nm": lam_sd_full,
            "uvvis_reference_lambda_exc_nm": rhu(ref_lam_full, "0.1"),
            "uvvis_reference_lambda_exc_unrounded_nm": ref_lam_full,
            "uvvis_dmf_control_lambda_exc_nm": rhu(dmf_lam_full, "0.1"),
            "uvvis_dmf_control_lambda_exc_unrounded_nm": dmf_lam_full,
            "uvvis_cos_theta_vs_reference_display": rhu(cref_full, "0.001"),
            "uvvis_cos_theta_vs_reference_unrounded": cref_full,
            "uvvis_cos_theta_vs_dmf_control_display": rhu(cdmf_full, "0.001"),
            "uvvis_cos_theta_vs_dmf_control_unrounded": cdmf_full,
            "diffraction_reflection": refl_desc,
            "giwaxs_ht_q_display_invA": q_ht_disp,
            "giwaxs_ht_q_with_sd_display_invA": q_ht_with_sd_disp,
            "giwaxs_ht_q_unrounded_invA": q_ht_7dec,
            "giwaxs_reference_q_display_invA": q_rg_disp,
            "giwaxs_reference_q_unrounded_invA": q_rg_7dec,
            "xrd_reference_q_display_invA": q_rx_disp,
            "xrd_reference_q_unrounded_invA": q_rx_7dec,
            "scxrd_cif_q_display_invA": q_sim_disp,
            "scxrd_cif_q_unrounded_invA": q_sim_7dec,
            "abs_delta_q1_vs_reference_giwaxs_and_xrd_display_invA": abs_dq_tab_pair,
            "abs_delta_q1_footnote_c_display_invA": abs_dq_fn_c_pair,
            "abs_delta_q1_vs_reference_giwaxs_from_tabulated_invA": abs_dq_rg_tab,
            "abs_delta_q1_vs_reference_xrd_from_tabulated_invA": abs_dq_rx_tab,
            "delta_q_ht_minus_reference_giwaxs_unrounded_invA": dq_rg_7dec,
            "delta_q_ht_minus_reference_xrd_unrounded_invA": dq_rx_7dec,
            "rga_ht_organic_outgassing_display_1e9_torr": rhu(ht_org_full, "0.01"),
            "rga_ht_organic_outgassing_unrounded_1e9_torr": ht_org_full,
            "rga_reference_organic_outgassing_display_1e9_torr": rhu(ref_org_full, "0.01"),
            "rga_reference_organic_outgassing_unrounded_1e9_torr": ref_org_full,
            "rga_cos_theta_20ion_display": rhu(rga_cos_full, "0.001"),
            "rga_cos_theta_20ion_unrounded": rga_cos_full,
        })

    table_df = pd.DataFrame(table_rows)
    excl_326 = sens_df[sens_df["variant"] == "beta_Ala_H_excluding_TF000326"].iloc[0]
    r0, r1, r2 = table_rows[0], table_rows[1], table_rows[2]

    tex_lines = [
        "% Auto-generated by scripts/plot_figure_s14.py from data/figs14_*.csv",
        "\\begin{table*}[!htbp]",
        "\\caption{\\textbf{Quantitative reproducibility summary comparing high-throughput (HT) films (annealed at 100 or 120~$^\\circ$C) with manually prepared reference and control films.} $X/M$, halide-to-metal ratio; $\\lambda_{\\mathrm{exc}}$, excitonic absorption maximum; $q_1$, position of the primary low-$q$ reflection(s). HT UV-vis values are from robotic UV-vis line scans; reference (cast from each cation's optimized solvent) and DMF control UV-vis values are from a Cary~5000. Uncertainties are $\\pm$1$\\sigma$ across films. UV-vis cosine similarities ($\\cos\\theta$) are means of per-film values relative to the indicated spectrum. Peak positions were refined by parabolic interpolation of unsmoothed mean profiles on a 0.005~\\AA$^{-1}$ grid; $|\\Delta q_1|$ values are differences of the tabulated positions. The film-to-film standard deviation of HT peak positions is 0.0002--0.002~\\AA$^{-1}$ for PEAOH, $\\beta$-Ala-$^t$Bu, and the $\\beta$-Ala-H low-$q$ reflection, and 0.004--0.005~\\AA$^{-1}$ for the $\\beta$-Ala-H (002) and ($\\bar{1}$01) reflections within the primary reflection windows (or 0.008--0.010~\\AA$^{-1}$ across the wider doublet window when two films carrying a shoulder near 0.568~\\AA$^{-1}$ are included); including the sensitivity to the refinement method (about 0.0015~\\AA$^{-1}$), differences of 0.002~\\AA$^{-1}$ or less are within uncertainty. Organic outgassing is the background-subtracted net partial pressure, time-averaged over the exposure window and summed per run over $m/z$~=~12--160 (clipped at zero), excluding $m/z$~=~18 and 32, and averaged across runs where $N_{\\mathrm{HT}} > 1$. The 20-ion set is defined in the text. The PEAOH HT RGA values are from a single 150~s run; an excluded 300~s commissioning run at 100~$^\\circ$C (TF000044) gives 0.32~$\\times$~10$^{-9}$~Torr and $\\cos\\theta$~=~0.985.}",
        "\\label{tab:si_reproducibility_summary}",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lccc}",
        "\\toprule",
        "Modality / metric & (PEAOH)BiI$_4$ & ($\\beta$-Ala-H)$_2$BiI$_5$ & ($\\beta$-Ala-$^t$Bu)$_3$Bi$_2$I$_9$ \\\\",
        " & (1:1, $X/M$~=~4.0) & (2:1, $X/M$~=~5.0) & (3:2, $X/M$~=~4.5) \\\\",
        "\\midrule",
        f"Casting solvent (HT / reference) & {r0['casting_solvent_ht_over_reference']} & {r1['casting_solvent_ht_over_reference']} & {r2['casting_solvent_ht_over_reference']} \\\\",
        "\\multicolumn{4}{@{}l}{\\textit{$N_\\mathrm{HT}$}} \\\\",
        f"\\quad UV-vis films & {r0['n_uvvis_films']} & {r1['n_uvvis_films']} & {r2['n_uvvis_films']} \\\\",
        f"\\quad GIWAXS patterns (films) & {r0['n_giwaxs_patterns']} ({r0['n_giwaxs_films']}) & {r1['n_giwaxs_patterns']} ({r1['n_giwaxs_films']}) & {r2['n_giwaxs_patterns']} ({r2['n_giwaxs_films']}) \\\\",
        f"\\quad RGA runs & {r0['n_rga_runs']} & {r1['n_rga_runs']} & {r2['n_rga_runs']} \\\\",
        "\\midrule",
        "\\multicolumn{4}{@{}l}{\\textit{UV-vis $\\lambda_{\\mathrm{exc}}$ (nm)}} \\\\",
        f"\\quad HT & {r0['uvvis_ht_lambda_exc_display_nm'].replace(' +/- ', '~\\pm~')} & {r1['uvvis_ht_lambda_exc_display_nm'].replace(' +/- ', '~\\pm~')}$^{{b}}$ & {r2['uvvis_ht_lambda_exc_display_nm'].replace(' +/- ', '~\\pm~')} \\\\",
        f"\\quad Reference & {r0['uvvis_reference_lambda_exc_nm']} (DMF) & {r1['uvvis_reference_lambda_exc_nm']} (MeOH) & {r2['uvvis_reference_lambda_exc_nm']} (MeOH) \\\\",
        f"\\quad DMF control & {r0['uvvis_dmf_control_lambda_exc_nm']} (DMF) & {r1['uvvis_dmf_control_lambda_exc_nm']} (DMF) & {r2['uvvis_dmf_control_lambda_exc_nm']} (DMF) \\\\",
        f"\\quad $\\cos\\theta$ vs.\\ reference / DMF control & {r0['uvvis_cos_theta_vs_reference_display']} / {r0['uvvis_cos_theta_vs_dmf_control_display']} & {r1['uvvis_cos_theta_vs_reference_display']} / {r1['uvvis_cos_theta_vs_dmf_control_display']} & {r2['uvvis_cos_theta_vs_reference_display']} / {r2['uvvis_cos_theta_vs_dmf_control_display']} \\\\",
        "\\midrule",
        "\\multicolumn{4}{@{}l}{\\textit{Primary $q_1$ (\\AA$^{-1}$)}} \\\\",
        f"\\quad HT GIWAXS (ITO) & {r0['giwaxs_ht_q_display_invA']} & {r1['giwaxs_ht_q_display_invA']}$^{{a}}$ & {r2['giwaxs_ht_q_display_invA']} \\\\",
        f"\\quad Reference GIWAXS (Si) & {r0['giwaxs_reference_q_display_invA']} & {r1['giwaxs_reference_q_display_invA']} & {r2['giwaxs_reference_q_display_invA']} \\\\",
        f"\\quad Reference XRD (Co~K$\\alpha$) & {r0['xrd_reference_q_display_invA']} & {r1['xrd_reference_q_display_invA']} & {r2['xrd_reference_q_display_invA']} \\\\",
        f"\\quad SCXRD (simulated)$^{{d}}$ & {r0['scxrd_cif_q_display_invA']} & {r1['scxrd_cif_q_display_invA']} & {r2['scxrd_cif_q_display_invA']} \\\\",
        f"\\quad $|\\Delta q_1|$ vs.\\ reference GIWAXS / XRD & {r0['abs_delta_q1_vs_reference_giwaxs_and_xrd_display_invA']} & {r1['abs_delta_q1_vs_reference_giwaxs_and_xrd_display_invA']}$^{{c}}$ & {r2['abs_delta_q1_vs_reference_giwaxs_and_xrd_display_invA']} \\\\",
        "\\midrule",
        "\\multicolumn{4}{@{}l}{\\textit{EUV-RGA organic outgassing (10$^{-9}$~Torr)}} \\\\",
        f"\\quad HT & {r0['rga_ht_organic_outgassing_display_1e9_torr']} & {r1['rga_ht_organic_outgassing_display_1e9_torr']} & {r2['rga_ht_organic_outgassing_display_1e9_torr']} \\\\",
        f"\\quad Reference & {r0['rga_reference_organic_outgassing_display_1e9_torr']} & {r1['rga_reference_organic_outgassing_display_1e9_torr']} & {r2['rga_reference_organic_outgassing_display_1e9_torr']} \\\\",
        "\\multicolumn{4}{@{}l}{\\textit{EUV-RGA spectral similarity}} \\\\",
        f"\\quad 20-ion $\\cos\\theta$ (HT vs.\\ reference) & {r0['rga_cos_theta_20ion_display']} & {r1['rga_cos_theta_20ion_display']} & {r2['rga_cos_theta_20ion_display']} \\\\",
        "\\bottomrule",
        "\\end{tabular*}",
        "",
        "\\vspace{2pt}",
        "\\begin{flushleft}",
        "\\footnotesize $^{a}$ In addition to the (002) and ($\\bar{1}$01) reflections at $q$~=~0.552 and 0.626~\\AA$^{-1}$, HT ($\\beta$-Ala-H)$_2$BiI$_5$ films exhibit a low-$q$ lamellar reflection at $q$~=~0.416~\\AA$^{-1}$ ($d$~=~15.1~\\AA) arising from a competing DMF-promoted phase.\\\\",
        f"$^{{b}}$ {excl_326['lambda_exc_reported_1dec'].replace(' +/- ', '~\\pm~')}~nm ($N$~=~17) excluding TF000326; six of the 18 films show an additional shoulder near 494~nm.\\\\",
        f"$^{{c}}$ Values for the (002) reflection; for the ($\\bar{{1}}$01) reflection, {r1['abs_delta_q1_footnote_c_display_invA']}~\\AA$^{{-1}}$.\\\\",
        "$^{d}$ Simulated from SCXRD structures collected at 100~K (PEAOH), 298~K ($\\beta$-Ala-H), and 273~K ($\\beta$-Ala-$^t$Bu). The higher $q$ of the simulated PEAOH reflection is consistent with thermal contraction at 100~K. For $\\beta$-Ala-$^t$Bu, the thin-film reflection corresponds to (10$\\bar{1}$), whose spacing is about 2\\% smaller in both HT and reference films than in the single crystal; the strongest simulated reflection, (100) at 0.415~\\AA$^{-1}$, is absent from the film patterns because of strong out-of-plane preferred orientation.",
        "\\end{flushleft}",
        "\\end{table*}",
    ]
    return table_df, "\n".join(tex_lines) + "\n"


def build_text_values(
    cohort_df: pd.DataFrame,
    uv_scans: pd.DataFrame,
    uv_sum: pd.DataFrame,
    diff_df: pd.DataFrame,
    peaks_df: pd.DataFrame,
    one_df: pd.DataFrame,
    sim_peaks_df: pd.DataFrame,
    rga_runs: pd.DataFrame,
    rga_sum: pd.DataFrame,
    sens_df: pd.DataFrame,
    table_df_raw: pd.DataFrame,
) -> pd.DataFrame:
    table_df = table_df_raw.set_index("composition_key")

    def get_peak(comp: str, refl: str, src: str) -> pd.Series:
        sub = peaks_df[
            (peaks_df["composition"] == comp)
            & (peaks_df["reflection"] == refl)
            & (peaks_df["source"] == src)
        ]
        return sub.iloc[0]

    rows = []

    def add_val(val_full: str, val_round: str, src_col: str, defn: str) -> None:
        rows.append({
            "value_full_precision": val_full,
            "rounded_as_printed": val_round,
            "source_script": "scripts/plot_figure_s14.py",
            "source_file_and_column": src_col,
            "definition": defn,
        })

    # 1. Cohort counts
    n_all_83 = len(cohort_df)
    n_all_82 = int(table_df["n_all_temp_films"].sum())
    n_coh_41 = int(table_df["n_cohort_100_120C_films"].sum())
    coh_41_df = cohort_df[cohort_df["included"] == "y"]
    n_coh_strict_34 = int((coh_41_df["temperature_source_field"] == "heater_sv_temp").sum())
    pv_7_df = coh_41_df[coh_41_df["temperature_source_field"].str.startswith("heater_pv_temp")]
    n_pv_7 = len(pv_7_df)
    n_dec2025_13 = int((cohort_df["substrate"] == "Sciprios-Glass").sum())

    uv_39_df = uv_scans[uv_scans["included_in_uvvis_cohort"] == True]
    n_uv_39 = len(uv_39_df)
    n_giw_pat_17 = int(table_df["n_giwaxs_patterns"].sum())
    n_giw_films_12 = int(table_df["n_giwaxs_films"].sum())
    n_rga_7 = int(table_df["n_rga_runs"].sum())

    giw_mfids = set()
    for col in diff_df.columns:
        if col.startswith("HT_scan_"):
            for part in col.split("_"):
                if part.startswith("TF"):
                    giw_mfids.add(part)
    rga_7_df = rga_runs[rga_runs["included_in_100_120C_cohort"] == True]
    measured_mfids = set(uv_39_df["mfid"]).union(giw_mfids).union(set(rga_7_df["mfid"]))
    unmeasured_mfids = sorted(set(coh_41_df["mfid"]) - measured_mfids)
    n_contributing_films = len(measured_mfids)

    add_val(str(n_all_83), str(n_all_83), "data/figs14_cohort_films.csv", "[cohort_all_temps_including_TF000004] Total BiI3 target-stoichiometry HT films across all annealing temperatures including commissioning film TF000004 (34 PEAOH + 39 beta-Ala-H + 10 beta-Ala-tBu = 83)")
    add_val(str(n_all_82), str(n_all_82), "figures/table_s1.csv:n_all_temp_films (sum)", "[cohort_all_temps_excluding_TF000004] Total BiI3 target-stoichiometry HT films across all annealing temperatures excluding commissioning film TF000004 (33 PEAOH + 39 beta-Ala-H + 10 beta-Ala-tBu = 82)")
    add_val(str(n_coh_41), str(n_coh_41), "figures/table_s1.csv:n_cohort_100_120C_films (sum)", "[cohort_100_120C_total_films] Total HT benchmark cohort films annealed at 100 or 120 deg C (18 PEAOH + 19 beta-Ala-H + 4 beta-Ala-tBu = 41)")
    add_val(str(n_contributing_films), str(n_contributing_films), "data/figs14_cohort_films.csv & data/figs14_uvvis_scans.csv", f"[cohort_100_120C_contributing_films] Cohort films contributing >= 1 quality-controlled measurement across UV-vis, GIWAXS, and RGA ({n_contributing_films} of {n_coh_41}; excludes {','.join(unmeasured_mfids)} which fail the UV-vis absorbance threshold >= 0.03 OD and have no GIWAXS or RGA scans)")
    add_val(str(n_coh_strict_34), str(n_coh_strict_34), "data/figs14_cohort_films.csv:temperature_source_field=='heater_sv_temp'", "[cohort_strict_heater_sv_temp_films] Cohort films with non-null heater_sv_temp in {100, 120} deg C (13 PEAOH + 17 beta-Ala-H + 4 beta-Ala-tBu = 34 ITO films)")
    add_val(str(n_pv_7), str(n_pv_7), "data/figs14_cohort_films.csv:temperature_source_field", f"[cohort_dec2025_pv_temp_films] Dec 2025 Sciprios-Glass cohort films included via logged heater_pv_temp='100C' ({';'.join(pv_7_df['mfid'].tolist())})")
    add_val(str(n_dec2025_13), str(n_dec2025_13), "data/figs14_cohort_films.csv:substrate=='Sciprios-Glass'", "[cohort_dec2025_null_annealing_temp_films] Total Dec 2025 Sciprios-Glass BiI3 target-stoichiometry films where both heater_sv_temp and annealing_temp_c are null (TF000004 through TF000118)")
    add_val(str(n_uv_39), str(n_uv_39), "figures/table_s1.csv:n_uvvis_films (sum)", "[n_ht_uvvis_total] Total valid as-cast HT UV-vis scans in 100 or 120 deg C cohort (17 + 18 + 4 = 39)")
    add_val(str(n_giw_pat_17), str(n_giw_pat_17), "figures/table_s1.csv:n_giwaxs_patterns (sum)", "[n_ht_giwaxs_patterns_total] Total unflagged HT GIWAXS patterns in 100 or 120 deg C cohort (4 + 9 + 4 = 17 patterns)")
    add_val(str(n_giw_films_12), str(n_giw_films_12), "figures/table_s1.csv:n_giwaxs_films (sum)", "[n_ht_giwaxs_films_total] Total unique HT GIWAXS films in 100 or 120 deg C cohort (2 + 8 + 2 = 12 films)")
    add_val(str(n_rga_7), str(n_rga_7), "figures/table_s1.csv:n_rga_runs (sum)", "[n_ht_rga_runs_total] Total primary 150 s HT EUV-RGA runs in 100 or 120 deg C cohort (1 + 4 + 2 = 7 runs)")

    # 2. UV-vis text values
    n_ito_32 = int(uv_sum["n_films_8of10_positions"].sum())
    n_glass_7 = int(uv_sum["n_films_3of5_positions"].sum())
    add_val(str(n_ito_32), str(n_ito_32), "data/figs14_uvvis_scans.csv:n_retained_positions==8", "[uvvis_n_ito_films] Number of HT UV-vis cohort films on ITO (central 8 of 10 positions retained: 12 PEAOH + 16 beta-Ala-H + 4 beta-Ala-tBu = 32)")
    add_val(str(n_glass_7), str(n_glass_7), "data/figs14_uvvis_scans.csv:n_retained_positions==3", "[uvvis_n_glass_films] Number of HT UV-vis cohort films on glass (central 3 of 5 positions retained: 5 PEAOH + 2 beta-Ala-H + 0 beta-Ala-tBu = 7)")

    n_pmma_uv_14 = int((uv_39_df["pmma_coated"] == "y").sum())
    add_val(f"{n_pmma_uv_14} / {n_uv_39}", "14 of 39", "data/figs14_uvvis_scans.csv:pmma_coated=='y'", "[uvvis_pmma_coated_count] Number of valid HT UV-vis cohort films carrying a PMMA overcoat (5 PEAOH + 7 beta-Ala-H + 2 beta-Ala-tBu = 14 of 39; 25 uncoated)")
    add_val("0 / 7", "0 of 7", "data/figs14_cohort_films.csv:pmma_coated", "[rga_pmma_coated_count] Number of primary HT RGA cohort films carrying a PMMA overcoat (0 of 7)")

    dlam_peaoh = float(uv_sum.loc["PEAOH", "ht_lambda_exc_mean_nm"]) - float(uv_sum.loc["PEAOH", "reference_lambda_exc_nm"])
    dlam_tbu = float(uv_sum.loc["beta-Ala-tBu", "ht_lambda_exc_mean_nm"]) - float(uv_sum.loc["beta-Ala-tBu", "reference_lambda_exc_nm"])
    add_val(f"{dlam_peaoh:.7f} / {dlam_tbu:.7f}", "3", "figures/table_s1.csv:uvvis_ht_lambda_exc_mean_unrounded_nm", "[uvvis_ht_minus_ref_peaoh_tbu_nm] HT minus Reference excitonic peak wavelength offset for PEAOH (+1.9411765 nm) and beta-Ala-tBu (+3.0000000 nm), both within 3 nm")

    dlam_peaoh_ctrl = float(uv_sum.loc["PEAOH", "dmf_control_lambda_exc_nm"]) - float(uv_sum.loc["PEAOH", "reference_lambda_exc_nm"])
    add_val(f"{dlam_peaoh_ctrl:.7f}", "4", "figures/table_s1.csv:uvvis_dmf_control_lambda_exc_unrounded_nm", "[uvvis_peaoh_dmf_ctrl_minus_ref_nm] Difference in excitonic maximum between PEAOH DMF control (526.0 nm) and PEAOH Reference (522.0 nm)")

    s_no326 = sens_df[sens_df["variant"] == "beta_Ala_H_excluding_TF000326"].iloc[0]
    add_val(
        f"{s_no326['lambda_exc_mean_nm_full']} +/- {s_no326['lambda_exc_sd_nm_full']}",
        s_no326["lambda_exc_reported_1dec"],
        "figures/figs14_sensitivity.csv:variant=='beta_Ala_H_excluding_TF000326'",
        "[uvvis_beta_ala_h_excl_TF000326_lambda_exc] HT beta-Ala-H excitonic peak mean +/- SD excluding TF000326 (N=17: 514.1764706 +/- 6.4443318 nm -> 514.2 +/- 6.4 nm; cos_theta = 0.955 vs Reference, 0.940 vs DMF control)",
    )
    tf326_row = uv_scans[uv_scans["mfid"] == "TF000326"].iloc[0]
    add_val(
        f"{float(tf326_row['lambda_exciton_nm']):.7f}",
        rhu(f"{float(tf326_row['lambda_exciton_nm']):.7f}", "1"),
        "data/figs14_uvvis_scans.csv:mfid=='TF000326'",
        f"[uvvis_TF000326_lambda_exc_nm] Excitonic local-maximum wavelength of optically thick PMMA-coated beta-Ala-H film TF000326 (has_local_max={tf326_row['has_local_max']}, lambda_peak_max_nm=472.0000000 nm, lambda_d2_min_nm={float(tf326_row['lambda_d2_min_nm']):.7f} nm)",
    )

    bala_uv18 = uv_39_df[uv_39_df["composition"] == "beta-Ala-H"].sort_values("mfid")
    sh_494_df = bala_uv18[(bala_uv18["lambda_d2_min_nm"] >= 492.0) & (bala_uv18["lambda_d2_min_nm"] <= 496.0)]
    sh_details = "; ".join([f"{r['mfid']}({r['lambda_d2_min_nm']:.1f} nm)" for _, r in sh_494_df.iterrows()])
    add_val(
        f"{len(sh_494_df)} / {len(bala_uv18)}",
        f"{len(sh_494_df)} of {len(bala_uv18)}",
        "data/figs14_uvvis_scans.csv:lambda_d2_min_nm in [492, 496]",
        f"[uvvis_beta_ala_h_494nm_shoulder_count] Number of HT beta-Ala-H UV-vis cohort films with primary 2nd-derivative minimum in 492-496 nm ({len(sh_494_df)} of {len(bala_uv18)} films, all at 493.0-494.0 nm: {sh_details})",
    )

    # 3. Diffraction text values
    one_prim = one_df[one_df["reflection"].isin(["q1", "(002)", "(-101)"])].copy()
    one_prim["abs_shift_float"] = one_prim["abs_shift_full_invA"].astype(float)
    max_one_row = one_prim.loc[one_prim["abs_shift_float"].idxmax()]
    max_one_val = float(max_one_row["abs_shift_full_invA"])
    add_val(
        f"{max_one_val:.7f}",
        rhu(f"{max_one_val:.10f}", "0.0001"),
        "data/figs14_diffraction_profiles.csv (one-pattern-per-film sensitivity)",
        f"[giwaxs_one_pattern_per_film_max_shift_invA] Maximum shift in primary reflection position when selecting one pattern per film closest to 0.3 deg incidence angle ({max_one_row['composition']} {max_one_row['reflection']}: {max_one_row['all_patterns_q_full']} -> {max_one_row['one_per_film_q_full']} A^-1, |Delta q| = {max_one_val:.7f} -> 0.0014 A^-1)",
    )

    ht_xrd_diffs = []
    for comp_k, r_k in [("PEAOH", "q1"), ("beta-Ala-H", "(002)"), ("beta-Ala-H", "(-101)"), ("beta-Ala-tBu", "q1")]:
        p_h = get_peak(comp_k, r_k, "HT GIWAXS")
        p_x = get_peak(comp_k, r_k, "Reference XRD")
        ht_xrd_diffs.append(abs(float(p_h["q_refined_full"]) - float(p_x["q_refined_full"])))
    max_ht_xrd_diff = max(ht_xrd_diffs)
    add_val(
        f"{max_ht_xrd_diff:.7f}",
        rhu(f"{max_ht_xrd_diff:.10f}", "0.001"),
        "figures/figs14_peaks_all.csv:q_refined_full (|HT GIWAXS - Reference XRD|)",
        f"[diffraction_ht_vs_ref_xrd_max_diff_invA] Maximum |Delta q1| between HT GIWAXS and Reference XRD across all four primary reflections (unrounded max = {max_ht_xrd_diff:.7f} -> 0.002 A^-1; from rounded tabulated values: 0.001, 0.002, 0.001, 0.000 A^-1, max = 0.002 A^-1)",
    )

    p_b1_rg = get_peak("beta-Ala-H", "(002)", "Reference GIWAXS")
    p_b1_rx = get_peak("beta-Ala-H", "(002)", "Reference XRD")
    p_b2_rg = get_peak("beta-Ala-H", "(-101)", "Reference GIWAXS")
    p_b2_rx = get_peak("beta-Ala-H", "(-101)", "Reference XRD")
    off_b1_full = float(p_b1_rx["q_refined_full"]) - float(p_b1_rg["q_refined_full"])
    off_b2_full = float(p_b2_rx["q_refined_full"]) - float(p_b2_rg["q_refined_full"])
    off_b1_tab = Decimal(p_b1_rx["q_3dec"]) - Decimal(p_b1_rg["q_3dec"])
    off_b2_tab = Decimal(p_b2_rx["q_3dec"]) - Decimal(p_b2_rg["q_3dec"])
    add_val(
        f"{off_b1_full:.7f} to {off_b2_full:.7f}",
        f"{rhu(f'{off_b1_full:.10f}', '0.001')}--{rhu(f'{off_b2_full:.10f}', '0.001')} (unrounded) / {off_b1_tab:.3f}--{off_b2_tab:.3f} (from tabulated)",
        "figures/figs14_peaks_all.csv:q_refined_full & q_3dec (beta-Ala-H Reference XRD minus Reference GIWAXS)",
        f"[diffraction_beta_ala_h_ref_giwaxs_vs_xrd_offset_invA] Offset of Reference GIWAXS to lower q relative to Reference XRD for beta-Ala-H: (002) = {off_b1_full:.7f} A^-1 (tabulated 0.003) and (-101) = {off_b2_full:.7f} A^-1 (tabulated 0.008)",
    )

    p_p_rg = get_peak("PEAOH", "q1", "Reference GIWAXS")
    p_p_rx = get_peak("PEAOH", "q1", "Reference XRD")
    p_t_rg = get_peak("beta-Ala-tBu", "q1", "Reference GIWAXS")
    p_t_rx = get_peak("beta-Ala-tBu", "q1", "Reference XRD")
    diff_p_rg_rx = abs(float(p_p_rg["q_refined_full"]) - float(p_p_rx["q_refined_full"]))
    diff_t_rg_rx = abs(float(p_t_rg["q_refined_full"]) - float(p_t_rx["q_refined_full"]))
    max_pt_rg_rx = max(diff_p_rg_rx, diff_t_rg_rx)
    add_val(
        f"{diff_p_rg_rx:.7f} (PEAOH) / {diff_t_rg_rx:.7f} (beta-Ala-tBu)",
        rhu(f"{max_pt_rg_rx:.10f}", "0.001"),
        "figures/figs14_peaks_all.csv:q_refined_full (|Reference GIWAXS - Reference XRD|)",
        f"[diffraction_peaoh_tbu_ref_giwaxs_vs_xrd_agreement_invA] Agreement between Reference GIWAXS and Reference XRD for PEAOH ({diff_p_rg_rx:.7f} A^-1) and beta-Ala-tBu ({diff_t_rg_rx:.7f} A^-1)",
    )

    q_max_xrd = float(p_b2_rx["q_refined_full"])
    dq_ka2_max = q_max_xrd * (1.0 - XRD_CO_KA1_A / XRD_CO_KA_WEIGHTED_A)
    add_val(
        f"{dq_ka2_max:.7f}",
        "<0.001",
        "scripts/plot_figure_s14.py:XRD_CO_KA1_A (1.78897 vs weighted 1.79026 A)",
        f"[xrd_co_ka2_max_shift_invA] Maximum shift in Reference XRD primary reflection q between Co Ka1 (1.78897 A) and weighted Co Ka1,2 (1.79026 A): {dq_ka2_max:.7f} A^-1 (< 0.001 A^-1)",
    )

    p_p_ht = get_peak("PEAOH", "q1", "HT GIWAXS")
    p_t_ht = get_peak("beta-Ala-tBu", "q1", "HT GIWAXS")
    p_blow_ht = get_peak("beta-Ala-H", "low-q", "HT GIWAXS")
    p_b1_ht = get_peak("beta-Ala-H", "(002)", "HT GIWAXS")
    p_b2_ht = get_peak("beta-Ala-H", "(-101)", "HT GIWAXS")

    add_val(
        f"{float(p_t_ht['film_to_film_SD']):.7f} / {float(p_p_ht['film_to_film_SD']):.7f} / {float(p_blow_ht['film_to_film_SD']):.7f}",
        f"{p_t_ht['film_to_film_SD_3dec']}--{p_blow_ht['film_to_film_SD_3dec']} (per-film) / 0.001--0.002 (per-pattern)",
        "figures/figs14_peaks_all.csv:film_to_film_SD & pattern_to_pattern_SD",
        f"[giwaxs_film_to_film_sd_single_peaks_invA] Film-to-film SD of per-film parabolic refined peak positions: beta-Ala-tBu q1 = {float(p_t_ht['film_to_film_SD']):.7f}, PEAOH q1 = {float(p_p_ht['film_to_film_SD']):.7f}, beta-Ala-H low-q = {float(p_blow_ht['film_to_film_SD']):.7f} A^-1",
    )
    add_val(
        f"{float(p_b1_ht['film_to_film_SD_narrow']):.7f} to {float(p_b2_ht['film_to_film_SD_narrow']):.7f}",
        f"{p_b1_ht['film_to_film_SD_narrow_3dec']}--{p_b2_ht['film_to_film_SD_narrow_3dec']}",
        "figures/figs14_peaks_all.csv:film_to_film_SD_narrow (beta-Ala-H (002) and (-101))",
        f"[giwaxs_film_to_film_sd_beta_ala_h_narrow_invA] Film-to-film SD of per-film parabolic refined positions for beta-Ala-H (002) and (-101) on narrow windows [0.542, 0.560] and [0.615, 0.635] A^-1",
    )
    add_val(
        f"{float(p_b1_ht['film_to_film_SD']):.7f} to {float(p_b2_ht['film_to_film_SD']):.7f}",
        f"{p_b1_ht['film_to_film_SD_3dec']}--{p_b2_ht['film_to_film_SD_3dec']}",
        "figures/figs14_peaks_all.csv:film_to_film_SD (beta-Ala-H (002) and (-101) wide windows)",
        f"[giwaxs_film_to_film_sd_beta_ala_h_wide_invA] Film-to-film SD of per-film parabolic refined positions for beta-Ala-H (002) and (-101) on wide windows [0.535, 0.575] and [0.605, 0.650] A^-1 (includes ~0.568 A^-1 shoulder on TF004966 and TF004968)",
    )

    prim_exp = peaks_df[
        peaks_df["reflection"].isin(["q1", "low-q", "(002)", "(-101)"])
        & peaks_df["source"].isin(["HT GIWAXS", "Reference GIWAXS", "Reference XRD"])
        & (peaks_df["q_bin_center"] != "N/A")
    ].copy()
    bin_ref_diffs = np.abs(prim_exp["q_refined_full"].astype(float) - prim_exp["q_bin_center"].astype(float))
    rms_ref_sens = float(np.sqrt(np.mean(bin_ref_diffs ** 2)))
    add_val(
        f"{rms_ref_sens:.7f}",
        rhu(f"{rms_ref_sens:.10f}", "0.0001"),
        "figures/figs14_peaks_all.csv:|q_refined_full - q_bin_center|",
        f"[diffraction_refinement_sensitivity_invA] Root-mean-square sensitivity of primary peak positions to peak refinement method (parabolic vertex vs. 0.005 A^-1 bin center: RMS = {rms_ref_sens:.7f} -> 0.0015 A^-1)",
    )

    q_low_full = float(p_blow_ht["q_refined_full"])
    d_low_full = float(p_blow_ht["d_spacing_A_from_full"])
    add_val(
        f"{q_low_full:.7f}",
        p_blow_ht["q_3dec"],
        "figures/figs14_peaks_all.csv:q_refined_full (beta-Ala-H, low-q, HT GIWAXS)",
        "[giwaxs_beta_ala_h_low_q_invA] Position of low-q lamellar reflection in HT (beta-Ala-H)2BiI5 films cast from DMF",
    )
    add_val(
        f"{d_low_full:.7f}",
        rhu(f"{d_low_full:.10f}", "0.1"),
        "figures/figs14_peaks_all.csv:d_spacing_A_from_full (beta-Ala-H, low-q, HT GIWAXS)",
        f"[giwaxs_beta_ala_h_low_q_d_A] Lamellar d-spacing of low-q reflection in HT (beta-Ala-H)2BiI5 films ({d_low_full:.7f} A -> 15.1 A)",
    )

    tbu_sim = sim_peaks_df[sim_peaks_df["composition"] == "beta-Ala-tBu"].set_index("hkl")
    r_10m1 = tbu_sim.loc["(1,0,-1)"]
    r_100 = tbu_sim.loc["(1,0,0)"]
    d_cif_exact = float(r_10m1["d_exact_A"])
    d_ht_tbu = float(p_t_ht["d_spacing_A_from_full"])
    p_t_sim = get_peak("beta-Ala-tBu", "q1", "SCXRD (simulated)")
    d_cif_broad = float(p_t_sim["d_spacing_A_from_full"])
    contr_exact_pct = 100.0 * (d_cif_exact - d_ht_tbu) / d_cif_exact
    contr_broad_pct = 100.0 * (d_cif_broad - d_ht_tbu) / d_cif_broad
    add_val(
        f"{contr_exact_pct:.7f}% (vs exact Bragg) / {contr_broad_pct:.7f}% (vs broadened CIF)",
        f"{rhu(f'{contr_broad_pct:.10f}', '0.1')}%--{rhu(f'{contr_exact_pct:.10f}', '0.1')}% (~2.2%, printed as about 2%)",
        "data/figs14_scxrd_simulated_peaks.csv:d_exact_A ((1,0,-1)) & figures/figs14_peaks_all.csv",
        f"[scxrd_beta_ala_tbu_10m1_contraction_pct] Relative d-spacing contraction of (10-1) reflection in HT beta-Ala-tBu thin films ({d_ht_tbu:.7f} A) vs 273 K simulated SCXRD ({d_cif_exact:.7f} A -> {contr_exact_pct:.7f}%)",
    )
    q_100_exact = float(r_100["q_exact_invA"])
    add_val(
        f"{q_100_exact:.7f}",
        rhu(f"{q_100_exact:.10f}", "0.001"),
        "data/figs14_scxrd_simulated_peaks.csv:q_exact_invA (hkl=='(1,0,0)')",
        f"[scxrd_beta_ala_tbu_100_q_invA] Simulated SCXRD q position of strongest reflection (100) in (beta-Ala-tBu)3Bi2I9 ({q_100_exact:.7f} -> 0.415 A^-1, I_rel = 100.0%)",
    )

    p_p_sec = get_peak("PEAOH", "secondary (~0.96-0.97)", "HT GIWAXS")
    p_t_sec = get_peak("beta-Ala-tBu", "second-order (~0.95)", "HT GIWAXS")
    add_val(
        f"{float(p_p_sec['q_refined_full']):.7f}",
        p_p_sec["q_3dec"],
        "figures/figs14_peaks_all.csv:q_refined_full & q_3dec (PEAOH, secondary)",
        "[giwaxs_peaoh_secondary_guide_invA] Position of secondary HT GIWAXS reflection guide for PEAOH in Fig. S14B",
    )
    add_val(
        f"{float(p_t_sec['q_refined_full']):.7f}",
        p_t_sec["q_3dec"],
        "figures/figs14_peaks_all.csv:q_refined_full & q_3dec (beta-Ala-tBu, second-order)",
        "[giwaxs_beta_ala_tbu_secondary_guide_invA] Position of second-order HT GIWAXS reflection guide for beta-Ala-tBu in Fig. S14B",
    )

    # 4. EUV-RGA text values
    ht_tbu_org = float(rga_sum.loc["beta-Ala-tBu", "ht_organic_outgassing_1e9_torr_mean"])
    ht_bala_org = float(rga_sum.loc["beta-Ala-H", "ht_organic_outgassing_1e9_torr_mean"])
    ht_peaoh_org = float(rga_sum.loc["PEAOH", "ht_organic_outgassing_1e9_torr_mean"])
    ref_tbu_org = float(rga_sum.loc["beta-Ala-tBu", "reference_organic_outgassing_1e9_torr"])
    ref_bala_org = float(rga_sum.loc["beta-Ala-H", "reference_organic_outgassing_1e9_torr"])
    ref_peaoh_org = float(rga_sum.loc["PEAOH", "reference_organic_outgassing_1e9_torr"])

    ratio_ht_low = ht_tbu_org / ht_bala_org
    ratio_ht_high = ht_tbu_org / ht_peaoh_org
    ratio_ref_low = ref_tbu_org / ref_peaoh_org
    ratio_ref_high = ref_tbu_org / ref_bala_org

    add_val(
        f"{ratio_ht_low:.7f} to {ratio_ht_high:.7f}",
        f"{rhu(f'{ratio_ht_low:.10f}', '0.1')}--{rhu(f'{ratio_ht_high:.10f}', '1')}",
        "figures/table_s1.csv:rga_ht_organic_outgassing_unrounded_1e9_torr",
        f"[rga_ht_tbu_fold_increase] Fold increase of HT beta-Ala-tBu organic outgassing ({ht_tbu_org:.7f}) over HT beta-Ala-H ({ht_bala_org:.7f}) and HT PEAOH ({ht_peaoh_org:.7f})",
    )
    add_val(
        f"{ratio_ref_low:.7f} to {ratio_ref_high:.7f}",
        f"{rhu(f'{ratio_ref_low:.10f}', '1')}--{rhu(f'{ratio_ref_high:.10f}', '1')}",
        "figures/table_s1.csv:rga_reference_organic_outgassing_unrounded_1e9_torr",
        f"[rga_reference_tbu_fold_increase] Fold increase of Reference beta-Ala-tBu organic outgassing ({ref_tbu_org:.7f}) over Reference PEAOH ({ref_peaoh_org:.7f}) and Reference beta-Ala-H ({ref_bala_org:.7f})",
    )

    ht_r44_28 = float(rga_sum.loc["beta-Ala-H", "ht_mz44_to_mz28_ratio"])
    ref_r44_28 = float(rga_sum.loc["beta-Ala-H", "reference_mz44_to_mz28_ratio"])
    sh_44_per_run = float(rga_sum.loc["beta-Ala-H", "delta_mz44_share_of_delta_organic_pct"])

    add_val(
        f"{ht_r44_28:.7f}",
        f"{rhu(f'{ht_r44_28:.10f}', '0.01')} ({rhu(f'{ht_r44_28:.10f}', '0.1')}:1)",
        "data/figs14_rga_spectra.csv (HT beta-Ala-H m/z 44:28)",
        "[rga_beta_ala_h_ht_mz44_to_28_ratio] Ratio of m/z 44 to m/z 28 in HT beta-Ala-H 100-120 C cohort (N=4)",
    )
    add_val(
        f"{ref_r44_28:.7f}",
        f"{rhu(f'{ref_r44_28:.10f}', '0.01')} ({rhu(f'{ref_r44_28:.10f}', '0.1')}:1 / about 1:1)",
        "data/figs14_rga_spectra.csv (Reference beta-Ala-H m/z 44:28)",
        "[rga_beta_ala_h_ref_mz44_to_28_ratio] Ratio of m/z 44 to m/z 28 in Reference beta-Ala-H spectrum",
    )
    add_val(
        f"{sh_44_per_run:.7f}%",
        f"{rhu(f'{sh_44_per_run:.10f}', '0.1')}%",
        "figures/figs14_sensitivity.csv:variant=='per_run_clip_and_sum_then_mean' (beta-Ala-H)",
        "[rga_beta_ala_h_mz44_share_per_run_pct] Share of HT-minus-Reference organic outgassing increase in beta-Ala-H accounted for by CO2+ at m/z 44 under per-run clip-and-sum definition",
    )

    tf44_rga = rga_runs[rga_runs["mfid"] == "TF000044"].iloc[0]
    tf54_rga = rga_runs[rga_runs["mfid"] == "TF000054"].iloc[0]
    tf44_org = float(tf44_rga["organic_outgassing_1e9_torr"])
    tf44_cos = float(tf44_rga["cos_20ion_self"])
    tf54_org = float(tf54_rga["organic_outgassing_1e9_torr"])
    tf54_cos = float(tf54_rga["cos_20ion_self"])

    add_val(
        f"{tf44_org:.7f}",
        rhu(f"{tf44_org:.10f}", "0.01"),
        "data/figs14_rga_spectra.csv:HT_TF000044_RGA000000_torr",
        "[rga_excluded_commissioning_PEAOH_TF000044_outgassing] Organic outgassing of excluded 100 C PEAOH commissioning RGA run TF000044 (300 s exposure in RGA000000)",
    )
    add_val(
        f"{tf44_cos:.7f}",
        rhu(f"{tf44_cos:.10f}", "0.001"),
        "data/figs14_rga_spectra.csv:HT_TF000044_RGA000000_torr",
        "[rga_excluded_commissioning_PEAOH_TF000044_cos_theta] 20-ion cosine similarity vs Reference for excluded 100 C PEAOH commissioning RGA run TF000044",
    )
    add_val(
        f"{tf54_org:.7f} / {tf54_cos:.7f}",
        f"{rhu(f'{tf54_org:.10f}', '0.01')} x 10^-9 Torr / cos_theta = {rhu(f'{tf54_cos:.10f}', '0.001')}",
        "data/figs14_rga_spectra.csv:HT_TF000054_RGA000000_torr",
        "[rga_excluded_commissioning_beta_Ala_H_TF000054] Organic outgassing and 20-ion cosine similarity of excluded 100 C beta-Ala-H commissioning RGA run TF000054 (300 s exposure in RGA000000)",
    )

    t_span = float(rga_sum.loc["PEAOH", "ref_t_span_starts_s"])
    t_win4 = float(rga_sum.loc["PEAOH", "ref_t_four_sweep_window_s"])
    add_val(
        f"{t_span:.7f} (sweep-start span) / {t_win4:.7f} (4-sweep duration)",
        f"{rhu(f'{t_span:.10f}', '1')} s (start-to-start) / {rhu(f'{t_win4:.10f}', '1')} s (total 4-sweep duration; ~35 s in text)",
        "data/figs14_rga_runs.csv:t_span_starts_s & t_four_sweep_window_s",
        "[rga_reference_averaging_window_s] Reference RGA post-shutter averaging window across the first 4 mass scans",
    )

    mz73_ht = float(rga_sum.loc["beta-Ala-H", "ht_mz73_signed_1e9_torr"])
    mz73_sd = float(rga_sum.loc["beta-Ala-H", "ht_mz73_std_1e9_torr"])
    rms_noise = float(rga_sum.loc["beta-Ala-H", "ht_rms_noise_mz60_120_1e9_torr"])
    add_val(
        f"{mz73_ht:+.7f} +/- {mz73_sd:.7f} (RMS noise = {rms_noise:.7f})",
        f"{rhu(f'{mz73_ht:.10f}', '0.001')} +/- {rhu(f'{mz73_sd:.10f}', '0.001')} x 10^-9 Torr (noise ~ {rhu(f'{rms_noise:.10f}', '0.001')} x 10^-9 Torr)",
        "data/figs14_rga_spectra.csv (HT beta-Ala-H m/z 73 and m/z 60-120 RMS noise)",
        "[rga_beta_ala_h_mz73_and_noise_1e9_torr] Net background-subtracted signal at m/z 73 (DMF parent ion) in HT beta-Ala-H 100-120 C cohort compared with baseline RMS noise over m/z 60-120",
    )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------- plotting
def enforce_box_and_arial(fig: plt.Figure) -> None:
    for ax in fig.axes:
        for spine_name in ("top", "right", "bottom", "left"):
            ax.spines[spine_name].set_visible(True)
            ax.spines[spine_name].set_linewidth(0.8)
            ax.spines[spine_name].set_color("#222222")
        ax.tick_params(
            top=False,
            right=False,
            bottom=True,
            left=ax.yaxis.get_ticks_position() != "none" and len(ax.get_yticks()) > 0,
            direction="out",
            width=0.8,
            length=3.0,
            colors="#222222",
        )
    for txt in fig.findobj(match=Text):
        txt.set_fontfamily("sans-serif")
        txt.set_fontname("Arial")


def add_outside_panel_label(ax: plt.Axes, letter: str, x: float = -0.12, y: float = 1.025) -> None:
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=11.5,
        fontweight="bold",
        fontname="Arial",
        va="bottom",
        ha="right",
        color="#000000",
        clip_on=False,
    )


def build_figure(
    uv_envelopes: dict,
    uv_sum: pd.DataFrame,
    diff_df: pd.DataFrame,
    peaks_df: pd.DataFrame,
    rga_ensemble: dict[str, np.ndarray],
    rga_sum: pd.DataFrame,
) -> plt.Figure:
    ht_peaks = peaks_df[peaks_df["source"] == "HT GIWAXS"].copy()
    guides_by_comp: dict[str, list[float]] = {}
    for comp_name, grp in ht_peaks.groupby("composition", sort=False):
        guides_by_comp[str(comp_name)] = [float(x) for x in grp["q_3dec"].tolist()]

    fig = plt.figure(figsize=(14.2, 6.4))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.0, 1.0, 1.12], wspace=0.25)

    # Align grey horizontal divider lines across Panels A and B
    b_step = 0.86 + 0.86 + 0.86 + 1.52  # 4.10 per composition block in Panel B
    b_div1_data = b_step - 0.28         # 3.82
    b_div2_data = 2.0 * b_step - 0.28   # 7.92
    b_ymin, b_ymax = -0.10, 3.0 * b_step + 0.88  # -0.10 to 13.18
    f_div1 = (b_div1_data - b_ymin) / (b_ymax - b_ymin)
    f_div2 = (b_div2_data - b_ymin) / (b_ymax - b_ymin)

    uv_scale = 0.80
    a_ymin, a_ymax = -0.06, 4.46
    a_div1_data = a_ymin + f_div1 * (a_ymax - a_ymin)
    a_div2_data = a_ymin + f_div2 * (a_ymax - a_ymin)
    a_off0 = 0.00
    a_off1 = a_div1_data + 0.10
    a_off2 = a_div2_data + 0.10

    # --- Panel A: UV-vis ---
    ax_a = fig.add_subplot(gs[0, 0])
    wl = uv_envelopes["wavelength_nm"]
    mask_uv = (wl >= 365) & (wl <= 625)
    w_sub = wl[mask_uv]

    tiers = [
        (
            "PEAOH",
            a_off0,
            "Reference_PEAOH_norm",
            "DMF_control_PEAOH_norm",
            "HT_PEAOH_norm_mean",
            "HT_PEAOH_norm_std",
            rf"PEAOH (1:1 BiI$_3$)  |  {uv_sum.loc['PEAOH', 'ht_lambda_exc_mean_nm']:.1f} nm (cos $\theta$ vs. ref. = {uv_sum.loc['PEAOH', 'cos_theta_vs_reference_mean']:.3f})",
            float(uv_sum.loc["PEAOH", "ht_lambda_exc_mean_nm"]),
        ),
        (
            "beta-Ala-H",
            a_off1,
            "Reference_beta_Ala_H_norm",
            "DMF_control_beta_Ala_H_norm",
            "HT_beta_Ala_H_norm_mean",
            "HT_beta_Ala_H_norm_std",
            rf"$\beta$-Ala-H (2:1 BiI$_3$)  |  {uv_sum.loc['beta-Ala-H', 'ht_lambda_exc_mean_nm']:.1f} nm (cos $\theta$ vs. ref. = {uv_sum.loc['beta-Ala-H', 'cos_theta_vs_reference_mean']:.3f})",
            float(uv_sum.loc["beta-Ala-H", "ht_lambda_exc_mean_nm"]),
        ),
        (
            "beta-Ala-tBu",
            a_off2,
            "Reference_beta_Ala_tBu_norm",
            "DMF_control_beta_Ala_tBu_norm",
            "HT_beta_Ala_tBu_norm_mean",
            "HT_beta_Ala_tBu_norm_std",
            rf"$\beta$-Ala-$^{{t}}$Bu (3:2 BiI$_3$)  |  {uv_sum.loc['beta-Ala-tBu', 'ht_lambda_exc_mean_nm']:.1f} nm (cos $\theta$ vs. ref. = {uv_sum.loc['beta-Ala-tBu', 'cos_theta_vs_reference_mean']:.3f})",
            float(uv_sum.loc["beta-Ala-tBu", "ht_lambda_exc_mean_nm"]),
        ),
    ]

    ax_a.axhline(a_div1_data, color="#D0D5DD", linestyle="-", linewidth=0.75, zorder=1)
    ax_a.axhline(a_div2_data, color="#D0D5DD", linestyle="-", linewidth=0.75, zorder=1)

    for idx, (comp, off, ref_col, dmf_col, ht_m, ht_s, hdr_lbl, lam_pk) in enumerate(tiers):
        col = CATION_COLORS[comp]
        ym = np.clip(uv_envelopes[ht_m][mask_uv], -0.02, 1.15) * uv_scale
        ys = np.minimum(uv_envelopes[ht_s][mask_uv], 0.22) * uv_scale
        lbl_ht = r"HT (DMF, mean $\pm$ 1$\sigma$)" if idx == 2 else None
        ax_a.plot(w_sub, ym + off, color=col, linewidth=1.85, label=lbl_ht, zorder=4)
        ax_a.fill_between(
            w_sub,
            np.maximum(ym - ys, 0.0) + off,
            np.minimum(ym + ys, 1.15 * uv_scale) + off,
            color=col,
            alpha=0.18,
            zorder=2,
        )

        yr = np.clip(uv_envelopes[ref_col][mask_uv], -0.02, 1.15) * uv_scale
        lbl_lr = "Reference (optimized solvent)" if idx == 2 else None
        ax_a.plot(w_sub, yr + off, color="black", linestyle="--", linewidth=1.35, label=lbl_lr, zorder=5)

        yd = np.clip(uv_envelopes[dmf_col][mask_uv], -0.02, 1.15) * uv_scale
        lbl_ld = "DMF control" if idx == 2 else None
        ax_a.plot(w_sub, yd + off, color="#555555", linestyle=":", linewidth=1.35, label=lbl_ld, zorder=5)

        ax_a.plot([lam_pk, lam_pk], [off, off + uv_scale], color=col, linestyle=":", linewidth=0.85, alpha=0.75)
        ax_a.text(
            622,
            off + 0.96,
            hdr_lbl,
            ha="right",
            va="bottom",
            fontsize=8.2,
            fontweight="bold",
            color=col,
            bbox=dict(boxstyle="square,pad=0.03", facecolor="white", edgecolor="none", alpha=0.86),
            zorder=6,
        )

    ax_a.set_xlim(365, 625)
    ax_a.set_ylim(a_ymin, a_ymax)
    ax_a.set_yticks([])
    ax_a.set_xlabel("Wavelength (nm)")
    ax_a.set_ylabel("Normalized absorbance")
    ax_a.set_title("UV-vis excitonic absorption", pad=6)
    ax_a.legend(
        loc="upper right",
        bbox_to_anchor=(0.995, 0.995),
        fontsize=8.0,
        ncol=2,
        frameon=False,
        handlelength=1.4,
        columnspacing=0.7,
        labelspacing=0.25,
    )
    add_outside_panel_label(ax_a, "A", x=-0.12, y=1.025)

    # --- Panel B: GIWAXS & XRD ---
    ax_b = fig.add_subplot(gs[0, 1])
    q = diff_df["q_invA"].to_numpy(dtype=float)
    mask_q = (q >= 0.32) & (q <= 1.98)
    q_sub = q[mask_q]

    g_peaoh = guides_by_comp["PEAOH"]
    g_bala = guides_by_comp["beta-Ala-H"]
    g_tbu = guides_by_comp["beta-Ala-tBu"]

    groups = [
        (
            "PEAOH",
            "PEAOH",
            "PEAOH (1:1, DMF)",
            rf"$q_1$ = {g_peaoh[0]:.3f} $\mathdefault{{\AA}}^{{-1}}$",
            g_peaoh,
        ),
        (
            "beta-Ala-H",
            "beta_Ala_H",
            r"$\beta$-Ala-H (2:1, DMF vs. MeOH)",
            rf"$q$ = {g_bala[0]:.3f}, {g_bala[1]:.3f} (002), {g_bala[2]:.3f} (-101) $\mathdefault{{\AA}}^{{-1}}$",
            g_bala,
        ),
        (
            "beta-Ala-tBu",
            "beta_Ala_tBu",
            r"$\beta$-Ala-$^{t}$Bu (3:2, DMF vs. MeOH)",
            rf"$q_1$ = {g_tbu[0]:.3f} $\mathdefault{{\AA}}^{{-1}}$",
            g_tbu,
        ),
    ]

    offset = 0.0
    for idx, (comp, ctag, title_lbl, sub_lbl, pk_qs) in enumerate(groups):
        base_off = offset
        col = CATION_COLORS[comp]

        y_sim = diff_df[f"SCXRD_sim_{ctag}_norm"].to_numpy(dtype=float)[mask_q]
        lbl = "SCXRD (simulated)" if idx == 2 else None
        ax_b.plot(q_sub, y_sim * 0.72 + offset, color="#7F8C8D", linewidth=1.0, label=lbl)
        ax_b.fill_between(q_sub, offset, y_sim * 0.72 + offset, color="#7F8C8D", alpha=0.14)
        offset += 0.86

        y_xrd = diff_df[f"Reference_XRD_{ctag}_norm"].to_numpy(dtype=float)[mask_q]
        lbl = r"Reference XRD (Co K$\alpha$)" if idx == 2 else None
        ax_b.plot(q_sub, y_xrd * 0.76 + offset, color="#6C3483", linestyle=":", linewidth=1.2, label=lbl)
        offset += 0.86

        y_lg = diff_df[f"Reference_GIWAXS_{ctag}_norm"].to_numpy(dtype=float)[mask_q]
        lbl = "Reference GIWAXS (Si)" if idx == 2 else None
        ax_b.plot(q_sub, y_lg * 0.76 + offset, color="black", linestyle="--", linewidth=1.3, label=lbl)
        offset += 0.86

        yc = diff_df[f"HT_{ctag}_mean"].to_numpy(dtype=float)[mask_q]
        ys = diff_df[f"HT_{ctag}_std"].to_numpy(dtype=float)[mask_q]
        lbl = "HT GIWAXS (ITO)" if idx == 2 else None
        cruc_base = offset
        c_scale = 0.78 if idx == 2 else 0.82
        ax_b.plot(q_sub, yc * c_scale + offset, color=col, linewidth=1.75, label=lbl)
        ax_b.fill_between(q_sub, (yc - ys) * c_scale + offset, (yc + ys) * c_scale + offset, color=col, alpha=0.20)
        top_off = offset + 0.88
        offset += 1.52

        for pq in pk_qs:
            ax_b.plot([pq, pq], [base_off, top_off], color=col, linestyle=":", linewidth=0.80, alpha=0.60)

        if comp == "beta-Ala-H":
            ax_b.text(
                g_bala[1] - 0.015,
                cruc_base + 0.72,
                "(002)",
                ha="right",
                va="bottom",
                fontsize=8.0,
                fontweight="bold",
                color=col,
                bbox=dict(boxstyle="square,pad=0.02", facecolor="white", edgecolor="none", alpha=0.82),
                zorder=6,
            )
            ax_b.text(
                g_bala[2] + 0.015,
                cruc_base + 0.72,
                "(-101)",
                ha="left",
                va="bottom",
                fontsize=8.0,
                fontweight="bold",
                color=col,
                bbox=dict(boxstyle="square,pad=0.02", facecolor="white", edgecolor="none", alpha=0.82),
                zorder=6,
            )

        y_hdr = cruc_base + 0.22 if idx == 2 else (cruc_base + 0.24 if idx == 1 else cruc_base + 0.34)
        hdr_text = f"{title_lbl}\n{sub_lbl}" if idx == 1 else f"{title_lbl}  |  {sub_lbl}"
        ax_b.text(
            1.95,
            y_hdr,
            hdr_text,
            ha="right",
            va="bottom",
            fontsize=8.2,
            fontweight="bold",
            color=col,
            linespacing=0.92,
            bbox=dict(boxstyle="square,pad=0.03", facecolor="white", edgecolor="none", alpha=0.86),
            zorder=6,
        )

        if idx < 2:
            ax_b.axhline(offset - 0.28, color="#D0D5DD", linestyle="-", linewidth=0.75)

    ax_b.set_xlim(0.32, 1.98)
    ax_b.set_ylim(b_ymin, b_ymax)
    ax_b.set_yticks([])
    ax_b.set_xlabel(r"Scattering vector, $q$ ($\mathdefault{\AA}^{-1}$)")
    ax_b.set_ylabel("Normalized intensity, $I(q)$")
    ax_b.set_title("GIWAXS & XRD phase matching", pad=6)
    ax_b.legend(
        loc="upper right",
        bbox_to_anchor=(0.995, 0.995),
        fontsize=8.0,
        ncol=2,
        frameon=False,
        handlelength=1.3,
        columnspacing=0.6,
        labelspacing=0.25,
    )
    add_outside_panel_label(ax_b, "B", x=-0.12, y=1.025)

    # --- Panel C: RGA Mass Spectra ---
    gs_sub = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0, 2], hspace=0.14)
    mz = rga_ensemble["mz"]
    mask_mz = (mz >= 12) & (mz <= 145) & (mz != 18) & (mz != 32)
    mz_sub = mz[mask_mz]

    peaks_annot = {
        "beta-Ala-tBu": [
            (15, r"CH$_3^+$", -1.5, 0.05),
            (28, r"CO$^+$/C$_2$H$_4^+$", 3.5, 0.05),
            (41, r"C$_3$H$_5^+$", -5.5, 0.05),
            (44, r"CO$_2^+$", 6.5, 0.05),
            (56, r"C$_4$H$_8^+$", 2.5, 0.04),
            (127, r"I$^+$", 0.0, 0.05),
            (142, r"CH$_3$I$^+$", -3.0, 0.05),
        ],
        "beta-Ala-H": [
            (15, r"CH$_3^+$", 0.0, 0.53),
            (28, r"CO$^+$/C$_2$H$_4^+$", -2.5, 0.05),
            (44, r"CO$_2^+$", 3.5, 0.05),
        ],
        "PEAOH": [
            (16, r"CH$_4^+$/O$^+$", -2.2, 0.44),
            (28, r"CO$^+$/C$_2$H$_4^+$", 9.0, 0.06),
            (44, r"CO$_2^+$", 2.5, 0.05),
            (127, r"I$^+$", 0.0, 0.05),
        ],
    }

    for idx, comp in enumerate(["beta-Ala-tBu", "beta-Ala-H", "PEAOH"]):
        ax_s = fig.add_subplot(gs_sub[idx])
        col = CATION_COLORS[comp]
        ctag = comp.replace("-", "_")
        y_leg = np.maximum(rga_ensemble[f"Reference_{ctag}_torr"][mask_mz], 0.0)
        y_ht = np.maximum(rga_ensemble[f"HT_{ctag}_100_120C_torr"][mask_mz], 0.0)
        if np.max(y_leg) > 0:
            y_leg = y_leg / np.max(y_leg)
        if np.max(y_ht) > 0:
            y_ht = y_ht / np.max(y_ht)

        ax_s.vlines(mz_sub - 0.30, 0, y_leg, color="#1E293B", linewidth=1.3, label="Reference (100 °C)")
        ax_s.vlines(mz_sub + 0.30, 0, y_ht, color=col, linewidth=1.3, label="HT (100 or 120 °C)")

        ax_s.set_xlim(6.5, 146)
        ax_s.set_ylim(0, 1.62)
        ax_s.set_yticks([0.0, 0.5, 1.0])
        if idx == 1:
            ax_s.set_ylabel(r"Normalized partial pressure, $\Delta P$")

        cos_20 = float(rga_sum.loc[comp, "cos_theta_20ion_100_120C"])
        ax_s.text(
            0.98,
            0.93,
            rf"{CATION_DISPLAY[comp]} (BiI$_3$, 20-ion cos $\theta$ = {cos_20:.3f})",
            transform=ax_s.transAxes,
            ha="right",
            va="top",
            fontsize=8.0,
            fontweight="bold",
            color=col,
        )

        if idx == 0:
            ax_s.set_title("EUV-RGA mass spectra", pad=6)
            add_outside_panel_label(ax_s, "C", x=-0.12, y=1.075)
            ax_s.legend(
                loc="upper right",
                bbox_to_anchor=(0.99, 0.75),
                fontsize=8.0,
                ncol=1,
                frameon=False,
                handlelength=1.4,
                labelspacing=0.22,
            )

        for pk_mz, pk_lbl, dx, dy in peaks_annot[comp]:
            pk_i = np.where(mz_sub == pk_mz)[0]
            if len(pk_i) > 0:
                pk_h = max(float(y_leg[pk_i[0]]), float(y_ht[pk_i[0]]))
                if pk_h > 0.06:
                    y_txt = min(pk_h + dy, 1.24)
                    ax_s.text(
                        pk_mz + dx,
                        y_txt,
                        f"{pk_lbl}\n({pk_mz})",
                        ha="center",
                        va="bottom",
                        fontsize=7.6,
                        color="#222222",
                        linespacing=0.86,
                    )

        if idx < 2:
            ax_s.set_xticklabels([])
        else:
            ax_s.set_xlabel("Mass-to-charge ratio, $m/z$")

    enforce_box_and_arial(fig)
    fig.canvas.draw()
    return fig


def main() -> None:
    apply_publication_style()

    cohort_df = pd.read_csv(DATA_DIR / "figs14_cohort_films.csv", keep_default_na=False)
    uv_spectra = pd.read_csv(DATA_DIR / "figs14_uvvis_spectra.csv")
    uv_scans = pd.read_csv(DATA_DIR / "figs14_uvvis_scans.csv")
    diff_df = pd.read_csv(DATA_DIR / "figs14_diffraction_profiles.csv")
    sim_peaks_df = pd.read_csv(DATA_DIR / "figs14_scxrd_simulated_peaks.csv")
    rga_spectra = pd.read_csv(DATA_DIR / "figs14_rga_spectra.csv")
    rga_runs = pd.read_csv(DATA_DIR / "figs14_rga_runs.csv")

    uv_sum, uv_envelopes = compute_uvvis_metrics(uv_spectra, uv_scans)
    peaks_df, one_df = compute_diffraction_metrics(diff_df)
    rga_sum, rga_per_run, rga_ensemble = compute_rga_metrics(rga_spectra, rga_runs)
    sens_df = build_sensitivity_table(cohort_df, uv_scans, rga_sum)
    table_df, table_tex = build_table_s1(cohort_df, uv_sum, peaks_df, one_df, rga_sum, sens_df)
    text_vals_df = build_text_values(
        cohort_df, uv_scans, uv_sum, diff_df, peaks_df, one_df, sim_peaks_df, rga_per_run, rga_sum, sens_df, table_df
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table_df.to_csv(OUTPUT_DIR / "table_s1.csv", index=False)
    (OUTPUT_DIR / "table_s1.tex").write_text(table_tex, encoding="utf-8")
    peaks_df.to_csv(OUTPUT_DIR / "figs14_peaks_all.csv", index=False)
    sens_df.to_csv(OUTPUT_DIR / "figs14_sensitivity.csv", index=False)
    text_vals_df.to_csv(OUTPUT_DIR / "figs14_text_values.csv", index=False)

    fig = build_figure(uv_envelopes, uv_sum, diff_df, peaks_df, rga_ensemble, rga_sum)
    fig.savefig(OUTPUT_DIR / "figure_s14.svg", dpi=RASTER_DPI, bbox_inches="tight", pad_inches=0.06, metadata={"Date": None})
    fig.savefig(OUTPUT_DIR / "figure_s14.pdf", dpi=RASTER_DPI, bbox_inches="tight", pad_inches=0.06, metadata={"CreationDate": None})
    fig.savefig(OUTPUT_DIR / "figure_s14.png", dpi=RASTER_DPI, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)

    print(f"wrote {OUTPUT_DIR / 'figure_s14'}.svg, .pdf and .png")
    print(f"wrote {OUTPUT_DIR / 'table_s1'}.csv and .tex")
    print(f"wrote {OUTPUT_DIR / 'figs14_peaks_all.csv'}, figs14_sensitivity.csv and figs14_text_values.csv")


if __name__ == "__main__":
    main()
