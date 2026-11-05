#!/usr/bin/env python3
"""Figure 4: RGA analysis of beta-alanine tert-butyl ester hydroiodide films
with rgakit (candidate generation, NIST matching, human review, quantification).

Input: data/fig04_rga_summary.json, data/fig04_rga_library.csv,
data/fig04_rga_runs.csv, data/fig04_rga_fit_before_review.csv,
data/fig04_rga_fit_after_review.csv, data/fig04_rga_partial_pressures.csv.
Output: figures/figure_04.{svg,pdf,png}.

186 mm wide, Arial; SVG text stays editable.
"""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / ".." / "data"
OUTPUT_DIR = HERE / ".." / "figures"

INK = "#122f3e"
RULE_COLOR = "#bccac9"
TEAL = "#207d83"
ORANGE = "#f36b21"

PUBLICATION_STYLE = {
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.family": "Arial",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.default": "regular",
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.35,
    "ytick.minor.width": 0.35,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "xtick.minor.size": 1.2,
    "ytick.minor.size": 1.2,
    "xtick.major.pad": 1.5,
    "ytick.major.pad": 1.5,
    "axes.labelpad": 1.5,
    "lines.linewidth": 0.8,
}


def apply_publication_style():
    """Raises if Arial is unavailable, since metrics would silently change."""
    matplotlib.rcParams.update(PUBLICATION_STYLE)
    font_manager.findfont("Arial", fallback_to_default=False)


def style_axis(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


FIG_WIDTH_IN = 186 / 25.4
RASTER_DPI = 600
SPECTRUM_MZ = (1, 60)
HIGH_MZ = (125, 145)
H2_MZ = 2
MUTED = "#4a5f6b"
LINK_GREEN = "#4e785f"
SLATE = "#5f7682"
RGA_PURPLE = "#806781"
RGA_PURPLE_TINT = "#f8f2f8"
HEATMAP_CMAP = LinearSegmentedColormap.from_list("ink_blue", ["#eef4f7", "#b8d2da", "#6f9db3", "#33586f", INK])
DISCARD_TEXT = "#a63f0a"
CATEGORY_COLORS = {"film": TEAL, "both": "#6fbcc0", "environment": "#7f7a70", "removed": ORANGE}
CATEGORY_LABELS = {"film": "film only", "both": "film or environment", "environment": "environment",
                   "removed": "removed in review"}
CATEGORY_ORDER = ["film", "both", "environment", "removed"]
METAL_LABEL = {"BiI": "BiI$_3$", "SbI": "SbI$_3$"}
DISCARD = CATEGORY_COLORS["removed"]
DISCARD_LIGHT = "#fff0e5"
BAND_ALPHA = {"film": 0.35, "both": 0.55, "environment": 0.45, "removed": 0.35}
ATOM_COLORS = {"C": INK, "O": "#d7301f", "N": "#2a5bd7", "I": "#7b2cbf"}
CONVENTIONAL_FORMULAS = {"H3N": "NH3"}
ISOMER_LABELS = {"Isobutane": "$i$-C$_4$H$_{10}$", "Butane": "$n$-C$_4$H$_{10}$"}
NOT_FITTED = "#8a8a8a"
NO_NIST_HEIGHT = 0.16
MIN_PRESENT = 0.25
MAX_ROWS = 12
CHEMISTRY_LABEL = "β-alanine\n$\\it{tert}$-butyl ester HI"


def formula_tex(formula):
    formula = CONVENTIONAL_FORMULAS.get(formula, formula)
    return re.sub(r"(\d+)", r"$_{\1}$", formula)


def ratio_value(ratio):
    a, b = ratio.split(":")
    return float(a) / float(b)


def draw_molecule(ax, molecule, fontsize=8.5, lw=1.1):
    """Depiction from precomputed 2D coordinates (RDKit, canonical orientation, counterion placed by the N)."""
    atoms, bonds = molecule["atoms"], molecule["bonds"]
    pos = np.array([[a["x"], a["y"]] for a in atoms])
    neighbors = {i: [] for i in range(len(atoms))}
    for b in bonds:
        neighbors[b["begin"]].append(b["end"])
        neighbors[b["end"]].append(b["begin"])
    labelled = set()
    for i, atom in enumerate(atoms):
        sym = atom["symbol"]
        if sym == "C" and atom["charge"] == 0 and atom["degree"] > 0:
            continue
        labelled.add(i)
        n_h, charge = atom["n_h"], atom["charge"]
        h = ("H" + (f"$_{{{n_h}}}$" if n_h > 1 else "")) if n_h else ""
        sup = ("$^{" + ("+" if charge > 0 else "−") + "}$") if charge else ""
        nbr = [pos[n] for n in neighbors[i]]
        left = bool(nbr) and np.mean([p[0] for p in nbr]) > pos[i][0] + 0.1
        style = {"va": "center", "fontsize": fontsize, "fontweight": "bold", "color": ATOM_COLORS[sym]}
        if not h:
            ax.text(*pos[i], sym + sup, ha="center", **style)
            continue
        ax.text(*pos[i], sym, ha="center", **style)
        shift = 0.36 * fontsize
        ax.annotate(h if left else h + sup, pos[i], xytext=(-shift if left else shift, 0),
                    textcoords="offset points", ha="right" if left else "left", **style)
    for bond in bonds:
        i, j = bond["begin"], bond["end"]
        p, q = pos[i].copy(), pos[j].copy()
        d = (q - p) / np.linalg.norm(q - p)
        if i in labelled:
            p += d * 0.42
        if j in labelled:
            q -= d * 0.42
        mid = (pos[i] + pos[j]) / 2
        offsets = [0.0] if bond["order"] == 1 else [-0.09, 0.09]
        normal = np.array([-d[1], d[0]])
        for o in offsets:
            for a, b, idx in ((p, mid, i), (mid, q, j)):
                ax.plot([a[0] + o * normal[0], b[0] + o * normal[0]], [a[1] + o * normal[1], b[1] + o * normal[1]],
                        color=ATOM_COLORS[atoms[idx]["symbol"]], lw=lw, solid_capstyle="round")
    ax.set_xlim(pos[:, 0].min() - 1.3, pos[:, 0].max() + 1.3)
    ax.set_ylim(pos[:, 1].min() - 0.5, pos[:, 1].max() + 0.5)
    ax.set_aspect("equal")
    ax.axis("off")


def _band(ax, xa, xb, ya, yb, color, alpha=0.35):
    xs = np.linspace(xa, xb, 30)
    t = (1 - np.cos(np.linspace(0, np.pi, 30))) / 2
    lo = ya[0] + (yb[0] - ya[0]) * t
    hi = ya[1] + (yb[1] - ya[1]) * t
    ax.add_patch(Polygon(np.r_[np.c_[xs, hi], np.c_[xs[::-1], lo[::-1]]], fc=color, ec="none", alpha=alpha))


def _bar_break(ax, x, w, y, rise=0.012, half=0.006, over=0.004):
    x0, x1 = x - over, x + w + over
    ax.add_patch(Polygon([(x0, y - rise - half), (x1, y + rise - half), (x1, y + rise + half), (x0, y - rise + half)],
                         fc="white", ec="none", zorder=6))
    for dy in (-half, half):
        ax.plot([x0, x1], [y - rise + dy, y + rise + dy], color=INK, lw=0.6, zorder=7)


def _stack(top, sizes, unit, gap=0.0):
    out, y = [], top
    for n in sizes:
        out.append((y - n * unit, y))
        y -= n * unit + gap
    return out


def _bar(ax, x, w, segments, ls="-", edge=MUTED):
    for (lo, hi), color in segments:
        ax.add_patch(Rectangle((x, lo), w, hi - lo, fc=color, ec="none"))
    lo, hi = min(s[0][0] for s in segments), max(s[0][1] for s in segments)
    ax.add_patch(Rectangle((x, lo), w, hi - lo, fc="none", ec=edge, lw=0.6, ls=ls))


def panel_flow(fig, molecule, n_frag, cat, generated, reviewed):
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    c = CATEGORY_COLORS
    env_only = {n for n in generated if cat[n] == "environment"}
    both = {n for n in generated if cat[n] == "both"}
    film = {n for n in generated if cat[n] == "film"}
    removed = generated - reviewed
    if removed - film:
        raise ValueError(f"flow layout assumes only film-only compounds are removed: {sorted(removed - film)}")
    kept = {k: len(s & reviewed) for k, s in (("environment", env_only), ("both", both), ("film", film))}
    n_nomatch = n_frag - len(both) - len(film)
    n_env = len(env_only) + len(both)
    n_cand = n_frag + len(env_only)
    xf, xe, xa, xs, xb, xc, w = 0.165, 0.225, 0.325, 0.415, 0.565, 0.8, 0.018
    top, block_gap, source_gap = 0.81, 0.014, 0.04
    unit = (0.75 - source_gap - NO_NIST_HEIGHT) / (n_env + n_frag - n_nomatch)
    n_broken = NO_NIST_HEIGHT / unit

    A = _stack(top, [len(env_only), len(both), len(film), n_broken], unit, block_gap)
    B = _stack(top, [kept["environment"], kept["both"], kept["film"], len(removed)], unit, block_gap)
    B_drop = (A[-1][0] - block_gap, A[-1][1] - block_gap)
    C = _stack(top, kept.values(), unit, block_gap)
    C_drop = (C[-1][0] - block_gap - len(removed) * unit, C[-1][0] - block_gap)
    E = _stack(top, [len(env_only), len(both)], unit)
    F = _stack(E[-1][0] - source_gap, [len(both), len(film) + n_broken], unit)
    F_split = _stack(F[1][1], [len(film), n_broken], unit)
    A_split = _stack(A[2][1], [kept["film"], len(removed)], unit)

    band = lambda x0, x1, ya, yb, k: _band(ax, x0, x1, ya, yb, c[k], alpha=BAND_ALPHA[k])
    band(xe + w, xa, E[0], A[0], "environment")
    band(xe + w, xa, E[1], A[1], "both")
    band(xe + w, xa, F[0], A[1], "both")
    band(xe + w, xa, F_split[0], A[2], "film")
    band(xe + w, xa, F_split[1], A[3], "film")
    _bar(ax, xe, w, [(E[0], c["environment"]), (E[1], c["both"])], ls=(0, (2, 1)))
    _bar(ax, xe, w, [(F[0], c["both"]), (F[1], c["film"])])
    y_env = (E[0][1] + E[-1][0]) / 2
    ax.text(xe - 0.008, y_env, f"Environment\nlibrary ({n_env})", ha="right", va="bottom",
            fontsize=8, fontweight="bold", color=MUTED, linespacing=1.1)
    ax.text(xe - 0.008, y_env - 0.006, "chamber gases,\nhydrocarbons", ha="right", va="top", fontsize=8,
            color=MUTED, linespacing=1.1)
    ax.text(xe + w, F[-1][0] - 0.012, f"{n_frag} fragments", ha="right", va="top", fontsize=8,
            fontweight="bold", color=MUTED)

    y_frag = (F[0][1] + F[-1][0]) / 2
    _band(ax, xf, xe, (y_frag - 0.012, y_frag + 0.012), (F[-1][0], F[0][1]), LINK_GREEN, alpha=0.2)
    ax.text(xf + 0.032, y_frag, "rgakit", ha="center", va="center", fontsize=8.5, fontweight="bold", color=LINK_GREEN)
    a_mol = ax.inset_axes([0.0, y_frag - 0.17, 0.165, 0.34])
    draw_molecule(a_mol, molecule)
    ax.text(0.085, y_frag - 0.18, CHEMISTRY_LABEL, ha="center", va="top", fontsize=8.5, fontweight="bold",
            linespacing=1.1)

    sections = [(0.005, xa + w, "1  Generate candidates"), (xa + w + 0.012, xb + w, "2  Match NIST spectra"),
                (xb + w + 0.012, xc + w, "3  Human in the loop")]
    for x0, x1, title in sections:
        ax.plot([x0, x0, x1, x1], [0.915, 0.93, 0.93, 0.915], color=MUTED, lw=0.6, solid_capstyle="butt")
        ax.text((x0 + x1) / 2, 0.94, title, ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                color=INK)
    ax.add_patch(FancyBboxPatch((xb + w + 0.012, C_drop[0] - 0.02), xc - xb - 0.012, 0.91 - C_drop[0] + 0.02,
                                boxstyle="round,pad=0,rounding_size=0.02", fc=RGA_PURPLE_TINT, ec=RGA_PURPLE, lw=0.5, zorder=0))

    band(xa + w, xb, A[0], B[0], "environment")
    band(xa + w, xb, A[1], B[1], "both")
    band(xa + w, xb, A_split[0], B[2], "film")
    band(xa + w, xb, A_split[1], B[3], "film")
    _band(ax, xa + w, xs, A[-1], B_drop, DISCARD, alpha=0.15)
    for seg_b, seg_c, k in zip(B[:3], C, kept):
        band(xb + w, xc, seg_b, seg_c, k)
    band(xb + w, xc, B[3], C_drop, "removed")

    for seg, k in zip(A, ("environment", "both", "film", "film")):
        _bar(ax, xa, w, [(seg, c[k])])
    _bar(ax, xs, w, [(B_drop, DISCARD_LIGHT)], edge=ORANGE)
    for seg, k in zip(B, ("environment", "both", "film", "film")):
        _bar(ax, xb, w, [(seg, c[k])])
    for seg, k in zip(C, kept):
        _bar(ax, xc, w, [(seg, c[k])])
    _bar(ax, xc, w, [(C_drop, DISCARD)])
    for x_bar, seg in ((xe, F_split[1]), (xa, A[3]), (xs, B_drop)):
        _bar_break(ax, x_bar, w, sum(seg) / 2)
    ax.text(xa + w + 0.006, sum(A[1]) / 2, f"{len(both)} in both", ha="left", va="center", fontsize=8,
            color=INK)

    y_head = top + 0.028
    head = {"va": "bottom", "fontsize": 8.5, "fontweight": "bold", "color": INK}
    ax.text(xa + w / 2, y_head, f"{n_cand} candidates", ha="center", **head)
    ax.text(xb + w, y_head, f"{len(generated)} matched (fit B)", ha="right", **head)
    ax.text(xs + w + 0.01, sum(B_drop) / 2, f"{n_nomatch} no NIST spectrum", ha="left", va="bottom", fontsize=8,
            fontweight="bold", color=DISCARD_TEXT)
    ax.text(xs + w + 0.01, sum(B_drop) / 2 - 0.006, "not to scale", ha="left", va="top", fontsize=8,
            color=MUTED)
    ax.text(xc + w + 0.01, (C[0][1] + C[-1][0]) / 2, f"Final library: {len(reviewed)}\n(fit C)", ha="left",
            va="center", fontsize=8, fontweight="bold", color=TEAL, linespacing=1.1)
    ax.text(xc + w + 0.01, sum(C_drop) / 2, f"Discarded: {len(removed)}", ha="left", va="center", fontsize=8,
            fontweight="bold", color=DISCARD_TEXT)


def category_legend(fig, y=0.995):
    handles = [plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLORS[k]) for k in CATEGORY_ORDER]
    fig.legend(handles, [CATEGORY_LABELS[k] for k in CATEGORY_ORDER], loc="upper center", bbox_to_anchor=(0.5, y),
               ncol=len(handles), frameon=False, handlelength=1.0, handletextpad=0.4, columnspacing=1.2, fontsize=8)


def observed_legend(fig, x=0.5, y=0.0):
    fig.legend([plt.Line2D([], [], color=INK, lw=0.8)], ["observed"], loc="lower center", bbox_to_anchor=(x, y),
               frameon=False, handlelength=1.2, handletextpad=0.4, fontsize=8, borderaxespad=0)


def _stacked_fit(ax, film, cat, reviewed, lo, hi, scale):
    film = film.loc[lo:hi]
    mz = film.index.values
    groups = {k: np.zeros(len(mz)) for k in CATEGORY_ORDER}
    for name in film.columns[1:]:
        groups[cat[name] if name in reviewed else "removed"] += film[name].values
    bottom = np.zeros(len(mz))
    for k in CATEGORY_ORDER:
        if groups[k].any():
            ax.bar(mz, groups[k] / scale, bottom=bottom / scale, width=0.8, color=CATEGORY_COLORS[k], lw=0)
            bottom += groups[k]
    ax.stairs(film.observed.values / scale, np.r_[mz - 0.5, mz[-1] + 0.5], baseline=0, color=INK, lw=0.7)
    ax.set_xlim(lo - 0.5, hi + 0.5)


def panel_film_fit(fig, film, cat, reviewed, title, rect=(0.15, 0.82), y_labels=True, film_label=None):
    left, width = rect
    gap = 0.012
    w_hi = width * 0.17
    ax = fig.add_axes([left, 0.2, width - w_hi - gap, 0.6])
    ax_hi = fig.add_axes([left + width - w_hi, 0.2, w_hi, 0.6], sharey=ax)
    scale = 1e-10
    _stacked_fit(ax, film, cat, reviewed, *SPECTRUM_MZ, scale)
    _stacked_fit(ax_hi, film, cat, reviewed, *HIGH_MZ, scale)
    ax.set_yscale("function", functions=(np.sqrt, np.square))
    obs = film.observed / scale
    top = float(obs.drop(index=H2_MZ).max()) * 1.5
    ax.set_ylim(0, top)
    if obs[H2_MZ] > top:
        ax.text(H2_MZ + 0.9, top * 0.97, "H$_2$", ha="left", va="top", fontsize=8,
                color=INK)
        for dy in (0.93, 0.955):
            ax.plot([H2_MZ - 0.7, H2_MZ + 0.7], [top * dy - top * 0.01, top * dy + top * 0.01], color="white",
                    lw=1.2, zorder=5)
    ax.set_yticks([t for t in (0.5, 2, 5, 10, 20, 50) if t < top])
    ax.set_title(title, loc="left", pad=3, fontweight="bold")
    if film_label:
        ax_hi.text(1.0, 0.98, film_label, transform=ax_hi.transAxes, ha="right", va="top", fontsize=8, color=INK)
    if y_labels:
        ax.set_ylabel("Ion current\n(10$^{-10}$ Torr, √ scale)")
    else:
        plt.setp(ax.get_yticklabels(), visible=False)
    ax.set_xlabel("m/z", x=(width * 0.5) / (width - w_hi - gap))
    ax.set_xticks([10, 20, 30, 40, 50])
    ax_hi.set_xticks([127, 142])
    plt.setp(ax_hi.get_yticklabels(), visible=False)
    for a in (ax, ax_hi):
        style_axis(a)
    ax_hi.spines["left"].set_visible(False)
    ax_hi.tick_params(axis="y", length=0)
    for a, x in ((ax, 1.0), (ax_hi, 0.0)):
        dx = 0.004 / a.get_position().width
        a.plot([x - dx, x + dx], [-0.025, 0.025], transform=a.transAxes, color=INK, lw=0.7, clip_on=False,
               zorder=10)


def panel_quantification(fig, pressures, runs, cat, formulas):
    runs = runs[runs.panel_d].set_index("run_id").copy()
    runs["r"] = runs.mixing_ratio.map(ratio_value)
    runs = runs.sort_values(["metal", "r", "anneal_c"])
    P = pressures.loc[runs.index]
    present = [n for n in P.columns if cat[n] in ("film", "both") and (P[n] > 0).mean() >= MIN_PRESENT]
    species = sorted(present, key=lambda n: -P[n].median())[:MAX_ROWS]
    x, pos, prev = [], 0.0, None
    for metal, ratio in zip(runs.metal, runs.mixing_ratio):
        if prev is not None and (metal, ratio) != prev:
            pos += 2.0 if metal != prev[0] else 0.8
        x.append(pos)
        prev = (metal, ratio)
        pos += 1.0
    x = np.array(x)
    ax_bar = fig.add_axes([0.115, 0.715, 0.79, 0.185])
    ax_hm = fig.add_axes([0.115, 0.17, 0.79, 0.52], sharex=ax_bar)
    plt.setp(ax_bar.get_xticklabels(), visible=False)
    ax_bar.tick_params(axis="x", length=0)
    ax_bar.bar(x, runs.outgas_area_torr_s / 1e-7, width=0.8, color=SLATE, lw=0)
    ax_bar.set_ylabel("Outgassing\n(10$^{-7}$ Torr s)")
    style_axis(ax_bar)
    cmap = HEATMAP_CMAP.copy()
    cmap.set_bad("white")
    norm = LogNorm(vmin=1e-12, vmax=5e-9)
    M = P[species].T.values
    M = np.where(M > 0, M, np.nan)
    for i, xi in enumerate(x):
        im = ax_hm.pcolormesh([xi - 0.5, xi + 0.5], np.arange(len(species) + 1) - 0.5, M[:, [i]], cmap=cmap,
                              norm=norm, edgecolors="none")
    empty_row, empty_col = np.nonzero(np.isnan(M))
    ax_hm.plot(x[empty_col], empty_row, ls="", marker="x", ms=3.2, mew=0.5, color=NOT_FITTED)
    x0 = x.min() - 0.7
    for k, n in enumerate(species):
        ax_hm.add_patch(Rectangle((x0 - 1.1, k - 0.4), 0.8, 0.8, fc=CATEGORY_COLORS[cat[n]], ec="none",
                                  clip_on=False))
    ax_hm.set_ylim(len(species) - 0.5, -0.5)
    ax_hm.set_yticks(range(len(species)), [ISOMER_LABELS.get(n, formula_tex(formulas[n])) for n in species])
    ax_hm.tick_params(axis="y", pad=13, length=0)
    ax_hm.tick_params(axis="x", length=1.5, pad=1)
    for side in ("top", "right", "bottom", "left"):
        ax_hm.spines[side].set_visible(False)
    ax_hm.set_xlim(x0, x.max() + 0.7)
    ax_hm.set_xticks(x, runs.anneal_c.astype(str), rotation=90)
    tr = ax_hm.get_xaxis_transform()
    y_ratio = -0.155
    for (_, ratio), xs in runs.assign(x=x).groupby(["metal", "mixing_ratio"], sort=False).x:
        lo, hi = xs.min() - 0.5, xs.max() + 0.5
        ax_hm.add_patch(Rectangle((lo, -0.5), hi - lo, len(species), fc="none", ec=RULE_COLOR, lw=0.4))
        ax_hm.plot([lo + 0.1, hi - 0.1], [y_ratio + 0.015] * 2, transform=tr, color=MUTED, lw=0.6, clip_on=False)
        ax_hm.text((lo + hi) / 2, y_ratio, ratio, transform=tr, ha="center", va="top", fontsize=8)
    ax_hm.text(np.mean([x.min(), x.max()]), y_ratio - 0.08, "Annealing temperature (°C), grouped by organic:metal ratio",
               transform=tr, ha="center", va="top", fontsize=8.5)
    for metal, xs in runs.assign(x=x).groupby("metal", sort=False).x:
        lo, hi = xs.min() - 0.4, xs.max() + 0.4
        ax_bar.plot([lo, lo, hi, hi], [1.0, 1.04, 1.04, 1.0], transform=ax_bar.get_xaxis_transform(),
                    color=MUTED, lw=0.6, clip_on=False)
        ax_bar.text((lo + hi) / 2, 1.07, f"{METAL_LABEL[metal]} ({len(xs)} films)",
                    transform=ax_bar.get_xaxis_transform(), ha="center", va="bottom", fontsize=10, fontweight="bold",
                    color=INK)
    ax_hm.set_ylabel("Film species", labelpad=4)
    cb = fig.colorbar(im, cax=ax_hm.inset_axes([1.012, 0.0, 0.015, 1.0]))
    cb.set_label("Partial pressure (Torr)")
    cb.ax.tick_params(labelsize=8, width=0.5, length=2)
    cb.outline.set_linewidth(0.5)
    cb.ax.add_patch(Rectangle((0, -0.125), 1, 0.04, transform=cb.ax.transAxes, fc="white", ec=INK, lw=0.5,
                              clip_on=False))
    cb.ax.plot(0.5, -0.105, transform=cb.ax.transAxes, marker="x", ms=3.2, mew=0.5, color=NOT_FITTED, clip_on=False)
    cb.ax.text(1.5, -0.105, "not fitted", transform=cb.ax.transAxes, ha="left", va="center", fontsize=8)


def label(fig, letter, y=1.0):
    fig.text(0.0, y, letter.upper(), fontsize=11, fontweight="bold", va="top", ha="left")


def load_data():
    summary = json.loads((DATA_DIR / "fig04_rga_summary.json").read_text())
    library = pd.read_csv(DATA_DIR / "fig04_rga_library.csv")
    runs = pd.read_csv(DATA_DIR / "fig04_rga_runs.csv")
    film1 = pd.read_csv(DATA_DIR / "fig04_rga_fit_before_review.csv", index_col=0)
    film2 = pd.read_csv(DATA_DIR / "fig04_rga_fit_after_review.csv", index_col=0)
    pressures = pd.read_csv(DATA_DIR / "fig04_rga_partial_pressures.csv", index_col=0)
    return summary, library, runs, film1, film2, pressures


def main():
    apply_publication_style()
    summary, library, runs, film1, film2, pressures = load_data()
    generated = set(library.compound)
    reviewed = set(library.compound[library.kept_after_review])
    cat = dict(zip(library.compound, library.source))
    formulas = dict(zip(library.compound, library.formula))
    rep_row = runs.set_index("run_id").loc[summary["representative_run"]]

    film_label = f"{rep_row.tf}: {METAL_LABEL[rep_row.metal]} {rep_row.mixing_ratio}, {rep_row.anneal_c} °C"
    W = FIG_WIDTH_IN
    heights = {"a": 2.6, "gap": 0.08, "bc": 2.2, "d": 3.3}
    panels = {
        "a": lambda f: panel_flow(f, summary["chemistry"], summary["n_fragments"], cat, generated, reviewed),
        "b": lambda f: panel_film_fit(f, film1, cat, reviewed, f"Before review: {len(generated)} compounds",
                                      rect=(0.17, 0.8), film_label=film_label),
        "c": lambda f: panel_film_fit(f, film2, cat, reviewed, f"After review: {len(reviewed)} compounds",
                                      rect=(0.04, 0.939), y_labels=False, film_label=film_label),
        "d": lambda f: panel_quantification(f, pressures, runs, cat, formulas),
    }

    fig = plt.figure(figsize=(W, sum(heights.values())))
    top, _, mid, bot = fig.subfigures(4, 1, height_ratios=[heights[k] for k in ("a", "gap", "bc", "d")], hspace=0)
    panels["a"](top)
    label(top, "a")
    for sub, key in zip(mid.subfigures(1, 2, wspace=0, width_ratios=[0.54, 0.46]), ("b", "c")):
        panels[key](sub)
        label(sub, key, y=0.885)
    category_legend(mid)
    observed_legend(mid, x=0.54, y=0.015)
    panels["d"](bot)
    label(bot, "d")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "figure_04.svg", dpi=RASTER_DPI, metadata={"Date": None})
    fig.savefig(OUTPUT_DIR / "figure_04.pdf", dpi=RASTER_DPI, metadata={"CreationDate": None})
    fig.savefig(OUTPUT_DIR / "figure_04.png", dpi=RASTER_DPI)
    plt.close(fig)
    print(f"wrote {OUTPUT_DIR / 'figure_04'}.svg, .pdf and .png")


if __name__ == "__main__":
    main()
