#!/usr/bin/env python3
"""Figure 3: image and UV-Vis spatial homogeneity of two exemplar films
(TF001028, TF003020) against the library-wide distributions, and the
10-spot UV-Vis + image UMAP of the films with complete data for both
modalities.

Input: data/exemplar_image_de00_maps_TF001028_TF003020.npz,
data/exemplar_uvvis_spectra_TF001028_TF003020.npz,
data/image_inhomogeneity_by_film.csv, data/uvvis_inhomogeneity_by_film.csv,
data/umap_clusters_by_film.csv. Output: figures/figure_03.{svg,pdf,png}.

186 mm wide, Arial, 7 pt minimum for full-size text. UMAP scatter points are
rasterized at 600 dpi; axes, text and contours stay vector, and SVG text
stays editable. All layout comes from the mm tables COLUMNS and ROWS below,
used by a single GridSpec.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize, PowerNorm, to_rgba
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator, NullFormatter
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / ".." / "data"
OUTPUT_DIR = HERE / ".." / "figures"

# ---------------------------------------------------------------- style
INK = "#122f3e"
RULE_COLOR = "#bccac9"
TEAL = "#207d83"
ORANGE = "#f36b21"
GOLD = "#edc238"
TF1028_COLOR = "#297e37"
TF3020_COLOR = "#aa0000"

TEAL_CMAP = LinearSegmentedColormap.from_list("paper_teal", ["#f4f9f9", "#a6d3d5", TEAL, INK])
ORANGE_CMAP = LinearSegmentedColormap.from_list(
    "paper_orange", ["#fff0e5", "#fbb98c", ORANGE, "#a63f0a", "#5c2206"]
)
GREEN_CMAP = LinearSegmentedColormap.from_list("paper_green", ["#e9f3e4", "#9cc98f", TF1028_COLOR, "#0f3d1c"])
PURPLE_CMAP = LinearSegmentedColormap.from_list(
    "paper_purple", ["#ece9fd", "#b7aefb", "#6b55e0", "#2a1a8a", "#170f4d"]
)
SCAN_POSITION_CMAP = LinearSegmentedColormap.from_list("paper_scan", [INK, TEAL, "#6aa86b", GOLD])

PUBLICATION_STYLE = {
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.family": "Arial",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.default": "regular",
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
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


# ---------------------------------------------------------------- layout
MM = 1 / 25.4
RASTER_DPI = 600
BOX = 21.0
FILM_COLORS = {"TF001028": TF1028_COLOR, "TF003020": TF3020_COLOR}
FILM_LABELS = {"TF001028": "TF1028", "TF003020": "TF3020"}
FILM_LABEL_HEIGHTS = {"TF001028": 0.80, "TF003020": 0.62}
FILMS = list(FILM_COLORS)

# Lengths in mm. Every edge in the figure comes from these two tables, which is what keeps panels aligned.
COLUMNS = {
    "a_label": 7.5, "a1": BOX, "a_gap": 4.0, "a2": BOX, "ab_gutter": 3.4,
    "b_label": 7.5, "b1": BOX, "b_gap": 4.0, "b2": BOX, "bc_gutter": 3.4,
    "c_label": 7.5, "c1": BOX, "cb1_gap": 0.8, "cb1": 1.4, "cb1_label": 8.8,
    "c2": BOX, "cb2_gap": 0.8, "cb2": 1.4, "c_right": 9.5,
}
ROWS = {
    "header": 5.0,
    "title1": 4.0, "row1": BOX, "xlabel1": 7.0,
    "title2": 4.0, "row2": BOX, "xlabel2": 7.0,
    "title3": 4.0, "row3": BOX, "xlabel3": 7.0,
    "hist_gap": 1.0, "row4": BOX, "bottom": 8.0,
}
FIGURE_WIDTH_MM = 186.0

# Slot order on the UMAP chains spatial neighbours (C0, C1, C2, C3, C7, C4, C6, C5, C8) for the CVD check.
CLUSTER_COLORS = {
    0: "#3766b5",
    1: "#3fb489",
    2: "#00707f",
    3: "#4f8f1f",
    7: "#e0a616",
    4: "#d9631e",
    6: "#85590a",
    5: "#c0408f",
    8: "#8a7be0",
}

UMAP_X = "umap_10spot_eq77_1"
UMAP_Y = "umap_10spot_eq77_2"
UMAP_XLIM = (-3.4, 10.4)
UMAP_YLIM = (-0.2, 10.8)

DELTA_E = r"$\Delta E_{00}$"
LINE_SHADE_RANGE = (0.25, 1.0)
CONTOUR_SIGMA = 2.0
CONTOUR_START = 3.0
CONTOUR_STEP = 2.5


def truncate_cmap(cmap, start=0.0, stop=1.0):
    return LinearSegmentedColormap.from_list(f"{cmap.name}_cut", cmap(np.linspace(start, stop, 256)))


def scatter_cmap(cmap):
    """Drop the near-white end so low-valued scatter points stay visible on the white background."""
    return truncate_cmap(cmap, start=0.14)


# The ramp stops short of its darkest end so contour lines stay visible over the highest dE00 regions.
DE_MAP_CMAP = truncate_cmap(ORANGE_CMAP, stop=0.72)


DESCRIPTORS = [
    {
        "column": "delta_e_2000_median",
        "cmap": scatter_cmap(TEAL_CMAP),
        "norm": LogNorm(vmin=0.30, vmax=5.0),
        "ticks": [0.5, 1, 2, 4],
        "label": f"Image {DELTA_E} median",
    },
    {
        "column": "uvvis_robust_spectral_inhomogeneity",
        "cmap": scatter_cmap(ORANGE_CMAP),
        "norm": LogNorm(vmin=1.5e-3, vmax=2.5e-1),
        "ticks": [0.01, 0.1],
        "label": "UV-Vis inhomogeneity",
    },
    {
        "column": "uv_abs_span_10spot",
        "cmap": scatter_cmap(GREEN_CMAP),
        "norm": Normalize(vmin=0.02, vmax=1.25),
        "ticks": [0.3, 0.6, 0.9, 1.2],
        "label": "Optical contrast (ΔA)",
    },
    {
        "column": "shape_centroid_nm_10spot",
        "cmap": scatter_cmap(PURPLE_CMAP),
        "norm": Normalize(vmin=425.0, vmax=505.0),
        "ticks": [440, 460, 480, 500],
        "label": "Spectral centroid (nm)",
    },
]


def load_data():
    with np.load(DATA_DIR / "exemplar_image_de00_maps_TF001028_TF003020.npz") as bundle:
        images = {
            film: {
                "full_rgb": bundle[f"{film}_full_rgb"],
                "roi_rgb": bundle[f"{film}_roi_rgb"],
                "bounds": bundle[f"{film}_roi_bounds"].astype(int),
                "de00": bundle[f"{film}_delta_e_2000"].astype(float),
            }
            for film in FILMS
        }
    with np.load(DATA_DIR / "exemplar_uvvis_spectra_TF001028_TF003020.npz") as bundle:
        wavelengths = bundle["wavelengths"].astype(float)
        spectra = {
            film: {
                "intensity": bundle[f"{film}_smoothed_intensity"].astype(float),
                "absorbance": bundle[f"{film}_smoothed_absorbance"].astype(float),
                "difference": bundle[f"{film}_difference_matrix"].astype(float),
            }
            for film in FILMS
        }
    image_scores = pd.read_csv(DATA_DIR / "image_inhomogeneity_by_film.csv")
    image_scores = image_scores[image_scores["qc_valid"].fillna(False).astype(bool)]
    uvvis_scores = pd.read_csv(DATA_DIR / "uvvis_inhomogeneity_by_film.csv")
    umap = pd.read_csv(DATA_DIR / "umap_clusters_by_film.csv", low_memory=False)
    return images, wavelengths, spectra, image_scores, uvvis_scores, umap


def make_layout():
    assert abs(sum(COLUMNS.values()) - FIGURE_WIDTH_MM) < 1e-9, sum(COLUMNS.values())
    height_mm = sum(ROWS.values())
    fig = plt.figure(figsize=(FIGURE_WIDTH_MM * MM, height_mm * MM))
    grid = GridSpec(
        len(ROWS), len(COLUMNS), figure=fig,
        width_ratios=list(COLUMNS.values()), height_ratios=list(ROWS.values()),
        left=0, right=1, bottom=0, top=1, wspace=0, hspace=0,
    )
    return fig, grid


def cell(grid, row, column):
    """GridSpec slot from row/column names, or (first, last) name tuples for spans."""
    rows, columns = list(ROWS), list(COLUMNS)
    r0, r1 = (row, row) if isinstance(row, str) else row
    c0, c1 = (column, column) if isinstance(column, str) else column
    return grid[rows.index(r0): rows.index(r1) + 1, columns.index(c0): columns.index(c1) + 1]


def centered_text(fig, spec, text, **kwargs):
    box = spec.get_position(fig)
    fig.text((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2, text, ha="center", va="center", **kwargs)


def bare_axis(fig, spec):
    axis = fig.add_subplot(spec)
    axis.set_xticks([])
    axis.set_yticks([])
    return axis


def despine(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def center_square(rgb):
    """Crop to a centered square, returning the (row, column) offset; some well photos are off-square by a few px."""
    height, width = rgb.shape[:2]
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    return rgb[top: top + side, left: left + side], top, left


def plot_image_rows(fig, grid, images):
    vmax = max(float(np.percentile(rec["de00"], 99.5)) for rec in images.values())
    norm = PowerNorm(0.5, vmin=0, vmax=vmax)
    levels = np.arange(CONTOUR_START, vmax, CONTOUR_STEP)
    axes = {}
    for film, column in zip(FILMS, ("a1", "a2")):
        rec = images[film]
        left, top, side = rec["bounds"]
        photo, crop_top, crop_left = center_square(rec["full_rgb"])

        axis = bare_axis(fig, cell(grid, "row1", column))
        axis.imshow(photo, interpolation="none")
        axis.add_patch(Rectangle(
            (left - crop_left - 0.5, top - crop_top - 0.5), side, side,
            fill=False, edgecolor=INK, linestyle="--", linewidth=0.6,
        ))
        axes[f"photo_{film}"] = axis

        axis = bare_axis(fig, cell(grid, "row2", column))
        axis.imshow(rec["roi_rgb"], interpolation="none")
        axes[f"roi_{film}"] = axis

        axis = bare_axis(fig, cell(grid, "row3", column))
        axis.imshow(rec["de00"], cmap=DE_MAP_CMAP, norm=norm, interpolation="none")
        smooth = gaussian_filter(rec["de00"], sigma=CONTOUR_SIGMA)
        if levels.size and smooth.max() >= levels[0]:
            axis.contour(smooth, levels=levels, colors=INK, linewidths=0.3, alpha=0.8)
        axes[f"de_map_{film}"] = axis
    return axes


def deviation_shades(block):
    deviation = np.sqrt(((block - block.mean(axis=0)) ** 2).mean(axis=1))
    low, high = LINE_SHADE_RANGE
    return ORANGE_CMAP(low + (high - low) * deviation / deviation.max())


def plot_spectra_rows(fig, grid, wavelengths, spectra):
    intensity_max = max(rec["intensity"].max() for rec in spectra.values())
    exponent = int(np.floor(np.log10(intensity_max)))
    scale = 10 ** exponent
    intensity_label = rf"Intensity ($\times 10^{{{exponent}}}$ a.u.)"
    difference_max = max(rec["difference"].max() for rec in spectra.values())
    axes = {}

    for index, (film, column) in enumerate(zip(FILMS, ("b1", "b2"))):
        rec = spectra[film]
        is_left = index == 0

        axis = fig.add_subplot(cell(grid, "row1", column))
        block = rec["intensity"] / scale
        colors = SCAN_POSITION_CMAP(np.linspace(0, 1, block.shape[0]))
        # Scan position 1 is the darkest shade; drawing it last keeps darker lines on top of lighter ones.
        for position in reversed(range(block.shape[0])):
            axis.plot(wavelengths, block[position], color=colors[position], linewidth=0.7)
        axis.set_ylim(0, intensity_max / scale * 1.05)
        axis.yaxis.set_major_locator(MultipleLocator(1))
        axes[f"intensity_{film}"] = axis

        axis_abs = fig.add_subplot(cell(grid, "row2", column))
        for line, color in zip(rec["absorbance"], deviation_shades(rec["absorbance"])):
            axis_abs.plot(wavelengths, line, color=color, linewidth=0.7)
        axis_abs.set_ylim(0, 1.3)
        axis_abs.yaxis.set_major_locator(MultipleLocator(0.4))
        axis_abs.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 10:g}"))
        axes[f"absorbance_{film}"] = axis_abs

        for line_axis, ylabel in ((axis, intensity_label), (axis_abs, r"Absorbance ($\times 10^{-1}$)")):
            despine(line_axis)
            line_axis.set_xlim(wavelengths.min(), wavelengths.max())
            line_axis.xaxis.set_major_locator(MultipleLocator(100))
            line_axis.xaxis.set_minor_locator(MultipleLocator(50))
            line_axis.set_xlabel("Wavelength (nm)")
            if is_left:
                line_axis.set_ylabel(ylabel)
            else:
                line_axis.tick_params(labelleft=False)

        axis_matrix = fig.add_subplot(cell(grid, "row3", column))
        size = rec["difference"].shape[0]
        edges = np.arange(size + 1)
        mesh = axis_matrix.pcolormesh(
            edges, edges, rec["difference"], cmap=ORANGE_CMAP,
            norm=PowerNorm(0.3, vmin=0, vmax=difference_max), shading="flat",
        )
        mesh.set_edgecolor("face")
        mesh.set_linewidth(0.2)
        axis_matrix.invert_yaxis()
        ticks = [0.5, 4.5, 9.5]
        axis_matrix.set_xticks(ticks, ["1", "5", "10"])
        axis_matrix.set_yticks(ticks, ["1", "5", "10"] if is_left else [])
        axis_matrix.set_xlabel("Scan position")
        if is_left:
            axis_matrix.set_ylabel("Scan position")
        axes[f"matrix_{film}"] = axis_matrix
    return axes


def plot_histogram(axis, values, bins, color, highlights, xlabel, xlim, yticks, median_side, film_sides):
    """film_sides maps each film to the side of its marker line that its label sits on."""
    finite = values[np.isfinite(values) & (values > 0)]
    counts, _, _ = axis.hist(finite, bins=bins, color=color, edgecolor="white", linewidth=0.15)
    axis.set_xscale("log")
    axis.set_xlim(xlim)
    axis.set_ylim(0, counts.max() * 1.2)
    axis.set_yticks(yticks)
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 100:g}"))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.set_xlabel(xlabel)
    axis.set_ylabel(r"Thin films ($\times 10^{2}$)")

    median = float(np.median(finite))
    axis.axvline(median, color=INK, linewidth=0.8, linestyle="--")
    offset = {"left": (-2, -1), "right": (2, -1)}
    axis.annotate(
        f"median {median:.3f}", xy=(median, 1), xycoords=("data", "axes fraction"),
        xytext=offset[median_side], textcoords="offset points", va="top",
        ha="right" if median_side == "left" else "left",
    )
    for film, score in highlights.items():
        axis.axvline(score, color="black", linewidth=1.0, linestyle=":")
        side = film_sides[film]
        axis.annotate(
            FILM_LABELS[film], xy=(score, FILM_LABEL_HEIGHTS[film]), xycoords=("data", "axes fraction"),
            xytext=offset[side], textcoords="offset points", va="center",
            ha="right" if side == "left" else "left", color=FILM_COLORS[film],
        )


def plot_histograms(fig, grid, image_scores, uvvis_scores):
    image_axis = fig.add_subplot(cell(grid, "row4", ("a1", "a2")))
    by_film = image_scores.set_index("ThinFilmID")["delta_e_2000_median"]
    plot_histogram(
        image_axis, image_scores["delta_e_2000_median"].to_numpy(),
        bins=np.logspace(-1, 1, 61), color=TEAL,
        highlights={film: float(by_film[film]) for film in FILMS},
        xlabel=f"Image {DELTA_E} median", xlim=(0.1, 10), yticks=[0, 200, 400, 600],
        median_side="left", film_sides={"TF001028": "right", "TF003020": "left"},
    )
    uvvis_axis = fig.add_subplot(cell(grid, "row4", ("b1", "b2")))
    column = "uvvis_robust_spectral_inhomogeneity"
    by_film = uvvis_scores.set_index("ThinFilmID")[column]
    plot_histogram(
        uvvis_axis, uvvis_scores[column].to_numpy(),
        bins=np.logspace(np.log10(4e-4), np.log10(2.5), 61), color=ORANGE,
        highlights={film: float(by_film[film]) for film in FILMS},
        xlabel="UV-Vis inhomogeneity score", xlim=(3e-4, 2), yticks=[0, 100, 200, 300],
        median_side="right", film_sides={"TF001028": "left", "TF003020": "right"},
    )
    return {"hist_image": image_axis, "hist_uvvis": uvvis_axis}


def pin_films(axis, umap, size):
    for film, color in FILM_COLORS.items():
        row = umap[umap["ThinFilmID"] == film].iloc[0]
        axis.scatter(
            row[UMAP_X], row[UMAP_Y], s=size, facecolors=color, edgecolors="white",
            linewidths=0.5, zorder=10,
        )
        axis.scatter(
            row[UMAP_X], row[UMAP_Y], s=size * 1.25, facecolors="none", edgecolors=INK,
            linewidths=0.35, zorder=11,
        )


def add_density_contours(axis, x, y):
    hist, x_edges, y_edges = np.histogram2d(x, y, bins=90, range=[UMAP_XLIM, UMAP_YLIM])
    smooth = gaussian_filter(hist.T, sigma=1.5)
    levels = np.unique(np.percentile(smooth[smooth > 0], [45, 70, 88, 96]))
    axis.contour(
        (x_edges[:-1] + x_edges[1:]) / 2, (y_edges[:-1] + y_edges[1:]) / 2, smooth,
        levels=levels, colors=INK, linewidths=0.3, alpha=0.25, zorder=5,
    )


def plot_hero_umap(fig, grid, umap):
    axis = fig.add_subplot(cell(grid, ("row1", "row2"), ("c1", "c2")))
    x, y = umap[UMAP_X].to_numpy(), umap[UMAP_Y].to_numpy()
    raw, assigned = umap["cluster_k9_raw"].to_numpy(int), umap["cluster_k9_assigned"].to_numpy(int)

    # One shuffled collection gives a single raster image and avoids any cluster systematically painting over another.
    is_core = raw >= 0
    rgba = np.array([
        to_rgba(CLUSTER_COLORS[cluster], 0.75 if core else 0.25)
        for cluster, core in zip(assigned, is_core)
    ])
    rng = np.random.default_rng(42)
    order = np.concatenate([rng.permutation(np.flatnonzero(~is_core)), rng.permutation(np.flatnonzero(is_core))])
    axis.scatter(x[order], y[order], s=1.5, c=rgba[order], linewidths=0, rasterized=True, zorder=3)

    add_density_contours(axis, x, y)
    pin_films(axis, umap, size=16)
    axis.set_xlim(UMAP_XLIM)
    axis.set_ylim(UMAP_YLIM)
    axis.set_xticks([-2, 0, 2, 4, 6, 8, 10])
    axis.set_yticks([0, 2, 4, 6, 8, 10])
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    return {"umap": axis}


def plot_descriptor_map(fig, axis_spec, cax_spec, umap, config, show_ylabel):
    axis = bare_axis(fig, axis_spec)
    values = umap[config["column"]].to_numpy(float)
    x, y = umap[UMAP_X].to_numpy(), umap[UMAP_Y].to_numpy()
    valid = np.flatnonzero(np.isfinite(values))
    order = valid[np.argsort(values[valid])]
    points = axis.scatter(
        x[order], y[order], c=np.clip(values[order], config["norm"].vmin, config["norm"].vmax),
        cmap=config["cmap"], norm=config["norm"], s=0.7, linewidths=0, rasterized=True, zorder=3,
    )
    pin_films(axis, umap, size=7)
    axis.set_xlim(UMAP_XLIM)
    axis.set_ylim(UMAP_YLIM)
    axis.set_xlabel("UMAP 1")
    if show_ylabel:
        axis.set_ylabel("UMAP 2")

    cax = fig.add_subplot(cax_spec)
    colorbar = fig.colorbar(points, cax=cax)
    colorbar.solids.set_rasterized(True)
    colorbar.outline.set_linewidth(0.5)
    cax.yaxis.set_major_locator(FixedLocator(config["ticks"]))
    cax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    cax.yaxis.set_minor_locator(FixedLocator([]))
    cax.yaxis.set_minor_formatter(NullFormatter())
    cax.tick_params(length=1.5, pad=1)
    colorbar.set_label(config["label"], labelpad=1.5)
    return axis, cax


def plot_descriptor_maps(fig, grid, umap):
    slots = [("row3", "c1", "cb1"), ("row3", "c2", "cb2"), ("row4", "c1", "cb1"), ("row4", "c2", "cb2")]
    axes = {}
    for (row, column, cax_column), config in zip(slots, DESCRIPTORS):
        axis, cax = plot_descriptor_map(
            fig, cell(grid, row, column), cell(grid, row, cax_column), umap, config, column == "c1",
        )
        axes[f"map_{config['column']}"] = axis
        axes[f"cbar_{config['column']}"] = cax
    return axes


def add_annotations(fig, grid):
    for column, film in zip(("a1", "a2", "b1", "b2"), FILMS * 2):
        centered_text(
            fig, cell(grid, "header", column), FILM_LABELS[film],
            fontsize=8, fontweight="bold", color=FILM_COLORS[film],
        )
    titles = {
        ("title1", ("a1", "a2")): "Sample well image",
        ("title2", ("a1", "a2")): "ROI image crop",
        ("title3", ("a1", "a2")): f"Image {DELTA_E} map",
        ("title1", ("b1", "b2")): "Raw UV-Vis spectra",
        ("title2", ("b1", "b2")): "Computed absorbance",
        ("title3", ("b1", "b2")): "Absorbance difference matrix",
        ("title1", ("c1", "c2")): "UMAP space",
        ("title3", ("c1", "c2")): "Physical descriptors map",
    }
    for (row, columns), text in titles.items():
        centered_text(fig, cell(grid, row, columns), text, fontsize=8)

    top = cell(grid, "header", "a_label").get_position(fig)
    separators = {}
    for name, gutter, letter_column in (("B", "ab_gutter", "b_label"), ("C", "bc_gutter", "c_label")):
        box = cell(grid, "header", gutter).get_position(fig)
        separators[name] = box.x0 + 0.6 * (box.x1 - box.x0)
        fig.add_artist(Line2D(
            [separators[name]] * 2, [0.012, 0.988], transform=fig.transFigure, color=RULE_COLOR, linewidth=0.6,
        ))
    letter_y = (top.y0 + top.y1) / 2
    inset = 1.0 / FIGURE_WIDTH_MM
    for letter, x in (("A", 0.0), ("B", separators["B"]), ("C", separators["C"])):
        fig.text(x + inset, letter_y, letter, fontsize=10, fontweight="bold", ha="left", va="center")


def build_figure():
    apply_publication_style()
    images, wavelengths, spectra, image_scores, uvvis_scores, umap = load_data()
    fig, grid = make_layout()
    axes = {}
    axes.update(plot_image_rows(fig, grid, images))
    axes.update(plot_spectra_rows(fig, grid, wavelengths, spectra))
    axes.update(plot_histograms(fig, grid, image_scores, uvvis_scores))
    axes.update(plot_hero_umap(fig, grid, umap))
    axes.update(plot_descriptor_maps(fig, grid, umap))
    add_annotations(fig, grid)
    fig.align_ylabels([axes[f"{name}_{FILMS[0]}"] for name in ("intensity", "absorbance", "matrix")] + [axes["hist_uvvis"]])
    return fig


def main():
    fig = build_figure()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "figure_03.svg", dpi=RASTER_DPI, metadata={"Date": None})
    fig.savefig(OUTPUT_DIR / "figure_03.pdf", dpi=RASTER_DPI, metadata={"CreationDate": None})
    fig.savefig(OUTPUT_DIR / "figure_03.png", dpi=300)
    plt.close(fig)
    print(f"wrote {OUTPUT_DIR / 'figure_03'}.svg, .pdf and .png")


if __name__ == "__main__":
    main()
