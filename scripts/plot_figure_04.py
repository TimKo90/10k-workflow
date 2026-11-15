#!/usr/bin/env python3
"""Figure 4: GIWAXS pipeline for one film, and the labels across the UV-Vis/image UMAP.

A: raw detector image -> q-space map -> labelling for TF001028 (film 1; the homogeneous
exemplar film of Figure 3; ALS 7.3.3, 1.0° incident angle, 5 s). Reduction (pyGID) uses the
calibration attached to the GIWAXS_Batch; the image is divided by the diode reading first,
so the q-map is in counts / diode. The label stage shows the share of all labelled films in
each class as two full-length tracks, crystalline filled from the left and amorphous from
the right.
B: the amorphous and the crystalline films on the UMAP of Figure 3C (only films that also
have UV-Vis and image data), each with example q-maps of Figure S14 benchmark films on the
colour scale of the q-map in A. Numbered discs tie each q-map to its film on the UMAP.

Layout: pipeline on top, one block per class below.

Input: data/fig04_pipeline_TF001028.npz, data/fig04_example_qmaps.npz, data/giwaxs_class_by_film.csv,
data/umap_clusters_by_film.csv, data/logos/mlexchange_logo.png.
Output: figures/figure_04.{svg,pdf,png}.

Images use the ink-blue sequential ramp of the Figure 5D heatmap. The raw image is shown without the
detector mask; module gaps and q-map areas without data are white.

fig04_pipeline_TF001028.npz holds the raw detector frame of TF001028 as recorded (counts per
pixel; gap pixels are negative) with its detector mask, and the reduced q-map and q axes at
full resolution, plus wavelength, sample-detector distance and beam center from the beamline
reduction. fig04_example_qmaps.npz holds the q-maps of the six example films, 2x2 binned.
Of the GIWAXS data, only the scans shown in the figure are included.

The MLExchange logo is a crop of docs/docs/figures/mlex.png in
https://github.com/mlexchange/mlex (BSD-3-Clause-style licence, Copyright (c) 2021,
The Regents of the University of California, through Lawrence Berkeley National
Laboratory).

All positions and sizes are in millimetres from the bottom-left corner of the figure.
"""
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import LogFormatterSciNotation, LogLocator, NullFormatter
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / ".." / "data"
OUTPUT_DIR = HERE / ".." / "figures"
PIPELINE_FILE = DATA_DIR / "fig04_pipeline_TF001028.npz"
EXAMPLE_QMAPS_FILE = DATA_DIR / "fig04_example_qmaps.npz"
LABELS_FILE = DATA_DIR / "giwaxs_class_by_film.csv"
UMAP_FILE = DATA_DIR / "umap_clusters_by_film.csv"
LOGO_FILE = DATA_DIR / "logos" / "mlexchange_logo.png"

# --- Colours and fonts -----------------------------------------------------------------------------------------
INK = "#122f3e"
MUTED = "#4a5f6b"
# Colour of the UMAP films outside the highlighted class.
CONTEXT_GREY = "#e3e8e8"
# Class colours from the paper palette: teal and orange.
CLASS_COLORS = {"Crystalline": "#207d83", "Amorphous": "#f36b21"}
# Ink-blue sequential ramp of the Figure 5D heatmap (light = low, dark = high).
IMAGE_CMAP = LinearSegmentedColormap.from_list("ink_blue", ["#eef4f7", "#b8d2da", "#6f9db3", "#33586f", INK])
NO_DATA_COLOR = "white"
MONO_FONT = "Courier New"

PANEL_LETTER_SIZE = 10
HEADING_SIZE = 8
BODY_SIZE = 7

STYLE = {
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.family": "Arial", "font.size": BODY_SIZE,
    "axes.labelsize": BODY_SIZE, "axes.titlesize": BODY_SIZE, "xtick.labelsize": BODY_SIZE,
    "ytick.labelsize": BODY_SIZE, "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 0.5, "xtick.major.width": 0.5,
    "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2, "xtick.major.pad": 1.5,
    "ytick.major.pad": 1.5, "axes.labelpad": 1.5,
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold", "mathtext.default": "regular",
}
QXY_LABEL = r"$\mathit{q}_{xy}$ (Å$^{-1}$)"
QZ_LABEL = r"$\mathit{q}_{z}$ (Å$^{-1}$)"

# --- Data ranges -----------------------------------------------------------------------------------------------
# Colour limits of the raw image, in counts per pixel.
RAW_COUNT_LIMITS = (1e2, 1e4)
# Colour limits of every q-map in the figure, in counts / diode (one shared scale).
QMAP_INTENSITY_LIMITS = (3, 1000)
# Axis limits of the example q-maps, in Å^-1.
EXAMPLE_QXY_LIMITS = (-1.4, 3.3)
EXAMPLE_QZ_LIMITS = (0, 3.2)
QXY_TICKS = [-1, 0, 1, 2, 3]
QZ_TICKS = [0, 1, 2, 3]
RAW_X_TICKS = [0, 700, 1400]
RAW_Y_TICKS = [0, 600, 1200]

# Same fixed limits as the UMAP of Figure 3C, so the maps line up.
UMAP_X_COLUMN, UMAP_Y_COLUMN = "umap_10spot_eq77_1", "umap_10spot_eq77_2"
UMAP_XLIM, UMAP_YLIM = (-3.4, 10.4), (-0.2, 10.8)
UMAP_DENSITY_BINS = 61
# Gaussian smoothing of the density, sigma in histogram bins.
UMAP_DENSITY_SMOOTHING = 1.5
# Contour levels as fractions of the peak density.
UMAP_CONTOUR_FRACTIONS = [0.12, 0.3, 0.55, 0.8]

# --- Film shown in panel A ---------------------------------------------------------------------------------------
PIPELINE_BADGE = 1
PIPELINE_INCIDENT_ANGLE_DEG = 1.0
PIPELINE_EXPOSURE_S = 5


class ExampleFilm(NamedTuple):
    film_id: str
    badge: int
    # Badge position relative to the film's UMAP point, in UMAP units.
    badge_offset: tuple


# Example films per class, from the benchmark set. The panel A film is marked on its class UMAP too.
EXAMPLE_FILMS = {
    # Both amorphous examples carry the label "Mostly Amorphous" and show one or two reflections.
    "Amorphous": [ExampleFilm("TF000325", 2, (-1.3, 1.1)), ExampleFilm("TF000327", 3, (1.3, 1.1))],
    # The crystalline examples are films of the 100-120 °C benchmark cohort of Figure S14; TF001838 and TF001894
    # are the scans whose profiles Figure S14 shows.
    "Crystalline": [ExampleFilm("TF004966", 4, (-1.2, 0.6)), ExampleFilm("TF001126", 5, (1.0, -1.2)),
                    ExampleFilm("TF001838", 6, (0.8, 1.2)), ExampleFilm("TF001894", 7, (-0.9, -1.4))],
}
PIPELINE_BADGE_OFFSET = (0.9, 1.2)

# --- Layout ------------------------------------------------------------------------------------------------------
MM_PER_INCH = 25.4
FIGURE_WIDTH, FIGURE_HEIGHT = 186.0, 103.5
LEFT_MARGIN = 10.5
RIGHT_EDGE = 184.0
PANEL_LETTER_X = 0.5

# Distance from the top of an image to the heading above it.
HEADING_GAP = 4.3
# Distance from the top of an image to the film ID above it.
FILM_LABEL_GAP = 0.6
# Distance of a badge centre from the top-left corner of an image.
BADGE_INSET = 2.4
COLORBAR_GAP, COLORBAR_WIDTH = 1.6, 2.0
# Room for the colorbar ticks and label.
COLORBAR_LABEL_ROOM = 12.0

# Panel A
PIPELINE_BOTTOM = 69.0
PIPELINE_IMAGE_HEIGHT = 25.0
# Offset of the scan info to the right of the film ID.
SCAN_INFO_OFFSET = 14.0
# Height of the pipeline arrows, as a fraction of the image height.
ARROW_HEIGHT_FRACTION = 0.62
REDUCTION_ARROW_LENGTH = 22.0
QMAP_AXIS_LABEL_ROOM = 9.5
LABELLING_ARROW_LENGTH = 17.5
LOGO_WIDTH = 8.5
# Gap between the labelling arrow and the class tracks.
TRACK_GAP = 3.5
TRACK_HEIGHT = 6.0
TRACK_CORNER_RADIUS = 1.3
# Track bottoms above the bottom of the pipeline row.
TRACK_BOTTOMS = {"Crystalline": 15.5, "Amorphous": 3.0}
TRACK_FILL_FROM_LEFT = {"Crystalline": True, "Amorphous": False}

# Panel B
PANEL_B_HEADING_Y = 55.3
CLASS_BLOCK_BOTTOM = 3.0
CLASS_BLOCK_GAP = 7.0
UMAP_WIDTH = 36.0
EXAMPLE_WIDTH, EXAMPLE_HEIGHT = 24.0, 15.8
EXAMPLE_COLUMN_GAP = 2.5
# Room for the film IDs between the two rows of example q-maps.
EXAMPLE_TITLE_ROOM = 5.0
# Room for the x-axis label below the lower row.
EXAMPLE_XLABEL_ROOM = 9.0
# Room for the q_z axis label between the UMAP and the first column of q-maps.
EXAMPLE_YLABEL_ROOM = 10.5
EXAMPLE_TITLE_GAP = 0.8

# Arial cap height and descent, as fractions of the font size.
ARIAL_CAP_HEIGHT, ARIAL_DESCENT = 0.716, 0.212
# Size of the beam-center cross in points: as tall as the capitals of the calibration text, on the image
# and in the text.
BEAM_CENTER_MARKER_SIZE = BODY_SIZE * ARIAL_CAP_HEIGHT
POINTS_PER_INCH = 72
# PNG resolution, and the resolution of the rasterized parts (images, scatter points) in the PDF and SVG.
PNG_DPI = 300
RASTER_DPI = 600
# Text stacks at the pipeline arrows: baseline-to-baseline distance as a multiple of the font size, and the
# gap between an arrow and the nearest text, in millimetres.
STEP_LINE_SPACING = 1.2
ARROW_TEXT_GAP = 1.2
# The calibration arrow points up at the reduction arrow and stops this far below it, in millimetres.
CALIBRATION_ARROW_LENGTH, CALIBRATION_ARROW_GAP = 3.6, 0.6
STEP_TITLE_STYLE = dict(fontsize=BODY_SIZE, fontweight="bold", color=INK)
STEP_NAME_STYLE = dict(fontsize=BODY_SIZE, family=MONO_FONT, color=INK)
STEP_DETAIL_STYLE = dict(fontsize=BODY_SIZE, color=MUTED)

BADGE_BOX = dict(boxstyle="circle,pad=0.28", fc=INK, ec="white", lw=0.6)


def draw_beam_center_marker(axes, x, y):
    """Dark cross with a white halo, readable on any part of the colour ramp."""
    axes.plot(x, y, marker="+", color="white", ms=BEAM_CENTER_MARKER_SIZE, mew=1.3, zorder=11)
    axes.plot(x, y, marker="+", color=INK, ms=BEAM_CENTER_MARKER_SIZE, mew=0.5, zorder=12)


def points_to_mm(points):
    return points / POINTS_PER_INCH * MM_PER_INCH


def short_film_id(film_id):
    """TF001028 -> TF1028, as in Figure 3."""
    return f"TF{int(film_id[2:])}"


def cmap_with_no_data_color():
    cmap = IMAGE_CMAP.copy()
    cmap.set_bad(NO_DATA_COLOR)
    return cmap


def positive_or_nan(values):
    """Zero and negative q-map values are outside the detector coverage."""
    return np.where(values > 0, values, np.nan)


class Canvas:
    """Figure with placement in millimetres from the bottom-left corner.

    Boxes, arrows and loose text go on one transparent overlay axes whose data
    units are millimetres, so shapes keep their proportions.
    """

    def __init__(self, width, height):
        self.width, self.height = width, height
        # Created at the PNG resolution, so text measured for inline markers matches the saved files.
        self.figure = plt.figure(figsize=(width / MM_PER_INCH, height / MM_PER_INCH), dpi=PNG_DPI)
        self.overlay = self.figure.add_axes([0, 0, 1, 1], zorder=10)
        self.overlay.set_xlim(0, width)
        self.overlay.set_ylim(0, height)
        self.overlay.axis("off")

    def axes(self, left, bottom, width, height, **kwargs):
        return self.figure.add_axes(
            [left / self.width, bottom / self.height, width / self.width, height / self.height], **kwargs)

    def text(self, x, y, text, **kwargs):
        return self.overlay.text(x, y, text, **kwargs)

    def arrow(self, x_start, y_start, x_end, y_end):
        self.overlay.annotate("", xy=(x_end, y_end), xytext=(x_start, y_start),
                              arrowprops=dict(arrowstyle="-|>", mutation_scale=7, lw=0.8, color=MUTED,
                                              shrinkA=0, shrinkB=0))

    def text_width(self, text_artist):
        """Width of a text artist, in millimetres."""
        extent = text_artist.get_window_extent(self.figure.canvas.get_renderer())
        (x0, _), (x1, _) = self.overlay.transData.inverted().transform(extent.get_points())
        return x1 - x0

    def centered_text_with_marker(self, x_center, baseline, before, after, draw_marker, marker_size_pt,
                                  marker_height_above_baseline, **text_kwargs):
        """One centred line of text with a marker between `before` and `after`."""
        marker_width = points_to_mm(marker_size_pt)
        gap = 0.5
        before_text = self.text(0, baseline, before, ha="left", va="baseline", **text_kwargs)
        after_text = self.text(0, baseline, after, ha="left", va="baseline", **text_kwargs)
        before_width, after_width = self.text_width(before_text), self.text_width(after_text)
        left = x_center - (before_width + gap + marker_width + after_width) / 2
        before_text.set_x(left)
        marker_x = left + before_width + gap + marker_width / 2
        after_text.set_x(marker_x + marker_width / 2)
        draw_marker(self.overlay, marker_x, baseline + marker_height_above_baseline)

    def badge(self, x, y, number):
        """Numbered disc centred at (x, y); the same badge marks the film on the UMAP."""
        self.overlay.text(x, y, str(number), ha="center", va="center", fontsize=BODY_SIZE, fontweight="bold",
                          color="white", bbox=BADGE_BOX, zorder=12)

    def image_badge(self, left, bottom, height, number):
        """Badge in the top-left corner of an image."""
        self.badge(left + BADGE_INSET, bottom + height - BADGE_INSET, number)

    def colorbar(self, image, image_left, image_width, bottom, height, label):
        """Log colorbar with decade labels, to the right of an image."""
        colorbar_axes = self.axes(image_left + image_width + COLORBAR_GAP, bottom, COLORBAR_WIDTH, height)
        colorbar = self.figure.colorbar(image, cax=colorbar_axes)
        colorbar.ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
        colorbar.ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10))
        colorbar.ax.yaxis.set_minor_formatter(NullFormatter())
        colorbar.set_label(label, labelpad=2)
        colorbar.outline.set_linewidth(0.5)
        colorbar.ax.tick_params(width=0.5, length=2, pad=1.5)


def right_of_colorbar(image_left, image_width):
    """First free x position after an image with its colorbar, ticks and label."""
    return image_left + image_width + COLORBAR_GAP + COLORBAR_WIDTH + COLORBAR_LABEL_ROOM


# --- Data --------------------------------------------------------------------------------------------------------

def load_data():
    umap_films = pd.read_csv(UMAP_FILE)
    labels = pd.read_csv(LABELS_FILE)
    return {
        "pipeline": np.load(PIPELINE_FILE),
        "example_qmaps": np.load(EXAMPLE_QMAPS_FILE),
        "class_counts": labels["giwaxs_class"].value_counts(),
        "umap_films": umap_films,
        "labelled_umap_films": umap_films.merge(labels, on="ThinFilmID"),
    }


# --- Panel A -----------------------------------------------------------------------------------------------------

def draw_raw_image(canvas, pipeline, left, bottom, height):
    """Raw detector image as recorded, with the beam center marked; returns its width.

    The detector mask is not drawn. Module-gap pixels are negative in the raw frame and are shown in the
    no-data colour.
    """
    counts = np.ma.masked_less_equal(pipeline["raw"].astype(float), 0)
    n_rows, n_columns = counts.shape
    width = height * n_columns / n_rows
    pixel_extent = [0, n_columns, n_rows, 0]
    axes = canvas.axes(left, bottom, width, height)
    image = axes.imshow(counts, cmap=cmap_with_no_data_color(), norm=LogNorm(*RAW_COUNT_LIMITS), origin="upper",
                        interpolation="antialiased", extent=pixel_extent)

    beam_x, beam_y = pipeline["beam_center_px"]
    draw_beam_center_marker(axes, beam_x, beam_y)

    axes.set_xticks(RAW_X_TICKS)
    axes.set_yticks(RAW_Y_TICKS)
    axes.set_xlabel("Pixel x")
    axes.set_ylabel("Pixel y")
    canvas.colorbar(image, left, width, bottom, height, "Counts per pixel")
    return width


def draw_qmap(canvas, pipeline, left, bottom, height):
    """Reduced q-space map at its true aspect ratio; returns its width."""
    q_xy, q_z = pipeline["q_xy"], pipeline["q_z"]
    q_extent = [q_xy.min(), q_xy.max(), q_z.min(), q_z.max()]
    width = height * (q_xy.max() - q_xy.min()) / (q_z.max() - q_z.min())
    axes = canvas.axes(left, bottom, width, height)
    image = axes.imshow(positive_or_nan(pipeline["qmap"]), cmap=cmap_with_no_data_color(),
                        norm=LogNorm(*QMAP_INTENSITY_LIMITS), origin="lower", interpolation="nearest",
                        extent=q_extent, aspect="equal")
    axes.set_xlabel(QXY_LABEL)
    axes.set_ylabel(QZ_LABEL)
    axes.set_xticks(QXY_TICKS)
    axes.set_yticks(QZ_TICKS)
    canvas.colorbar(image, left, width, bottom, height, "Counts / diode (a.u.)")
    return width


def draw_text_stack(canvas, x_center, first_baseline, lines):
    """Centred lines at a fixed line spacing. A line is (text, style), or (before, style, after) with the
    beam-center marker between `before` and `after`."""
    pitch = points_to_mm(BODY_SIZE * STEP_LINE_SPACING)
    for index, line in enumerate(lines):
        baseline = first_baseline - index * pitch
        if len(line) == 3:
            before, style, after = line
            canvas.centered_text_with_marker(x_center, baseline, before, after, draw_beam_center_marker,
                                             BEAM_CENTER_MARKER_SIZE,
                                             points_to_mm(BODY_SIZE * ARIAL_CAP_HEIGHT) / 2, **style)
        else:
            text, style = line
            canvas.text(x_center, baseline, text, ha="center", va="baseline", **style)


def stack_above(canvas, x_center, y, lines):
    """Text stack whose lowest descender sits ARROW_TEXT_GAP above y."""
    pitch = points_to_mm(BODY_SIZE * STEP_LINE_SPACING)
    lowest_baseline = y + ARROW_TEXT_GAP + points_to_mm(BODY_SIZE * ARIAL_DESCENT)
    draw_text_stack(canvas, x_center, lowest_baseline + (len(lines) - 1) * pitch, lines)


def stack_below(canvas, x_center, y, lines):
    """Text stack whose first capitals start ARROW_TEXT_GAP below y."""
    draw_text_stack(canvas, x_center, y - ARROW_TEXT_GAP - points_to_mm(BODY_SIZE * ARIAL_CAP_HEIGHT), lines)


def draw_reduction_step(canvas, x_start, x_end, y):
    """Reduction arrow with the reduction software above and the calibration feeding it from below."""
    canvas.arrow(x_start, y, x_end, y)
    x_center = (x_start + x_end) / 2
    stack_above(canvas, x_center, y, [("Reduction", STEP_TITLE_STYLE), ("pyGID", STEP_NAME_STYLE)])
    calibration_arrow_top = y - CALIBRATION_ARROW_GAP
    calibration_arrow_bottom = calibration_arrow_top - CALIBRATION_ARROW_LENGTH
    canvas.arrow(x_center, calibration_arrow_bottom, x_center, calibration_arrow_top)
    stack_below(canvas, x_center, calibration_arrow_bottom, [
        ("Calibration", STEP_TITLE_STYLE),
        ("GIWAXS_Batch", STEP_NAME_STYLE),
        ("wavelength, distance,", STEP_DETAIL_STYLE),
        ("beam center", STEP_DETAIL_STYLE, ", mask"),
    ])


def draw_labelling_step(canvas, x_start, x_end, y):
    """Labelling arrow with the MLExchange logo above it and the labeling step below it."""
    canvas.arrow(x_start, y, x_end, y)
    x_center = (x_start + x_end) / 2
    logo = plt.imread(LOGO_FILE)
    logo_height = LOGO_WIDTH * logo.shape[0] / logo.shape[1]
    logo_axes = canvas.axes(x_center - LOGO_WIDTH / 2, y + ARROW_TEXT_GAP, LOGO_WIDTH, logo_height)
    logo_axes.imshow(logo)
    logo_axes.axis("off")
    stack_below(canvas, x_center, y, [("Labeling", STEP_TITLE_STYLE), ("mlex_labelmaker", STEP_NAME_STYLE)])


def draw_class_tracks(canvas, class_counts, pipeline_class, left, bottom):
    """Two full-length tracks, each filled by its class share of the labelled films.

    Crystalline fills from the left and amorphous from the right, so the two fills meet at the same x.
    """
    width = RIGHT_EDGE - left
    n_labelled = int(class_counts.sum())
    style = f"round,pad=0,rounding_size={TRACK_CORNER_RADIUS}"
    for class_name, track_offset in TRACK_BOTTOMS.items():
        color = CLASS_COLORS[class_name]
        track_bottom = bottom + track_offset
        share = class_counts[class_name] / n_labelled
        track = FancyBboxPatch((left, track_bottom), width, TRACK_HEIGHT, boxstyle=style, fc="white", ec="none",
                               zorder=1)
        canvas.overlay.add_patch(track)
        fill_left = left if TRACK_FILL_FROM_LEFT[class_name] else left + width * (1 - share)
        fill = Rectangle((fill_left, track_bottom), width * share, TRACK_HEIGHT, fc=color, ec="none", zorder=2)
        canvas.overlay.add_patch(fill)
        fill.set_clip_path(track)
        canvas.overlay.add_patch(FancyBboxPatch((left, track_bottom), width, TRACK_HEIGHT, boxstyle=style,
                                                fc="none", ec=color, lw=0.9, zorder=3))
        label_y = track_bottom + TRACK_HEIGHT + 0.9
        canvas.text(left, label_y, class_name, ha="left", va="bottom", fontsize=BODY_SIZE, fontweight="bold",
                    color=color)
        canvas.text(left + width, label_y, f"{class_counts[class_name]:,} ({share:.1%})", ha="right",
                    va="bottom", fontsize=BODY_SIZE, color=INK)
        if class_name == pipeline_class:
            canvas.badge(left + 3.3, track_bottom + TRACK_HEIGHT / 2, PIPELINE_BADGE)
    canvas.text(left + width / 2, bottom + 1.4, f"of {n_labelled:,} labeled films", ha="center", va="top",
                fontsize=BODY_SIZE, color=MUTED)


def draw_heading(canvas, left, image_top, text):
    canvas.text(left, image_top + HEADING_GAP, text, fontsize=HEADING_SIZE, fontweight="bold", va="baseline")


def draw_pipeline(canvas, data):
    """Panel A: raw image -> pyGID reduction -> q-space map -> MLExchange labelling -> class shares."""
    pipeline = data["pipeline"]
    bottom, height = PIPELINE_BOTTOM, PIPELINE_IMAGE_HEIGHT
    top = bottom + height
    arrow_y = bottom + height * ARROW_HEIGHT_FRACTION

    raw_left = LEFT_MARGIN
    raw_width = draw_raw_image(canvas, pipeline, raw_left, bottom, height)
    draw_heading(canvas, raw_left, top, "Raw detector image")
    canvas.image_badge(raw_left, bottom, height, PIPELINE_BADGE)
    canvas.text(raw_left, top + FILM_LABEL_GAP, short_film_id(str(pipeline["thin_film_id"])), fontsize=BODY_SIZE,
                fontweight="bold", color=INK, va="bottom")
    canvas.text(raw_left + SCAN_INFO_OFFSET, top + FILM_LABEL_GAP,
                f"{PIPELINE_INCIDENT_ANGLE_DEG:.1f}°, {PIPELINE_EXPOSURE_S} s", fontsize=BODY_SIZE, color=MUTED,
                va="bottom")

    reduction_start = right_of_colorbar(raw_left, raw_width)
    reduction_end = reduction_start + REDUCTION_ARROW_LENGTH
    draw_reduction_step(canvas, reduction_start, reduction_end, arrow_y)

    qmap_left = reduction_end + QMAP_AXIS_LABEL_ROOM
    qmap_width = draw_qmap(canvas, pipeline, qmap_left, bottom, height)
    draw_heading(canvas, qmap_left, top, "q-space map")
    canvas.image_badge(qmap_left, bottom, height, PIPELINE_BADGE)

    labelling_start = right_of_colorbar(qmap_left, qmap_width)
    labelling_end = labelling_start + LABELLING_ARROW_LENGTH
    draw_labelling_step(canvas, labelling_start, labelling_end, arrow_y)

    tracks_left = labelling_end + TRACK_GAP
    draw_heading(canvas, tracks_left, top, "Label")
    draw_class_tracks(canvas, data["class_counts"], str(pipeline["giwaxs_class"]), tracks_left, bottom)


# --- Panel B -----------------------------------------------------------------------------------------------------

def draw_example_qmap(canvas, data, film, color, left, bottom, show_xlabel, show_ylabel):
    """Small q-map of an example film, on the colour scale of panel A, framed in its class colour."""
    film_id = film.film_id
    example_qmaps = data["example_qmaps"]
    axes = canvas.axes(left, bottom, EXAMPLE_WIDTH, EXAMPLE_HEIGHT)
    q_xy, q_z = example_qmaps[f"{film_id}_q_xy"], example_qmaps[f"{film_id}_q_z"]
    axes.imshow(positive_or_nan(example_qmaps[f"{film_id}_qmap"]), cmap=cmap_with_no_data_color(),
                norm=LogNorm(*QMAP_INTENSITY_LIMITS), origin="lower", interpolation="antialiased", aspect="auto",
                extent=[q_xy.min(), q_xy.max(), q_z.min(), q_z.max()])
    axes.set_xlim(*EXAMPLE_QXY_LIMITS)
    axes.set_ylim(*EXAMPLE_QZ_LIMITS)
    axes.set_xticks(QXY_TICKS)
    axes.set_yticks(QZ_TICKS)
    if show_ylabel:
        axes.set_ylabel(QZ_LABEL)
    else:
        axes.set_yticklabels([])
    if show_xlabel:
        axes.set_xlabel(QXY_LABEL)
    else:
        axes.set_xticklabels([])
    for spine in axes.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(1.3)
    canvas.image_badge(left, bottom, EXAMPLE_HEIGHT, film.badge)
    canvas.text(left, bottom + EXAMPLE_HEIGHT + EXAMPLE_TITLE_GAP, short_film_id(film_id), fontsize=BODY_SIZE,
                fontweight="bold", color=INK, va="bottom")


def draw_class_umap(axes, umap_films, class_films, color, marked_films):
    """All films in grey, the class in its colour with density contours, and the marked films with badges."""
    axes.scatter(umap_films[UMAP_X_COLUMN], umap_films[UMAP_Y_COLUMN], s=0.4, c=CONTEXT_GREY, lw=0,
                 rasterized=True, zorder=1)
    axes.scatter(class_films[UMAP_X_COLUMN], class_films[UMAP_Y_COLUMN], s=1.4, c=color, lw=0, alpha=0.8,
                 rasterized=True, zorder=2)

    x_edges = np.linspace(*UMAP_XLIM, UMAP_DENSITY_BINS)
    y_edges = np.linspace(*UMAP_YLIM, UMAP_DENSITY_BINS)
    density, _, _ = np.histogram2d(class_films[UMAP_X_COLUMN], class_films[UMAP_Y_COLUMN], bins=[x_edges, y_edges])
    density = gaussian_filter(density, UMAP_DENSITY_SMOOTHING).T
    axes.contour((x_edges[:-1] + x_edges[1:]) / 2, (y_edges[:-1] + y_edges[1:]) / 2, density,
                 levels=density.max() * np.array(UMAP_CONTOUR_FRACTIONS), colors=INK, linewidths=0.35, alpha=0.6,
                 zorder=4)

    # Marked as in Figure 3 (filled dot, white edge, thin dark ring), with the numbered badge joined to the dot.
    positions = umap_films.set_index("ThinFilmID")
    for film in marked_films:
        x, y = positions.loc[film.film_id, UMAP_X_COLUMN], positions.loc[film.film_id, UMAP_Y_COLUMN]
        axes.scatter(x, y, s=8, facecolors=INK, edgecolors="white", linewidths=0.4, zorder=10)
        axes.scatter(x, y, s=10, facecolors="none", edgecolors=INK, linewidths=0.3, zorder=11)
        offset_x, offset_y = film.badge_offset
        axes.annotate(str(film.badge), xy=(x, y), xytext=(x + offset_x, y + offset_y), ha="center", va="center",
                      fontsize=BODY_SIZE, fontweight="bold", color="white", zorder=13, bbox=BADGE_BOX,
                      arrowprops=dict(arrowstyle="-", lw=0.5, color=INK, shrinkA=0, shrinkB=1.5))
    axes.set_xlim(*UMAP_XLIM)
    axes.set_ylim(*UMAP_YLIM)
    axes.set_xticks([])
    axes.set_yticks([])


def draw_class_block(canvas, data, class_name, left, extra_marked_films=()):
    """UMAP of one class with its example films marked, and the examples' q-maps in a 2-row grid.

    Returns the x position of the block's right edge.
    """
    color = CLASS_COLORS[class_name]
    examples = EXAMPLE_FILMS[class_name]
    lower_row_bottom = CLASS_BLOCK_BOTTOM + EXAMPLE_XLABEL_ROOM
    upper_row_bottom = lower_row_bottom + EXAMPLE_HEIGHT + EXAMPLE_TITLE_ROOM
    umap_height = upper_row_bottom + EXAMPLE_HEIGHT - lower_row_bottom

    umap_axes = canvas.axes(left, lower_row_bottom, UMAP_WIDTH, umap_height)
    labelled = data["labelled_umap_films"]
    draw_class_umap(umap_axes, data["umap_films"], labelled[labelled["giwaxs_class"] == class_name], color,
                    list(examples) + list(extra_marked_films))
    umap_axes.set_xlabel("UMAP 1")
    umap_axes.set_ylabel("UMAP 2")
    umap_axes.set_title(class_name, loc="left", pad=3, color=color, fontweight="bold", fontsize=HEADING_SIZE)

    examples_left = left + UMAP_WIDTH + EXAMPLE_YLABEL_ROOM
    row_bottoms = [upper_row_bottom, lower_row_bottom]
    right_edge = examples_left
    for index, film in enumerate(examples):
        column, row = divmod(index, len(row_bottoms))
        example_left = examples_left + column * (EXAMPLE_WIDTH + EXAMPLE_COLUMN_GAP)
        draw_example_qmap(canvas, data, film, color, example_left, row_bottoms[row],
                          show_xlabel=(row == len(row_bottoms) - 1), show_ylabel=(column == 0))
        right_edge = max(right_edge, example_left + EXAMPLE_WIDTH)
    return right_edge


def draw_class_blocks(canvas, data):
    """Panel B: one block per class."""
    n_labelled = int(data["class_counts"].sum())
    n_in_umap = len(data["labelled_umap_films"])
    canvas.text(LEFT_MARGIN, PANEL_B_HEADING_Y,
                f"GIWAXS labels on the UV-vis/image UMAP ({n_in_umap:,} of the {n_labelled:,} labeled films)",
                fontsize=HEADING_SIZE, fontweight="bold", va="baseline")
    pipeline = data["pipeline"]
    pipeline_film = ExampleFilm(str(pipeline["thin_film_id"]), PIPELINE_BADGE, PIPELINE_BADGE_OFFSET)
    pipeline_class = str(pipeline["giwaxs_class"])

    left = LEFT_MARGIN
    for class_name in ("Amorphous", "Crystalline"):
        extra = [pipeline_film] if class_name == pipeline_class else []
        right_edge = draw_class_block(canvas, data, class_name, left, extra_marked_films=extra)
        left = right_edge + CLASS_BLOCK_GAP


def make_figure(data):
    canvas = Canvas(FIGURE_WIDTH, FIGURE_HEIGHT)
    draw_pipeline(canvas, data)
    draw_class_blocks(canvas, data)
    # Panel letters share the baseline of the first heading of their panel.
    panel_a_heading_y = PIPELINE_BOTTOM + PIPELINE_IMAGE_HEIGHT + HEADING_GAP
    for letter, heading_y in (("A", panel_a_heading_y), ("B", PANEL_B_HEADING_Y)):
        canvas.text(PANEL_LETTER_X, heading_y, letter, fontsize=PANEL_LETTER_SIZE, fontweight="bold", va="baseline")
    return canvas.figure


def main():
    matplotlib.rcParams.update(STYLE)
    font_manager.findfont("Arial", fallback_to_default=False)
    font_manager.findfont(MONO_FONT, fallback_to_default=False)
    figure = make_figure(load_data())
    OUTPUT_DIR.mkdir(exist_ok=True)
    # No timestamps in the SVG and PDF, so repeated runs give identical files.
    figure.savefig(OUTPUT_DIR / "figure_04.svg", dpi=RASTER_DPI, metadata={"Date": None})
    figure.savefig(OUTPUT_DIR / "figure_04.pdf", dpi=RASTER_DPI, metadata={"CreationDate": None})
    figure.savefig(OUTPUT_DIR / "figure_04.png", dpi=PNG_DPI)
    plt.close(figure)
    print("wrote", OUTPUT_DIR / "figure_04")


if __name__ == "__main__":
    main()
