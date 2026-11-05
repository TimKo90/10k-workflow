#!/usr/bin/env python3
"""Processing-timeline figure: calendar-day lags for synthesis steps (solid
precursor weighed, stock solution made, precursor solution mixed) alongside
sub-day lags for characterization steps (line-scan, imaging, RGA, GIWAXS),
all measured relative to each film's spin-coat time.

Input: data/event_timestamps.csv (see data/README.md for the schema).

Synthesis step timestamps are date-only in the source records (no
time-of-day), so lags for SP/SS/PS are computed as whole calendar-day
differences rather than exact durations. A lag of 0 ("same calendar day as
spin-coating") could mean anywhere from minutes to ~24 h; for display
purposes these are plotted at an assumed 12 h before spin-coating, marked
with a "*" on the axis. Calendar days are taken in the lab's local time zone
(US Pacific), which is the offset carried by the characterization-step
timestamps in the released CSV. Lags for the characterization steps use the
full-precision timestamps directly.
"""
import csv
import datetime as dt
import math
import os

DAY = 86400.0

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "..", "data", "event_timestamps.csv")
OUT_SVG = os.path.join(HERE, "..", "figures", "figure_2c.svg")

LEFT_STAGES = [("sp", "Solid precursor weighed", "#25737e"),
               ("ss", "Stock solution made", "#2b8265"),
               ("ps", "Precursor solution mixed", "#328539")]
RIGHT_STAGES = [("line", "In-situ line-scan", "#94ba5e"),
                ("img", "Well imaging", "#89983e"),
                ("rga", "RGA / TEY", "#b5a9cf"),
                ("giwaxs", "GIWAXS (beamtime)", "#634999")]

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
SPIN_COLOR = "#008300"


def parse_timestamp(s):
    """Characterization-step timestamps carry a UTC offset
    ('YYYY-MM-DDTHH:MM:SS-07:00'); synthesis timestamps are date-only
    ('YYYY-MM-DD')."""
    if "T" in s:
        return dt.datetime.fromisoformat(s)
    return dt.datetime.strptime(s, "%Y-%m-%d").date()


def fmt_days(d):
    if d < 1 / 24:
        return f"{d * 1440:.0f} min"
    if d < 1:
        return f"{d * 24:.0f} h"
    if d < 14:
        return f"{d:.1f} d"
    return f"{d:.0f} d"


# ---------------------------------------------------------------- load + aggregate
per_film = {}
with open(CSV_PATH, newline="") as fh:
    for row in csv.DictReader(fh):
        per_film.setdefault(row["film"], []).append(row)

syn_lags = {code: [] for code, _, _ in LEFT_STAGES}
char_lags = {code: [] for code, _, _ in RIGHT_STAGES}

for film, rows in per_film.items():
    spin_ts = [parse_timestamp(r["timestamp"]) for r in rows if r["event_type"] == "spin"]
    if not spin_ts:
        continue
    spin_dt = min(spin_ts)
    spin_date = spin_dt.date()

    for code in syn_lags:
        dates = [parse_timestamp(r["timestamp"]) for r in rows if r["event_type"] == code]
        if not dates:
            continue
        days = (spin_date - max(dates)).days
        if days < 0:
            continue
        syn_lags[code].append(days)

    for code in char_lags:
        times = [parse_timestamp(r["timestamp"]) for r in rows if r["event_type"] == code]
        if not times:
            continue
        lag = (min(times) - spin_dt).total_seconds() / DAY
        if lag > -1:
            char_lags[code].append(lag)

n_films = len(per_film)

# ---------------------------------------------------------------- minimal SVG writer
class SVG:
    def __init__(self, width, height, bg=SURFACE):
        self.width, self.height = width, height
        self.parts = [f'<rect x="0" y="0" width="{width}" height="{height}" fill="{bg}"/>'] if bg else []

    def rect(self, x, y, w, h, fill="none", opacity=None, rx=0):
        op = f' opacity="{opacity}"' if opacity is not None else ""
        self.parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{rx}" fill="{fill}"{op}/>')

    def line(self, x1, y1, x2, y2, stroke=BASELINE, stroke_width=1, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" stroke-width="{stroke_width}"{d}/>')

    def path(self, d, fill="none", opacity=None):
        op = f' opacity="{opacity}"' if opacity is not None else ""
        self.parts.append(f'<path d="{d}" fill="{fill}" stroke="none"{op}/>')

    def polyline(self, pts, stroke, stroke_width=2):
        p = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        self.parts.append(f'<polyline points="{p}" fill="none" stroke="{stroke}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"/>')

    def text(self, x, y, s, size=12, fill=INK_PRIMARY, weight=400, anchor="start", tabular=False):
        tn = ' font-variant-numeric="tabular-nums"' if tabular else ""
        s = str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.parts.append(f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, Helvetica, sans-serif" '
                           f'font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{tn}>{s}</text>')

    def save(self, path):
        body = "\n  ".join(self.parts)
        with open(path, "w") as f:
            f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
                     f'viewBox="0 0 {self.width} {self.height}">\n  {body}\n</svg>')


# ---------------------------------------------------------------- layout
W = 1750
TOP = 70
TL_H = 560
H = TOP + TL_H + 40
TL_GAPC = 56
TXC = W / 2
TL_MR = 110
TPW = min(TXC - 210 - TL_GAPC / 2, W - TL_MR - TXC - TL_GAPC / 2)
TL_ML = TXC - TL_GAPC / 2 - TPW
TPH = TL_H - 90

DLOG_MIN, DLOG_MAX = math.log10(0.5), math.log10(400)
LOG_MIN, LOG_MAX = math.log10(1 / 24 / 6), math.log10(400)


def txr(days):
    return TXC + TL_GAPC / 2 + (math.log10(max(days, 10 ** LOG_MIN)) - LOG_MIN) / (LOG_MAX - LOG_MIN) * TPW


def txl(days):
    return TXC - TL_GAPC / 2 - (math.log10(max(days, 10 ** DLOG_MIN)) - DLOG_MIN) / (DLOG_MAX - DLOG_MIN) * TPW


svg = SVG(W, H)
svg.text(60, 28, f"Processing timeline for the same batch, population-wide ({n_films:,} films)", size=17, weight=700, fill=INK_PRIMARY)
svg.text(60, 48, "Left: calendar-day lags for synthesis steps (date-only timestamps). Right: sub-day lags for characterization steps.",
         size=11.5, fill=INK_SECONDARY)

tl_rows = len(LEFT_STAGES) + len(RIGHT_STAGES)
tl_lane_h = TPH / tl_rows
MT2 = TOP + 76

for d, lab in [(1 / 24, "1h"), (1, "1d"), (7, "1wk"), (30, "1mo"), (100, "100d"), (365, "1yr")]:
    svg.line(txr(d), MT2, txr(d), MT2 + TPH, stroke=GRID)
    svg.text(txr(d), MT2 + TPH + 18, lab, size=10.5, fill=INK_MUTED, anchor="middle")
for d, lab in [(0.5, "12h*"), (1, "1d"), (2, "2d"), (7, "1wk"), (30, "1mo"), (100, "100d"), (365, "1yr")]:
    svg.line(txl(d), MT2, txl(d), MT2 + TPH, stroke=GRID)
    svg.text(txl(d), MT2 + TPH + 18, lab, size=10.5, fill=INK_MUTED, anchor="middle")
svg.text(TXC - TL_GAPC / 2 - TPW / 2, MT2 + TPH + 48, "calendar days before spin-coating", size=11.5, fill=INK_SECONDARY, anchor="middle")
svg.text(TXC + TL_GAPC / 2 + TPW / 2, MT2 + TPH + 48, "time after spin-coating", size=11.5, fill=INK_SECONDARY, anchor="middle")

svg.rect(TXC - TL_GAPC / 2 + 5, MT2 - 8, TL_GAPC - 10, TPH + 16, fill=SPIN_COLOR, opacity=0.08, rx=5)
svg.line(TXC, MT2 - 8, TXC, MT2 + TPH + 8, stroke=SPIN_COLOR, stroke_width=1.8)
svg.text(TXC, MT2 - 16, "spin-coat", size=12.5, weight=700, anchor="middle")


def row_label(i, label, n):
    base = MT2 + (i + 1) * tl_lane_h - 9
    svg.text(TL_ML - 12, base - tl_lane_h * 0.35, label, size=12, weight=700, anchor="end")
    svg.text(TL_ML - 12, base - tl_lane_h * 0.35 + 15, f"{n:,} films", size=10, fill=INK_MUTED, anchor="end", tabular=True)
    return base


def left_ridge(i, days, label, col):
    """Each integer day maps to exactly one x position (no jitter or
    interpolation, since the source data has no sub-day resolution). The
    plotted height is the raw count per day, not a smoothed curve. Films
    with a same-day (0-day) lag are plotted at an assumed 12 h position so
    they still appear on the curve, and their fraction is also called out
    as a text label."""
    base = row_label(i, label, len(days))
    n = len(days)
    hmax = tl_lane_h * 0.74
    counts = {}
    n_same = 0
    for d in days:
        if d == 0:
            n_same += 1
        else:
            counts[d] = counts.get(d, 0) + 1
    if n_same:
        counts[0.5] = n_same
    peak = max(counts.values(), default=0) or 1
    x0, x1 = TXC - TL_GAPC / 2, TXC - TL_GAPC / 2 - TPW
    pts = [(txl(d), base - min(v / peak, 1.0) * hmax) for d, v in sorted(counts.items())]
    fill_pts = [(x0, base)] + pts + [(x1, base)]
    d_path = f"M {fill_pts[0][0]:.1f},{fill_pts[0][1]:.1f} L " + " L ".join(f"{a:.1f},{b:.1f}" for a, b in fill_pts[1:])
    svg.path(d_path, fill=col, opacity=0.18)
    svg.polyline(pts, stroke=col, stroke_width=1.6)
    svg.line(x1, base, x0, base, stroke=BASELINE)
    med = sorted(days)[n // 2]
    mx = txl(max(med, 1))
    med_lab = "same day" if med == 0 else f"{med} d"
    svg.line(mx, base, mx, base - hmax - 3, stroke=INK_PRIMARY, stroke_width=0.9, dash="3,3")
    svg.text(mx - 6, base - hmax + 4, f"median {med_lab}", size=10.5, weight=700, anchor="end")
    svg.text(x0 + 4, base - 6, f"{100 * n_same / n:.0f}% same day (plotted at assumed 12h)", size=9.5, fill=INK_SECONDARY, anchor="start")


def right_ridge(i, lags, label, col):
    base = row_label(i, label, len(lags))
    lags = [max(l, 10 ** LOG_MIN) for l in lags]
    nb = 90
    hist = [0] * nb
    for l in lags:
        b = int((math.log10(l) - LOG_MIN) / (LOG_MAX - LOG_MIN) * nb)
        hist[min(max(b, 0), nb - 1)] += 1
    sm = [(hist[max(j - 1, 0)] + 2 * hist[j] + hist[min(j + 1, nb - 1)]) / 4 for j in range(nb)]
    peak = max(sm) or 1
    x0, x1 = TXC + TL_GAPC / 2, TXC + TL_GAPC / 2 + TPW
    pts = [(x0 + (x1 - x0) * (j + 0.5) / nb, base - v / peak * tl_lane_h * 0.74) for j, v in enumerate(sm)]
    d = f"M {x0:.1f},{base:.1f} L " + " L ".join(f"{a:.1f},{b:.1f}" for a, b in pts) + f" L {x1:.1f},{base:.1f} Z"
    svg.path(d, fill=col, opacity=0.18)
    svg.polyline(pts, stroke=col, stroke_width=1.6)
    svg.line(x0, base, x1, base, stroke=BASELINE)
    med = sorted(lags)[len(lags) // 2]
    svg.line(txr(med), base, txr(med), base - tl_lane_h * 0.74 - 3, stroke=INK_PRIMARY, stroke_width=0.9, dash="3,3")
    svg.text(txr(med) + 6, base - tl_lane_h * 0.74 + 4, f"median {fmt_days(med)}", size=10.5, weight=700)


for i, (code, lab, col) in enumerate(LEFT_STAGES):
    left_ridge(i, syn_lags[code], lab, col)
for k, (code, lab, col) in enumerate(RIGHT_STAGES):
    right_ridge(len(LEFT_STAGES) + k, char_lags[code], lab, col)

tysep = MT2 + len(LEFT_STAGES) * tl_lane_h
svg.line(TL_ML, tysep, W - TL_MR, tysep, stroke=BASELINE, stroke_width=1, dash="4,4")

os.makedirs(os.path.dirname(OUT_SVG), exist_ok=True)
svg.save(OUT_SVG)
print(f"wrote {OUT_SVG} ({n_films:,} films)")
