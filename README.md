# 10k-workflow

Data and reproduction code for selected figures from:

> Kodalle, et al. "Cross-Facility Materials Discovery Platform: Integrating
> Heterogeneous Data and Automated Experimental Pipelines into
> High-Throughput Workflows."

This repository accompanies the paper's Data Availability statement.

## Repository layout

```
data/       input data for each figure's reproduction script
scripts/    one self-contained script per figure; each writes to figures/
figures/    generated output (svg/pdf/png), not checked in
```

Each script hardcodes its paths relative to its own location (`data/` and
`figures/` one level up), so they can be run directly from `scripts/` with no
arguments.

## Requirements

`plot_processing_timeline.py` (Figure 2C) uses only the Python standard
library. `plot_figure_03.py` (Figure 3) needs the packages in
`requirements.txt` (`pip install -r requirements.txt`), and Arial installed
system-wide — it raises an error rather than silently substituting a
different font, since that would change the figure's metrics.

## Figure 2C — processing timeline

```
scripts/plot_processing_timeline.py
```

Reproduces the calendar-day lags for synthesis steps (solid precursor
weighed, stock solution made, precursor solution mixed) alongside sub-day
lags for characterization steps (line-scan, imaging, RGA, GIWAXS), all
relative to each film's spin-coat time. Synthesis-step lags are computed as
whole calendar days (the source bench records are date-only) in the lab's
local time zone (US Pacific); a same-day lag is displayed at an assumed 12 h
offset for plotting purposes, since the true sub-day time isn't resolvable
from the source data.

**Data**: `data/event_timestamps.csv` — one row per
`(film, event_type, entity_id, timestamp)` for all 6,105 films. See
`data/README.md` for the full column reference.

## Figure 3 — image and UV-Vis spatial homogeneity, and the 10-spot UMAP

```
scripts/plot_figure_03.py
```

Three panels, built from the two exemplar films TF001028 (homogeneous) and
TF003020 (inhomogeneous):

- **A** — well photo, central-40% ROI crop, pixel-wise CIEDE2000 map against
  the ROI median color, and the library-wide histogram of per-film image
  ΔE₀₀ median (5,935 QC-valid films).
- **B** — 10-spot UV-Vis: raw intensity and computed absorbance spectra,
  pairwise mean |ΔA| matrix across the 10 scan positions, and the
  library-wide histogram of UV-Vis spectral inhomogeneity (5,989 films).
- **C** — UMAP (9 HDBSCAN clusters) of the 5,730 films with complete data for
  both modalities, plus the same UMAP space colored by image ΔE₀₀, UV-Vis
  inhomogeneity, optical contrast, and spectral centroid.

**Data**:
- `data/exemplar_image_de00_maps_TF001028_TF003020.npz` — well photos, ROI
  crops, ROI bounds and CIEDE2000 maps for the two exemplar films.
- `data/exemplar_uvvis_spectra_TF001028_TF003020.npz` — wavelength grid,
  smoothed intensity and absorbance spectra, and the 10×10 |ΔA| matrix for
  the two exemplar films.
- `data/image_inhomogeneity_by_film.csv` — `ThinFilmID`, `delta_e_2000_median`,
  `qc_valid`, for every imaged film.
- `data/uvvis_inhomogeneity_by_film.csv` — `ThinFilmID`,
  `uvvis_robust_spectral_inhomogeneity`, for every film with a UV-Vis
  measurement.
- `data/umap_clusters_by_film.csv` — `ThinFilmID`, UMAP coordinates
  (`umap_10spot_eq77_1/2`), cluster assignment (`cluster_k9_raw`,
  `cluster_k9_assigned`), and the four physical descriptors plotted in panel
  C. The underlying 77 engineered features used to compute the embedding are
  not included (not needed to reproduce the figure from these coordinates,
  and outside this repository's scope — see Scope above).

The UMAP embedding and HDBSCAN clustering are not recomputed by this script;
the coordinates and cluster labels in `umap_clusters_by_film.csv` are the
ones used in the published figure.

## Figure 4

Data and reproduction code to be added.

## License

BSD 2-Clause, see `LICENSE`.
