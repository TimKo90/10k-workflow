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
library. `plot_figure_03.py` (Figure 3) and `plot_figure_04.py` (Figure 4)
need the packages in `requirements.txt` (`pip install -r requirements.txt`),
and Arial installed system-wide — each raises an error rather than silently substituting a
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
`(film, event_type, entity_id, timestamp)` for all 6,105 films.

- `film`: thin film ID, `TF000000` style.
- `event_type`: one of
  - measurement/process events (full datetime precision): `spin`, `img`,
    `cimg`, `line`, `giwaxs`, `rga`, `rgaout`
  - synthesis events (date only): `sp` (solid precursor weighed), `ss`
    (stock solution made), `ps` (precursor solution mixed)
- `entity_id`: the specific sample ID the row refers to (e.g. `SP000001`,
  `SS00003`, `PS000010`) for synthesis events; empty for measurement events,
  which are attributes of the film itself.
- `timestamp`: ISO 8601 with UTC offset (`YYYY-MM-DDTHH:MM:SS-07:00` or
  `-08:00`, US Pacific, the lab's local time zone) for measurement events;
  date-only (`YYYY-MM-DD`) for synthesis events, since the source bench
  records carry no time-of-day.

Notes:
- A film's `event_type` is **not unique per film** — a row can repeat for
  the same film (e.g. two `sp` and two `ss` rows), since a precursor
  solution is typically mixed from 2 stock solutions, each from its own
  solid precursor. `entity_id` is what disambiguates these, not
  `event_type`.
- Measurement timestamps use the corrected per-modality source: imaging
  from camera EXIF, RGA from `t0`, GIWAXS from `measurement_timestamp`.
- `rga`/`rgaout` mark sample load/unload on the RGA holder, not that a
  usable spectrum was necessarily recorded — cross-reference against the
  RGA dataset inventory for measurement completeness.

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
  not included — not needed to reproduce the figure from these coordinates.

The UMAP embedding and HDBSCAN clustering are not recomputed by this script;
the coordinates and cluster labels in `umap_clusters_by_film.csv` are the
ones used in the published figure.

## Figure 4 - RGA library review and quantification

```
scripts/plot_figure_04.py
```

Residual gas analysis (RGA) of the 36 analysed films made with β-alanine tert-butyl ester hydroiodide (`CC(C)(C)OC(=O)CC[NH3+].[I-]`) and BiI₃ or SbI₃. Spectra were fitted by non-negative least squares against NIST EI reference spectra with [rgakit](https://github.com/roncofaber/rgakit). Four panels:

- **A** - library construction: 634 rgakit fragments of the cation, 146 of them with a NIST spectrum, plus the environment library (16 chamber gases and hydrocarbons, 10 of them also fragments), giving 152 matched compounds; human review keeps 26.
- **B, C** - fit of the representative film TF001993 (SbI₃ 3:1, 80 °C) before (152 compounds) and after (26 compounds) review, coloured by compound source.
- **D** - total outgassing per film and the partial pressure of the film species in each film, one film per (metal, ratio, annealing temperature) condition.

**Data**:
- `data/fig04_rga_summary.json` - the cation (SMILES and precomputed 2D depiction coordinates, so RDKit is not needed), the number of generated fragments, and the representative run.
- `data/fig04_rga_library.csv` - the 152 matched compounds: `compound` (NIST name), `formula`, `smiles`, `source` (`film` = cation fragment only, `environment` = environment library only, `both`), and `kept_after_review`.
- `data/fig04_rga_runs.csv` - one row per run: `run_id`, `tf`, `batch` (RGA session), `metal`, `mixing_ratio` (organic:metal), `substrate`, `anneal_c` (annealing set temperature, °C), `outgas_area_torr_s` (background-corrected total pressure integrated over the exposure), `misfit_all` (Σ|observed − fit| / Σ observed over m/z 2-300 with the reviewed library), and `panel_d` (the lowest-misfit run per condition, shown in panel D).
- `data/fig04_rga_fit_before_review.csv`, `data/fig04_rga_fit_after_review.csv` - for the representative run, the observed spectrum (`observed`, Torr) and each fitted compound's contribution per integer m/z (2-300).
- `data/fig04_rga_partial_pressures.csv` - partial pressure (Torr) of each reviewed compound per run, from the fit weights scaled by ionization cross section relative to N₂.

The fits are not recomputed by this script; the values in these files are the ones used in the published figure.

## License

BSD 2-Clause, see `LICENSE`.
