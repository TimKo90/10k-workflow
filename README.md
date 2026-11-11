# 10k-workflow

Data and reproduction code for selected figures of Kodalle et al., "Cross-Facility Materials Discovery Platform: Integrating Heterogeneous Data and Automated Experimental Pipelines into High-Throughput Workflows". This repository accompanies the paper's Data Availability statement.

## Layout and usage

```
data/       input data for the figure scripts
scripts/    one self-contained script per figure
figures/    generated output, not checked in
```

Each script reads from `data/` and writes to `figures/` using paths relative to its own location, so it runs with no arguments:

```
pip install -r requirements.txt
python scripts/plot_figure_03.py
```

| Figure        | Script                        | Output                                           | Needs                     |
|---------------|-------------------------------|--------------------------------------------------|---------------------------|
| 2C            | `plot_processing_timeline.py` | `figure_2c.svg`                                  | Python standard library   |
| 3             | `plot_figure_03.py`           | `figure_03.{svg,pdf,png}`                        | `requirements.txt`, Arial |
| 4             | `plot_figure_04.py`           | `figure_04.{svg,pdf,png}`                        | `requirements.txt`, Arial |
| S14, Table S1 | `plot_figure_s14.py`          | `figure_s14.{svg,pdf,png}`, `table_s1.{csv,tex}` | `requirements.txt`, Arial |

Figures 3, 4 and S14 raise an error if Arial is not installed rather than substituting a font with different metrics. Figures 3 and 4 plot precomputed values (UMAP and clustering, RGA fits); Figure S14 recomputes the peak positions, cosine similarities and Table S1 summary statistics from the reduced 1D spectra and diffraction profiles.

## Figure 2C: processing timeline

Lag of each processing step relative to the film's spin-coat time: calendar days for synthesis steps (solid precursor weighed, stock solution made, precursor solution mixed) and hours for characterization steps (line scan, imaging, RGA, GIWAXS). Synthesis records are date-only, so their lags are whole calendar days in US Pacific time, and a same-day lag is drawn at 12 h.

`data/event_timestamps.csv`: one row per event for all 6,105 films.

- `film`: thin-film ID (`TF000000`).
- `event_type`: measurement events `spin`, `img`, `cimg`, `line`, `giwaxs`, `rga`, `rgaout`; synthesis events `sp` (solid precursor), `ss` (stock solution), `ps` (precursor solution).
- `entity_id`: sample ID for synthesis events (`SP000001`, `SS00003`, `PS000010`); empty for measurement events.
- `timestamp`: ISO 8601 with US Pacific offset for measurement events; date only for synthesis events.

A film can have several rows of the same `event_type` (a precursor solution is usually mixed from two stock solutions, each from its own solid precursor); `entity_id` tells them apart. Imaging times come from camera EXIF, RGA from `t0` and GIWAXS from `measurement_timestamp`. `rga`/`rgaout` mark loading and unloading on the RGA holder, not that a usable spectrum was recorded.

## Figure 3: image and UV-Vis homogeneity, and the 10-spot UMAP

Built around two exemplar films, TF001028 (homogeneous) and TF003020 (inhomogeneous).

- **A**: well photo, central ROI crop, pixel-wise CIEDE2000 map against the ROI median colour, and the histogram of per-film median ΔE₀₀ (5,935 QC-valid films).
- **B**: 10-spot UV-Vis intensity and absorbance spectra, the pairwise mean |ΔA| matrix, and the histogram of UV-Vis spectral inhomogeneity (5,989 films).
- **C**: UMAP with 9 HDBSCAN clusters of the 5,730 films with both modalities, and the same map coloured by image ΔE₀₀, UV-Vis inhomogeneity, optical contrast and spectral centroid.

Data:

- `exemplar_image_de00_maps_TF001028_TF003020.npz`: well photos, ROI crops and bounds, and CIEDE2000 maps of the two exemplars.
- `exemplar_uvvis_spectra_TF001028_TF003020.npz`: wavelength grid, smoothed intensity and absorbance spectra, and the 10×10 |ΔA| matrices.
- `image_inhomogeneity_by_film.csv`: `ThinFilmID`, `delta_e_2000_median`, `qc_valid` for the 5,967 imaged films.
- `uvvis_inhomogeneity_by_film.csv`: `ThinFilmID`, `uvvis_robust_spectral_inhomogeneity` for the 5,989 films with UV-Vis.
- `umap_clusters_by_film.csv`: `ThinFilmID`, UMAP coordinates (`umap_10spot_eq77_1/2`), clusters (`cluster_k9_raw`, `cluster_k9_assigned`) and the four descriptors of panel C. The 77 input features of the embedding are not included.

## Figure 4: RGA library review and quantification

Residual gas analysis of the 36 films made from β-alanine tert-butyl ester hydroiodide (`CC(C)(C)OC(=O)CC[NH3+].[I-]`) with BiI₃ (16) or SbI₃ (20), one film per metal, organic:metal ratio and annealing temperature. Spectra were fitted by non-negative least squares against NIST reference spectra with [rgakit](https://github.com/roncofaber/rgakit).

- **A**: library construction. rgakit generates 634 fragments of the cation, 146 of which have a NIST spectrum. Adding the 16-compound environment library (10 shared with the fragments) gives 640 candidates and 152 matched compounds; human review keeps 26.
- **B, C**: fit of the representative film TF001993 (SbI₃ 3:1, 80 °C) with the 152 matched compounds and with the 26 reviewed compounds, coloured by compound source.
- **D**: total outgassing of each film and the partial pressures of the film species.

Data:

- `fig04_rga_summary.json`: the cation (SMILES and 2D depiction coordinates, so RDKit is not needed), the number of generated fragments and the representative run.
- `fig04_rga_library.csv`: the 152 matched compounds with `compound` (NIST name), `formula`, `smiles`, `source` (`film` fragment only, `environment` only, or `both`) and `kept_after_review`.
- `fig04_rga_runs.csv`: one row per film with `run_id`, `tf`, `batch` (RGA session), `metal`, `mixing_ratio`, `substrate`, `anneal_c` (annealing set temperature, °C), `outgas_area_torr_s` (background-corrected total pressure integrated over the exposure), `misfit_all` (Σ|observed - fit| / Σ observed over m/z 2-300 with the reviewed library) and `panel_d` (shown in panel D; the lowest-misfit run per condition, currently all 36).
- `fig04_rga_fit_before_review.csv`, `fig04_rga_fit_after_review.csv`: observed spectrum (`observed`, Torr) and the contribution of each fitted compound at each integer m/z from 2 to 300, for the representative film.
- `fig04_rga_partial_pressures.csv`: partial pressure (Torr) of each of the 26 reviewed compounds per film, from the fit weights scaled by the ionization cross section relative to N₂.

## Figure S14 and Table S1: multimodal reproducibility benchmark

Benchmark of automated high-throughput (HT) films annealed at 100 or 120 °C across three representative bismuth-iodide compositions — (PEAOH)BiI₄ (1:1, X/M = 4.0), (β-Ala-H)₂BiI₅ (2:1, X/M = 5.0) and (β-Ala-tBu)₃Bi₂I₉ (3:2, X/M = 4.5) — against manually prepared reference films (McCarthy et al., 2026), simulated single-crystal X-ray diffraction (SCXRD) powder patterns and manually prepared DMF control films (41 cohort films out of 82 with recorded annealing temperatures across all temperatures; 39 valid as-cast UV-vis scans, 17 unflagged GIWAXS patterns across 12 films and 7 primary 150 s EUV-RGA runs).

- **A**: normalized UV-vis excitonic absorption spectra comparing the HT cohort mean ± 1σ (cast from DMF) against reference films (cast from each cation's optimized solvent) and DMF control films, with excitonic peak wavelengths (λ_exc) and 400–650 nm spectral cosine similarities (cos θ).
- **B**: normalized 1D diffraction profiles comparing HT GIWAXS on ITO, reference GIWAXS on Si, reference powder XRD (Co Kα₁) and simulated SCXRD powder curves on a common 0.005 Å⁻¹ grid, with parabolic-refined HT reflection guides.
- **C**: normalized background-subtracted EUV-RGA mass spectra (m/z = 12–145 excluding H₂O at 18 and O₂ at 32) comparing HT and reference exposures, annotated with major radiolytic fragments and 20-ion diagnostic cosine similarities.

In addition to `figure_s14.{svg,pdf,png}` and `table_s1.{csv,tex}`, `plot_figure_s14.py` writes `figs14_peaks_all.csv`, `figs14_sensitivity.csv` and `figs14_text_values.csv` to `figures/`.

Data:

- `figs14_cohort_films.csv`: film metadata and inclusion flags for the 83 target-stoichiometry BiI₃ HT films (82 with recorded annealing temperatures plus early bare-glass commissioning film TF000004).
- `figs14_uvvis_spectra.csv`, `figs14_uvvis_scans.csv`: smoothed, NIR-baseline-subtracted absorbance spectra on a 1 nm grid (350–800 nm) for the cohort HT scans, reference films and DMF controls, together with per-film scan metadata, peak positions, cosine similarities and spatial-aggregation sensitivity columns.
- `figs14_diffraction_profiles.csv`, `figs14_scxrd_simulated_peaks.csv`: 1D radial profiles on a 0.005 Å⁻¹ grid (0.0025–2.9975 Å⁻¹) for all 17 individual HT GIWAXS scans, HT cohort means/SDs, reference GIWAXS, reference powder XRD and simulated SCXRD powder patterns, plus precomputed simulated Bragg positions and relative intensities for the cited low-q reflections.
- `figs14_rga_spectra.csv`, `figs14_rga_runs.csv`: background-subtracted partial pressures (Torr) at integer m/z = 1–300 and run-level metadata for the 3 reference runs, the 15 primary 150 s HT BiI₃ runs (7 at 100 or 120 °C) and 4 excluded 300 s commissioning runs.

## License

BSD 2-Clause, see `LICENSE`.
