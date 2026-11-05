# data/event_timestamps.csv

Long-format table of every timestamp underlying the pipeline-timing figures
(incl. Figure 2C): one row per `(film, event_type, entity_id, timestamp)`.

## Columns

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

## Notes

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
