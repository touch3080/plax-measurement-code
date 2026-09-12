# Conditional CMR axis-scale diagnostics

This diagnostic asks how the recorded diameters and three-reader mean EF change
under two explicitly declared axis-scale assumptions. It does not identify raw
source candidates, certify FOV definitions, measure scanner spacing or reconstruct
images. The caller must select an authorized local evidence subset first.

The original study used a nominal **1.5 mm/source-pixel** diameter conversion.
The fixed nominal diameter is the input here. Its saved M-mode distance and
display-resize correction must already have been applied upstream; the length of
the supplied source-frame line is used only for its direction.

## Run

Use the standard repository environment; no model download is required.

```bash
python scripts/analyze_conditional_scale.py --input examples/conditional_scale_synthetic.csv --output-dir outputs/conditional-synthetic
python scripts/analyze_conditional_scale.py --input /authorized/preselected.csv --output-dir /results/new-conditional --scale-mode fov
```

The default output is `conditional_scale_aggregate.json`, without case keys or
measurements. Add `--write-case-results` to write local
`conditional_measurements.csv` and `diagnostic_selection.csv`. These optional
files carry the supplied case keys and should remain subject to the applicable
data agreement. The CLI refuses a nonempty output directory and checks the input
SHA-256 before and after processing.

The supplied example contains four invented cases with no source-data counterpart.
It illustrates the schema and software behavior; it is not evidence of accuracy.

## One long CSV: 18 rows per case

Supply every combination of three readers, three methods and two phases.

| Column | Meaning |
|---|---|
| `case_id` | Nonempty opaque pairing key, sorted lexicographically. |
| `reader` | `reader01`, `reader02` or `reader03`. |
| `method` | `static`, `dynamic` or `peak_to_peak`. |
| `phase` | `ED` or `ES`. |
| `nominal_diameter_cm` | Recorded diameter after the nominal 1.5 mm/pixel conversion, in **cm**. Used ED must be positive; ES can be zero. |
| `line_dx_px`, `line_dy_px` | Endpoint differences `b_x-a_x`, `b_y-a_y` in the stored source-frame grid. x is the column axis and y is the row axis. These are not displayed M-mode coordinates. |
| `reference_ef` | Supplied frozen case-level functional comparator, in percentage points; identical on all 18 rows. |
| `frozen_reader_ef` (optional) | Frozen nominal EF for the row's reader/method, repeated on both phases. When present, the recomputed nominal EF must agree within 1e-10 percentage points. It is never a fitting target. |

For `--scale-mode fov`, also supply the following case-level fields, repeated
identically on all 18 rows:

| Column | Conditional input |
|---|---|
| `fov_x_mm`, `fov_y_mm` | Positive FOV values in **mm**; their physical interpretation remains an assumption. |
| `recon_matrix_x`, `recon_matrix_y` | Positive integer reconstruction-matrix fields. |
| `stored_width_px` | Positive integer stored x/readout grid width. |

The declared alternatives are:

- **A:** `sx = fov_x_mm / recon_matrix_x`, `sy = fov_y_mm / recon_matrix_y`.
- **B:** `sx = fov_x_mm / stored_width_px`, with the same `sy` as A.

B is one alternative readout-FOV convention; A/B are not an exhaustive model of
the unknown original spatial conversion. A stored width twice the matrix field
does not by itself determine which physical FOV convention applies. Matching grid
dimensions alone also do not verify physical calibration or source identity.

Alternatively, `--scale-mode axes` accepts four explicit positive case-level
columns: `a_sx_mm_per_px`, `a_sy_mm_per_px`, `b_sx_mm_per_px`,
`b_sy_mm_per_px`. These are declared hypothetical axis scales; the CLI does not
interpret their values as calibrated spacing. FOV/grid columns are unnecessary
in this mode. Both modes always evaluate A and B together on one common subset.

## Computation and missing information

For each used source-frame line, normalize `(dx,dy)` to `(ux,uy)` and calculate:

```text
effective_mm_per_pixel = sqrt((ux*sx)^2 + (uy*sy)^2)
conditional_diameter_cm = nominal_diameter_cm * effective_mm_per_pixel / 1.5
volume_ml = 7 * diameter_cm^3 / (2.4 + diameter_cm)
EF_percent = 100 * (EDV - ESV) / EDV
case_method_EF = mean(EF_reader01, EF_reader02, EF_reader03)
```

Equal axis scales use their exact common value. Thus A/B axis scales of
1.5/1.5 reproduce the nominal EF exactly for complete directions. Reversing or
rescaling the direction vector has no effect. Do not multiply the vector's full
length by the scale in place of the supplied nominal diameter.

**Dynamic ED always reuses the same reader's static ED diameter and direction.**
Dynamic ES uses its dynamic row; peak-to-peak uses its own ED and ES. The dynamic
ED row is retained for a uniform schema but its diameter/direction fields are
ignored. EF is calculated separately for each reader before averaging. Neither
diameters nor a varying number of available readers are averaged.

Missing, infinite or zero-length used directions exclude the entire case from
both scenarios and all three methods. Directions are not filled from other
records. At least three common complete cases are required. Duplicate or absent
reader/method/phase rows, inconsistent reference/scale fields, invalid used
diameters and nonpositive scales raise errors. Reference values and diameters do
not select the diagnostic subset. No EF clipping or physiologic-range exclusion
is performed. Undefined correlations from a constant vector are reported as
JSON `null` rather than a numerical correlation.

The output reports axis/effective-scale distributions, per-reader diameter/EF
changes, three-reader mean EF changes, same-subset nominal/conditional agreement
and paired method-minus-static point differences. No confidence interval,
hypothesis test, scale optimization, calibration fit, refit or replacement of
frozen study estimates is performed.

## Provenance

The pure functions are in `src/plax_measurement/conditional_scale.py`.
The mathematical source and the scoped local comparison record are documented
in `conditional_scale_provenance.json`. Model assets remain at
[release v0.1.2](https://github.com/touch3080/plax-measurement-code/releases/tag/v0.1.2).
