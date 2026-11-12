# Per-line table of the calibration

`line_table.csv.gz` holds one row per traced field line of the extrapolations of exact force-free fields
behind the per-line confidence: 85 324 lines, 24 columns, comma-separated, gzip-compressed. It is written
by `examples/21_line_table.py` from the outputs of the calibration scripts and lets the model be refitted,
or another one tried, without repeating the extrapolations.

```python
import pandas as pd
t = pd.read_csv("docs/resultados/line_table.csv.gz")
```

## Sets

| `set` | Extrapolations | Lines | Content |
|---|---|---|---|
| `calibration` | 108 | 31 862 | the six families on which the published model is fitted |
| `rerun` | 126 | 37 262 | rerun of the same suite plus the held-out rope `TD2-relax`; used for the baselines, the label definitions and the external test |
| `external` | 54 | 16 200 | the three stress-test families `LL-IV`, `LFFF-delta` and `TD3-relax` |

Each family has 18 extrapolations: 3 noise models, 3 grid sizes and 2 noise realisations.

## Columns

| Column | Meaning |
|---|---|
| `set`, `family` | set and field family (see above) |
| `noise_model` | `uniform-10G`, `uniform-30G` (isotropic noise) or `HMI-10G-60G` (10 G in $B_z$, 60 G in $B_t$) |
| `size` | pixels per side of the base (32, 48 or 64) |
| `realization` | noise realisation (0 or 1) |
| `case`, `line` | index of the extrapolation within the set and of the line within the extrapolation |
| `sin_jb` | current-weighted mean sine of the angle between $\mathbf{J}$ and $\mathbf{B}$ along the line |
| `alpha_spread` | relative spread of $\alpha$ along the line: standard deviation weighted by arc length, divided by $\lvert\bar\alpha\rvert + \alpha_0$ |
| `div` | mean relative divergence along the line |
| `buffer` | fraction of the arc length inside the buffer layer |
| `open` | 1 if the line leaves through a side or the top of the box |
| `foot_snr` | transverse field at the weakest footpoint divided by its noise |
| `base_change` | relative correction of the transverse field at the footpoints |
| `apex` | apex height divided by the height of the box |
| `length` | length of the line divided by the horizontal diagonal of the box |
| `shape_error` | mean distance between the extrapolated and the exact line, sampled at 40 equal fractions of length, relative to the length of the exact line |
| `shape_error_abs` | the same distance in Mm |
| `endpoint_error` | mean distance between the end points of the two lines (Mm) |
| `topology_change` | 1 if one line is closed and the other open |
| `length_ref`, `length_model` | length of the exact and of the extrapolated line (Mm) |
| `wrong` | label used in the paper: 1 if `shape_error` > 0.1 or `topology_change` = 1 |
| `confidence` | 1 − probability of a wrong line given by the published model (`solarff.confidence.DEFAULT_CALIBRATION`) |

The published model uses `log10(alpha_spread)`, `buffer`, `log10(length)` and `(log10(length) + 0.54)²`.
`confidence` is empty (`nan`) for the four lines of the calibration set with an undefined indicator.
For the `calibration` set the column is the published model applied to its own training lines; the
out-of-sample figures of the paper come from refitting the model with each family left out.
