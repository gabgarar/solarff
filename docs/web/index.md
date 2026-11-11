# solarff

**Extrapolation of the solar coronal magnetic field from SDO/HMI vector magnetograms, with a
calibrated confidence for every field line.**

A nonlinear force-free (NLFFF) extrapolation gives a full 3D field, but not all of it deserves the
same trust. `solarff` computes the extrapolation and then tells you, line by line, how likely each
field line is to reproduce the true geometry.

![The five steps of the method](img/fig1_pipeline.png)

## What it does

| Task | Functions |
|---|---|
| Potential, linear and nonlinear force-free extrapolation | `potential_field`, `lfff`, `nlfff_lbfgs`, `nlfff` |
| Preprocessing of the vector magnetogram | `preprocess` |
| Field-line tracing | `trace_field_lines`, `seeds_from_magnetogram` |
| **Per-line confidence** | `assess_lines`, `confidence_map`, `reliable_volume` |
| Free energy and relative helicity with their uncertainties | `helicity_report`, `monte_carlo` |
| Twist, decay index and squashing factor Q | `twist_map`, `decay_index_profile`, `squashing_q` |
| Download and reading of SHARP data, sunpy interoperability | `download_sharp`, `load_sharp_cea`, `from_sunpy`, `to_sunpy` |
| PDF report of a region | `run_study`, `generate_report` |

## The idea in one figure

The confidence is a logistic regression on three indicators of each line: the variation of the
force-free parameter α along it, the fraction of the line inside the boundary layer of the
extrapolation, and its length. It is calibrated on exact force-free fields, where the true error of
every line is known, and validated by leaving out one whole family of fields at a time.

![Calibration and risk-coverage curve](img/fig2_calibration.png)

With the default threshold (confidence ≥ 0.8) about half of the lines are accepted, and about 4 % of
the accepted ones are wrong, against about 30 % of all lines.

## Where to go next

- [Getting started](getting-started.md): installation, data and a first extrapolation.
- [Method and equations](method.md): the physical context, every equation, and the inputs and outputs of each step.
- [Per-line confidence](confidence.md): how to apply it to your own extrapolation, and its limits.
- [Diagnostics](diagnostics.md): energy, helicity, twist, decay index, QSLs and Monte Carlo errors.
- [Reproducing the paper](reproduce.md): every figure and number, from the scripts.
- [API reference](api/core.md): generated from the source code.

`solarff` is free software under the BSD 3-Clause license. If you use it, please [cite it](cite.md).
