# Reproducing the paper

Every figure, table and number of the associated paper is produced by a script in `examples/`.

## One command

```bash
sh tools/reproduce_all.sh you@example.org          # everything, in order
sh tools/reproduce_all.sh "" calib                 # only one phase
```

The email is only passed to JSOC for the downloads and can be empty if the data are already in
`data/`. The phases are:

| Phase | What it runs | Rough time on an 8-core laptop |
|---|---|---|
| `calib` | calibration suite in boxes of four proportions, noise bias, robustness, flux ropes and external families | 5–7 h |
| `regiones` | the three active regions before and after each flare, with Monte Carlo | 6–8 h |
| `analisis` | AIA comparison, flare energies, systematic base errors, tables and figures | 1–2 h |
| `revision` | additional statistical and physical checks | a few hours |
| `pruebas` | tests beyond the calibration suite: resolution and unobserved faces in the regions, fields built on the observed flux, region models taken as the truth | about 8 h |

## The scripts

| Script | Purpose |
|---|---|
| `01_lowlou_benchmark.py` | benchmark against the Low & Lou analytical solution |
| `02_hmi_sharp.py` | full extrapolation of an HMI active region |
| `04_calibration_suite.py` | the calibration suite and the published coefficients |
| `06_robustness.py` | sensitivity to the label definition, solver and preprocessing parameters |
| `07_flux_rope_test.py` | external test on the Titov–Démoulin flux rope |
| `08_aia_validation.py`, `13`, `14` | comparison with AIA loops and its null tests |
| `12_flare_energy.py` | free-energy changes across three flares |
| `16_paper_figures.py` | the figures of the paper |
| `17_external_families.py` | three further external families |
| `18_applicability.py` | search for a global applicability indicator |
| `20_systematic_base.py` | systematic errors in the base: scale, height and azimuth |
| `21_line_table.py` | per-line table of the calibration (`docs/resultados/line_table.csv.gz`) |
| `22_resolution_check.py` | each region at half and at twice the published pixel, and with the smoothing of the preprocessing fixed in Mm |
| `23_mhs_test.py` | two families of exact magnetohydrostatic fields |
| `25_observed_flux_fields.py` | exact fields built on the observed flux of the three regions |
| `28_model_as_truth.py` | the published model of each region, cropped, taken as the truth |
| `29_boundary_sensitivity.py` | unobserved faces taken from the linear force-free field instead of the potential one |
| `24`, `26`, `27`, `30` | how the fifth term of the model was reached: ensemble indicator, length in box heights, boxes of other proportions, recalibration. They use the model of version 1.1 |

The additional checks are in `examples/revision/`:

| Script | Purpose | Where it appears in the paper |
|---|---|---|
| `r0_suite.py` | calibration suite rerun that also stores absolute errors and the external rope | baselines and label definitions |
| `r1_flare_energy.py` | split of the energy drop by zones and its sensitivity | flare figure and the hypotheses table |
| `r2_stats.py` | AUC, ECE, Brier score, calibration slope, confidence intervals, risk–coverage | validation results |
| `r3_natural_variation.py` | energy variation over the four pre-flare times | flare energies |
| `r4_physics.py` | volume potential, divergence against gauge, the two free energies | uncertainties appendix |
| `r7_models.py` | external rope, wrong-line definition, rule for the reduced model | model and external tests |
| `r8_indicators.py` | the nine indicators, alone and when removed or added | indicators table |
| `r9_coefficients.py` | coefficients with grouped-bootstrap intervals | coefficients table |
| `r10_energy_identity.py`, `r12_helicity_check.py` | convergence of the energy identity and of the helicity on exact fields | uncertainties appendix |
| `r11_label_nested.py` | other wrong-line definitions and nested validation | sensitivity and nested AUC |
| `r13_mc_indicator.py` | whether Monte Carlo stability adds to the three indicators (four terms, boxes of one proportion) | external tests |
| `r14_baselines.py` | baselines under the same validation scheme, folds and balance | baselines table and folds appendix |
| `r15_extra_checks.py` | unpenalised fit, equal-risk threshold, leave-one-noise-out and leave-one-resolution-out | training and sensitivity |
| `r16_decision_checks.py` | penalised, unpenalised and recalibrated models; grouped intervals of the risk; composition of the wrong-line label; correlations between indicators | training, validation results and external tests |
| `r17_external_coverage.py` | external families: risk against the threshold and size of the accepted errors | external tests |
| `r18_equal_height_robustness.py`, `r19_stability_vs_aia.py` | resolution test at equal apex height with harder controls; stability under resolution against the AIA loops | tests beyond the suite |
| `r20_height_rule_regions.py`, `r21_m3_regions.py` | the length measured in box heights applied to the regions, before it became the fifth term | history of the model |

## Result files

Every script writes its numbers to a JSON file in `docs/resultados/` or `docs/revision/resultados/`. The figures and tables of the paper are built from those files.

## Tests

```bash
pytest                 # unit and integration tests
pytest -m slow         # also the Low & Lou NLFFF at 64³
```

The unit tests check each piece against an exact answer. The integration tests run the full workflow
against Low & Lou and reproduce the published coefficients when the data are available.
