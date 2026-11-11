# Per-line confidence

Every traced line gets a confidence between 0 and 1, defined as one minus the estimated probability
that the line is wrong. A line is *wrong* when its apex height differs from that of the true line by
more than 10 %.

## Usage

```python
import solarff as sff

mag = sff.load_sharp_cea("data/harp377", rebin=4).balance_flux()
nl = sff.nlfff_lbfgs(sff.preprocess(mag), nz=48)

seeds = sff.seeds_from_magnetogram(mag, n=300)
lines = sff.trace_field_lines(nl, seeds)

a = sff.assess_lines(nl, lines, mag=mag)
a["confidence"]      # one value per line
a["reliable"]        # boolean mask: confidence >= 0.8
a["indicators"]      # the indicators the confidence is computed from
```

Two more helpers build on it:

```python
cm = sff.confidence_map(nl, mag=mag, step=2)            # confidence of the line leaving each pixel
vol = sff.reliable_volume(nl, lines, a["reliable"])     # 3D mask of cells threaded by reliable lines
```

![Reliable and unreliable lines in an active region](img/fig4_harp377.png)

## How it was calibrated

- **Training set:** six families of exact force-free fields (three Low & Lou variants, two exact
  linear fields and a relaxed Titov–Démoulin flux rope), three noise models and three grid sizes.
  That is 108 extrapolations and 31 862 lines whose true error is known.
- **Model:** a logistic regression on the variation of α along the line, the fraction of the line
  in the boundary layer, and the logarithm of the length with a quadratic term. No more complex
  classifier does better.
- **Validation:** leaving out one whole family at a time, so the model is always tested on a
  geometry it has not seen.

| Threshold | Lines accepted | Wrong among the accepted |
|---|---|---|
| 0.7 | 61 % | 7.5 % |
| **0.8 (default)** | 51 % | 4.0 % |
| 0.9 | 37 % | 1.6 % |

On an independent flux rope that was never used for training, the wrong lines among the accepted
ones drop from 34 % to 3 %.

![External flux rope test](img/fig3_flux_rope.png)

## Applying it to your own extrapolation

1. **Use the same code and the same preprocessing.** The preprocessing is the critical choice: with
   other parameters the confidence degrades and does not notice.
2. **Check the range of the extrapolation.** The mean current-weighted angle `CWsin` should be within
   0.05–0.21, the divergence energy fraction `E_div/E` within 0.006–0.19 and the boundary weight
   within 0.04–0.21. Outside these ranges, do not use the published probability.
3. **Pick the threshold for your use.** Use 0.9 if almost no accepted line may be wrong, 0.7 to keep
   more lines.
4. **Report results on reliable lines too.** If a per-line quantity changes when restricted to
   reliable lines, say by how much.
5. **Do not attribute energy to a zone of the volume** without repeating the calculation with
   different seed densities, dilations and thresholds.
6. **With another NLFFF code, recalibrate** with the same procedure
   (`examples/04_calibration_suite.py`).

!!! warning "Read it as a conditional score"
    Passing the range checks is necessary, not sufficient. For real magnetograms the confidence is a
    conditional reliability score, not a guaranteed error probability. Within one region, use it
    above all to rank lines.

## Known limits

- **Optimistic cases that nothing flags.** On four external families the ranking holds (AUC 0.86–0.95),
  but in two of them 18–21 % of the accepted lines are wrong. Their wrong lines look like good ones to the
  indicators, and no global quantity of the extrapolation warns of it. A high failure rate is not the
  criterion: another external family fails on a similar fraction of lines and its accepted set stays clean.
- **Systematic errors in the magnetogram.** A scale error of ±30 % in the transverse field, or taking
  it one cell higher than the vertical field, barely changes the result. A systematic rotation of the
  azimuth does: at about 5° or more the probability becomes optimistic, and at 10° the ranking
  degrades as well. The force and torque balance of the magnetogram (Aly criteria) does not reveal it.
- **Idealised training fields.** The calibration fields have an exactly force-free photosphere. Real
  photospheres do not.
- **One code.** The published coefficients are valid for the solver in this library.

The coefficients are in `solarff.confidence.DEFAULT_CALIBRATION`. The full analysis is in the
associated paper.
