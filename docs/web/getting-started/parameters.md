# Parameter reference

The parameters of the functions used in a normal workflow, what they do and when to change them.
The defaults are those of the code. The [API reference](../api/core.md) has the full signatures.

## Data

### `download_sharp` {#download_sharp}

`download_sharp(harpnum, time, email, out_dir="data", segments=(...))`

*Parameters of `download_sharp`. **Parameter**: its name. **Default**: its value if not given; *required* if it has none. **Meaning**: what it is.*

| Parameter | Default | Meaning |
|---|---|---|
| `harpnum` | required | HARP number of the region. `search_sharp(time, noaa=...)` finds it |
| `time` | required | time in ISO format, in TAI, e.g. `"2011-02-15T00:00:00"` |
| `email` | required | address registered at JSOC; passed to JSOC only |
| `out_dir` | `"data"` | folder for the FITS files |
| `segments` | `Br`, `Bp`, `Bt` and their errors | the errors are needed for the Monte Carlo |

### `load_sharp_cea` {#load_sharp_cea}

`load_sharp_cea(br, bp=None, bt=None, rebin=1)`

*Parameters of `load_sharp_cea`. Columns as in the table above.*

| Parameter | Default | Meaning |
|---|---|---|
| `br` | required | the `Br` file, or the folder that holds the three segments |
| `bp`, `bt` | `None` | the other two files. With `br` alone the magnetogram has only $B_z$: enough for the potential and linear fields |
| `rebin` | 1 | average blocks of `rebin × rebin` pixels. 2 to 4 for HMI |

The `Magnetogram` it returns has `rebin(factor)`, `crop(xslice, yslice)`, `balance_flux()` and
`pad(...)`, and the components `bx`, `by`, `bz` in G on a grid of side `dx` Mm.

## Preprocessing {#preprocess}

`preprocess(mag, mu1=1, mu2=1, mu3=0.01, mu4=0.01, max_iter=500, fix_bz=False, verbose=True)`

It minimises $\mu_1 L_1 + \mu_2 L_2 + \mu_3 L_3 + \mu_4 L_4$ ([equations](../method.md#step-1-preprocessing)).

*Parameters of `preprocess`. **Parameter**, **Default**, **Meaning** as above. **When to change it**: the situations that call for another value.*

| Parameter | Default | Meaning | When to change it |
|---|---|---|---|
| `mu1`, `mu2` | 1 | weight of the net force and of the net torque | rarely |
| `mu3` | 0.01 | weight of the distance to the data. Larger keeps the result closer to the observation | 0.1 for photospheric data ([why](../realistic.md#recommended-treatment-of-a-photospheric-base)) |
| `mu4` | 0.01 | weight of the smoothing | rarely; it is the ratio `mu4/mu3` that matters |
| `max_iter` | 500 | iterations of the minimiser | if it stops before converging |
| `fix_bz` | `False` | change only the transverse field | to keep the observed $B_z$ exactly |
| `verbose` | `True` | print the progress | |

!!! warning
    The per-line confidence is calibrated with the defaults.

## Extrapolation {#extrapolation}

### `potential_field` and `lfff`

`potential_field(mag, nz, pad=1.0)` and `lfff(mag, alpha, nz, pad=1.0, heights=None)`

*Parameters of `potential_field` and `lfff`. Columns as in the first table of this page.*

| Parameter | Default | Meaning |
|---|---|---|
| `nz` | required | levels in height. The step is `mag.dx` |
| `alpha` | required | force-free parameter of the linear field, in Mm⁻¹. `estimate_alpha(mag)` gives one. It must be below `alpha_max(mag)` |
| `pad` | 1 | zero padding of the Fourier domain. 2 reduces the effect of the periodic copies, which matters for the decay index; it also lowers the largest admissible α |
| `heights` | `None` | explicit heights in Mm, to evaluate only some layers |

### `nlfff_lbfgs` {#nlfff_lbfgs}

`nlfff_lbfgs(mag, nz, initial=None, nd=None, relax_bottom=0.1, eps=0.03, smooth=0.01, div_weight=1.0, max_iter=2000, tol=1e-6, chunk=50, threads=None, verbose=True, return_result=False)`

The recommended solver. It needs PyTorch ([equations](../method.md#step-2-extrapolation)).

*Parameters of `nlfff_lbfgs`. Columns as in the table of `preprocess`.*

| Parameter | Default | Meaning | When to change it |
|---|---|---|---|
| `mag` | required | vector magnetogram, preferably preprocessed | |
| `nz` | required | levels in height | about half of the shorter side |
| `initial` | potential field | starting field; its side and top faces stay fixed | to start from another model |
| `nd` | about 10 % of the shorter side, at least 4 | width of the buffer layer next to the side and top faces, in cells | |
| `relax_bottom` | 0.1 | weight $\nu$ of the term that ties the transverse field of the base to the data. Smaller: freer base. `None`: fixed base | 0.01 for photospheric data; 1 or `None` for a nearly force-free base ([why](../realistic.md#how-much-to-trust-the-base)) |
| `eps` | 0.03 | regularisation near null points, as a fraction of max $\lvert B_z\rvert$ | |
| `smooth` | 0.01 | weight $\lambda$ of the smoothness term, which suppresses grid-scale oscillations | |
| `div_weight` | 1 | weight of the divergence term against the force term | 4 as a sensitivity test only |
| `max_iter` | 2000 | maximum iterations | 4000 for large regions |
| `tol` | 10⁻⁶ | stop when the functional falls relatively by less than this per iteration, averaged over `chunk` iterations | |
| `threads` | `None` | CPU threads for PyTorch; by default PyTorch decides, usually all the cores | to leave cores free |
| `return_result` | `False` | return also the history | |

After the run, `nl.meta` keeps `iterations`, `message`, `L_final`, and the two terms of the
functional, `L_force` and `L_data`.

### `nlfff` (classic) {#nlfff}

`nlfff(mag, nz, initial=None, nd=None, max_iter=5000, tol=1e-5, patience=50, mu0=None, eps=0.03, ...)`

Optimisation method of Wheatland et al. (2000) and Wiegelmann (2004), with a fixed base and no
dependencies beyond NumPy.
`patience` is the number of iterations over which the change of the functional is averaged, and
`mu0` the initial step. Use it when PyTorch is not available.

### `nlfff_grad_rubin` {#nlfff_grad_rubin}

`nlfff_grad_rubin(mag, nz, polarity="P", initial=None, max_iter=40, tol=1e-3, relax=0.5, stride=None, ...)`

*Parameters of `nlfff_grad_rubin`. Columns as in the first table of this page.*

| Parameter | Default | Meaning |
|---|---|---|
| `polarity` | `"P"` | take α from the positive (`"P"`) or the negative (`"N"`) polarity |
| `max_iter` | 40 | Grad–Rubin iterations |
| `tol` | 10⁻³ | stop when the field changes by less than this between iterations |
| `relax` | 0.5 | under-relaxation, which stabilises the iteration |
| `stride` | automatic | α is traced from one point of every `stride` per axis and interpolated. 1 is more accurate and slower |
| `alpha_threshold` | 0.05 | α is taken only where $\lvert B_z\rvert$ exceeds this fraction of the maximum |

## Field lines {#lines}

### `seeds_from_magnetogram` {#seeds_from_magnetogram}

See the table in [Field lines and confidence](lines-confidence.md#1-seeds-where-the-lines-start).

### `trace_field_lines` {#trace_field_lines}

`trace_field_lines(field, seeds, ds=0.5, max_steps=4000, b_min=0.001)`

*Parameters of `trace_field_lines`. Columns as in the first table of this page.*

| Parameter | Default | Meaning |
|---|---|---|
| `seeds` | required | array `(N, 3)` in Mm |
| `ds` | 0.5 | integration step, in cells |
| `max_steps` | 4000 | longest line, in steps |
| `b_min` | 0.001 | stop where the field strength falls below this value, in G |

### `assess_lines` and `confidence_map` {#assess_lines}

`assess_lines(field, lines, mag=None, noise_bt=100.0, calibration=None, extra=None)`

*Parameters of `assess_lines`. Columns as in the first table of this page.*

| Parameter | Default | Meaning |
|---|---|---|
| `mag` | `None` | the magnetogram used as the boundary condition; needed by some indicators |
| `noise_bt` | 100 | noise of the transverse field, in G |
| `calibration` | published coefficients | another set of coefficients, e.g. after recalibrating |
| `extra` | `None` | more indicators per line, such as the Monte Carlo instability |

`confidence_map` takes the same arguments plus `step` (one seed every `step` pixels) and `threshold`
(weakest field seeded, as a fraction of the maximum).

## A full study {#run_study}

See the tables in [A full study and its report](study-report.md).

## Analytical fields {#analytical-fields}

`low_lou_magnetogram(noise=0.0, rng=0, nx=64, ny=64, nz=64, n=1, m=1, l=0.3, phi=π/4, length_mm=50, bmax=1000)`

*Parameters of `low_lou_magnetogram`. Columns as in the first table of this page.*

| Parameter | Default | Meaning |
|---|---|---|
| `noise` | 0 | Gaussian noise added to the magnetogram, in G |
| `rng` | 0 | seed of the noise |
| `nx`, `ny`, `nz` | 64 | size of the grid |
| `n`, `m` | 1 | indices of the Low & Lou solution |
| `l` | 0.3 | depth of the source below the photosphere |
| `phi` | π/4 | tilt of the source axis |
| `length_mm` | 50 | side of the box in Mm |
| `bmax` | 1000 | maximum field in G |

The other fields are in [Benchmarks and simulations](benchmarks.md).
