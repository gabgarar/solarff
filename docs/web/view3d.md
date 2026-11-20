---
hide:
  - navigation
  - toc
---

# Interactive 3D view

The field lines of three active regions, extrapolated with `solarff`, in 3D. Drag to rotate, scroll to
zoom, and hover over a line to see its confidence. The panel on the side changes the region, the
colouring and the arrows of the horizontal field.

<div class="view3d">
<iframe src="../3d/lines3d.html" title="Interactive 3D view of the field lines" loading="lazy"></iframe>
</div>

[Open it in a full window](3d/lines3d.html){ target="_blank" } · The page weighs about 5 MB and works offline once
loaded.

*The regions in the 3D view. **Region** and **HARP**: their NOAA and SHARP numbers. **Time (TAI)**: the magnetogram used. **Lines**: number of lines traced.*

| Region | HARP | Time (TAI) | Lines |
|---|---|---|---|
| NOAA 11158 | 377 | 2011-02-15 00:00 | 561 |
| NOAA 11429 | 1449 | 2012-03-06 23:36 | 452 |
| NOAA 12673 | 7115 | 2017-09-06 11:36 | 1035 |

- **Lines:** traced from seeds every two pixels where $\lvert B_z\rvert$ exceeds 5 % of its maximum.
- **Colouring:** by the height of the apex, by the [per-line confidence](confidence.md), or by which
  confidence model accepts the line.
- **Arrows:** the horizontal field of the magnetogram, of the preprocessed one, or of the extrapolation
  at any height, over a plane with $B_z$ at that height.
- **Language:** the interface switches between English and Spanish.

!!! note "Read the confidence with its limits"
    The confidence is calibrated on exact force-free fields and does not hold against a simulated
    active region. See [Tests against simulations](realistic.md).

## Your own region

```bash
python tools/lines3d.py my_study.pkl -o my_region.html    # a pickled Study with an NLFFF extrapolation
python tools/lines3d.py a.pkl b.pkl --every 1             # several regions, one seed per pixel
```

The input is a pickle of the result of [`run_study`](getting-started/study-report.md). The first run
downloads three.js r147 and checks its SHA-256; after that the page carries everything it needs.
