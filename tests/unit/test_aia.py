"""Measurement of the loop orientation on synthetic images."""
import numpy as np


def test_aia_ridge_misalignment_synthetic():
    from solarff.aia import line_misalignment, ridge_orientation
    rng = np.random.default_rng(1)
    N = 200
    yy, xx = np.mgrid[0:N, 0:N].astype(float)
    img = np.full((N, N), 50.0)
    arcs = []
    for _ in range(6):
        cx, cy = rng.uniform(60, 140, 2)
        a, b, rot = rng.uniform(20, 50), rng.uniform(15, 40), rng.uniform(0, np.pi)
        t = np.linspace(0, np.pi, 200)
        X = cx + a * np.cos(t) * np.cos(rot) - b * np.sin(t) * np.sin(rot)
        Y = cy + a * np.cos(t) * np.sin(rot) + b * np.sin(t) * np.cos(rot)
        arcs.append(np.column_stack([X, Y]))
        for x, y in zip(X, Y):
            img += 50 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 4.5)
    img += rng.normal(0, 5, img.shape)
    th, st = ridge_orientation(np.clip(img, 1, None), sigma=2.0)
    thr = np.percentile(st, 90)
    assert np.nanmedian(line_misalignment(arcs, th, st, thr)) < 3  # aligned with the loops
    rnd = [np.column_stack([rng.uniform(40, 160) + np.arange(60) * np.cos(a),
                            rng.uniform(40, 160) + np.arange(60) * np.sin(a)]) for a in rng.uniform(0, np.pi, 150)]
    assert 35 < np.nanmean(line_misalignment(rnd, th, st, thr)) < 55  # unrelated: ~45°
