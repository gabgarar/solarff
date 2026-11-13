"""External families in boxes of other proportions: does the height term of the length transfer to unseen geometries?

The published model was fitted with the six calibration families in boxes of four proportions, and its fifth term (the
length in box heights) evens out the wrong lines accepted between those boxes, leaving out one family at a time. The
four external families (``TD2-relax``, ``LL-IV``, ``LFFF-delta``, ``TD3-relax``) had only been extrapolated in boxes of
standard proportion (``examples/17_external_families.py``). Here they are extrapolated in the two flat boxes and in the
tall one (``solarff.calibration.proportion_suite``: three noise models, one realisation, 36 extrapolations), and the
published model and that of version 1.1 (four terms, fitted with boxes of one proportion) are applied without refitting.

The criteria and the predictions were registered before running. The nine cases of ``LFFF-delta`` are those of
``examples/27_box_proportion.py --families LFFF-delta``, seen before the published model was fixed; the other 27 are new.

Writes ``docs/resultados/external_boxes.json``; checkpoint in ``data/proportion_external.ckpt``.

    python -u examples/31_external_boxes.py          # ~25 min
"""
import json
import os
import pickle
import warnings

import numpy as np

from solarff.calibration import PROPORTION_BOXES, add_box, box_name, ece, proportion_suite
from solarff.confidence import CALIBRATION_V1, DEFAULT_CALIBRATION, auc, confidence
from solarff.utils import code_version

warnings.filterwarnings("ignore")
FAMILIES = ["TD2-relax", "LL-IV", "LFFF-delta", "TD3-relax"]
T = DEFAULT_CALIBRATION["threshold"]
N_BOOT = 2000
OUT = os.path.join("docs", "resultados", "external_boxes.json")


def evaluate(d, cal):
    """Figures of the model ``cal`` on the cases ``d``; the interval of the risk resamples whole extrapolations."""
    cs = [np.nan_to_num(confidence(x["ind"], cal), nan=-1.0) for x in d]
    ys = [np.asarray(x["bad"]).astype(int) for x in d]
    ok = [c >= 0 for c in cs]
    c, y = np.concatenate([a[k] for a, k in zip(cs, ok)]), np.concatenate([a[k] for a, k in zip(ys, ok)])
    acc = c >= T
    out = {"n_cases": len(d), "n_lines": int(y.size), "bad_rate": float(y.mean()), "auc": auc(1 - c, y), "ece": ece(1 - c, y),
           "accepted": float(acc.mean()), "bad_accepted": float(y[acc].mean()) if acc.any() else None}
    if acc.any():
        rng = np.random.default_rng(0)
        b = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(d), len(d))
            a = np.concatenate([cs[j][ok[j]] >= T for j in i])
            if a.any():
                b.append(float(np.concatenate([ys[j][ok[j]] for j in i])[a].mean()))
        out["bad_accepted_ci"] = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    return out


def both(d):
    return {"default": evaluate(d, DEFAULT_CALIBRATION), "v1": evaluate(d, CALIBRATION_V1)}


data = proportion_suite(fields=FAMILIES, checkpoint=os.path.join("data", "proportion_external.ckpt"))
# reference: the same families in boxes of standard proportion
std = dict(pickle.load(open(os.path.join("data", "external_families.pkl"), "rb")))
std["TD2-relax"] = [x for x in pickle.load(open(os.path.join("data", "revision", "suite_r0.pkl"), "rb")) if x["field"] == "TD2-relax"]

out = {"code_version": code_version()["label"], "threshold": T, "n_boot": N_BOOT, "boxes": [list(b) for b in PROPORTION_BOXES],
       "calibration": "DEFAULT_CALIBRATION (five terms) and CALIBRATION_V1 (four terms, version 1.1), not refitted", "families": {}}
for fam in FAMILIES:
    d = [x for x in data if x["field"] == fam]
    r = {"standard": both(add_box(std[fam]))}
    for g in ("flat", "very flat", "tall"):
        r[g] = both([x for x in d if box_name(x) == g])
    r["flat boxes"] = both([x for x in d if box_name(x) in ("flat", "very flat")])     # the two flat boxes together
    out["families"][fam] = r
json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

fmt = lambda v: "-" if v is None else f"{100 * v:.1f}"  # noqa: E731
for fam, r in out["families"].items():
    print(fam)
    for g, q in r.items():
        a, b = q["default"], q["v1"]
        ci = a.get("bad_accepted_ci")
        print(f"  {g:10s} {a['n_cases']:2d} cases | wrong {100 * a['bad_rate']:.0f} % | five terms: AUC {a['auc']:.3f}, accepted {100 * a['accepted']:.0f} %, "
              f"wrong among them {fmt(a['bad_accepted'])} %" + (f" [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]" if ci else "")
              + f" | four terms: AUC {b['auc']:.3f}, accepted {100 * b['accepted']:.0f} %, wrong among them {fmt(b['bad_accepted'])} %")
print("written", OUT)
