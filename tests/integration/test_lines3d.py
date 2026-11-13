"""The 3D viewer of ``tools/lines3d.py``: what it exports from a study and the page it writes (without network)."""
import base64
import importlib.util
import json
import os
import pickle
import re

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
_spec = importlib.util.spec_from_file_location("lines3d", os.path.join(ROOT, "tools", "lines3d.py"))
lines3d = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lines3d)


def test_export_and_page(lowlou_study, tmp_path):
    st = lowlou_study
    R = lines3d.export_region(st, every=2)
    n = len(R["counts"])
    assert n > 0 and all(len(R[k]) == n for k in ("conf5", "conf4", "acc5", "acc4", "apex", "length", "closed"))
    # the points decode inside the box (a cell of margin: the lines end on its faces)
    p = np.frombuffer(base64.b64decode(R["points"]), np.int16).reshape(-1, 3) / R["q"]
    assert len(p) == sum(R["counts"])
    x0, x1, y0, y1, z0, z1 = R["box"]
    m = R["pixel_mm"]
    assert p[:, 0].min() >= x0 - m and p[:, 0].max() <= x1 + m and p[:, 1].min() >= y0 - m and p[:, 1].max() <= y1 + m
    assert p[:, 2].min() >= z0 - m and p[:, 2].max() <= z1 + m
    # acceptance is decided before rounding the confidence to four decimals
    thr = lines3d.DEFAULT_CALIBRATION["threshold"]
    for c, a in (("conf5", "acc5"), ("conf4", "acc4")):
        c, a = np.array(R[c]), np.array(R[a], bool)
        assert np.all(c[a] >= thr - 1e-4) and np.all(c[~a] < thr + 1e-4)
    # vector fields: the three bases and every height of the extrapolation
    V = R["vec"]
    assert {"obs", "pp", "ext"} <= set(V)
    nb = V["nx"] * V["ny"] * 3
    assert np.frombuffer(base64.b64decode(V["obs"]["data"]), np.int16).size == nb
    assert np.frombuffer(base64.b64decode(V["ext"]["data"]), np.int16).size == len(V["z"]) * nb
    assert len(V["ext"]["png"]) == len(V["ext"]["scale"]) == len(V["z"])
    # the page: every marker replaced, and the data readable back
    html = lines3d.build_html({"test": {**R, "label": "Low & Lou </script>"}}, "// three", "// orbit", lang="es")
    assert not any(mk in html for mk in lines3d.MARKERS)
    assert html.lower().count("</script") == html.lower().count("<script")
    D = json.loads(re.search(r"^const D = (.*);$", html, re.M).group(1))
    assert D["lang"] == "es" and D["regions"]["test"]["label"] == "Low & Lou </script>"
    assert D["regions"]["test"]["acc5"] == R["acc5"]
    # a pickled study is found alone or inside a tuple, as examples/02_hmi_sharp.py writes it
    for i, obj in enumerate((st, (st, {"mc": None}))):
        path = tmp_path / f"s{i}.pkl"
        path.write_bytes(pickle.dumps(obj))
        assert isinstance(lines3d.load_study(str(path)), lines3d.Study)
