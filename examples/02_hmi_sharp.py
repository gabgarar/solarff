"""Full analysis of an active region with SDO/HMI SHARP CEA data.

Downloads the data from JSOC (if not already present), extrapolates (potential,
LFFF, NLFFF), runs the Monte Carlo uncertainties and the Grad–Rubin solutions,
and generates the PDF report (Spanish and English).

Examples::

    # NOAA 11158 before the X2.2 flare (default)
    python examples/02_hmi_sharp.py --email you@example.com

    # Any other region, by NOAA or HARP number
    python examples/02_hmi_sharp.py --noaa 12673 --time 2017-09-06T08:00 --email you@example.com
    python examples/02_hmi_sharp.py --harp 7115 --time 2017-09-06T08:00 --email you@example.com

    # Faster: no Monte Carlo and no Grad–Rubin
    python examples/02_hmi_sharp.py --noaa 12673 --time 2017-09-06T08:00 --mc 0 --no-grad-rubin

``--rebin`` and ``--nz`` are chosen automatically from the size of the cut-out
(longer side ~190 pixels after rebinning; height ~ half of the shorter side) if
not given. The computations are saved in ``data/`` and reused on the next run,
so the report can be regenerated without repeating the extrapolations.
"""

import argparse
import json
import math
import os
import pickle
import sys
import time

import solarff as sff

p = argparse.ArgumentParser(description="Extrapolación NLFFF completa de una región activa HMI")
g = p.add_mutually_exclusive_group()
g.add_argument("--harp", type=int, help="número HARP (p. ej. 377)")
g.add_argument("--noaa", type=int, help="número NOAA (se busca su HARP en JSOC)")
p.add_argument("--time", default="2011-02-15T00:00:00", help="instante (ISO, TAI), p. ej. 2011-02-15T00:00")
p.add_argument("--email", help="email registrado en JSOC (sólo para descargar)")
p.add_argument("--rebin", type=int, help="factor de reducción de resolución (auto)")
p.add_argument("--nz", type=int, help="niveles en altura (auto)")
p.add_argument("--mc", type=int, default=10, help="realizaciones del Monte Carlo (0 = no)")
p.add_argument("--no-grad-rubin", action="store_true", help="no calcular Grad–Rubin")
p.add_argument("--no-report", action="store_true", help="sólo la extrapolación (caché), sin informe")
p.add_argument("--out", help="carpeta del informe (por defecto docs/informes/harp<N>_<fecha>)")
args = p.parse_args()

if args.harp is None and args.noaa is None:
    args.harp = 377  # NOAA 11158 by default
if args.noaa is not None:
    rows = sff.search_sharp(args.time, noaa=args.noaa, email=args.email)
    if len(rows) == 0:
        sys.exit(f"No hay SHARP para NOAA {args.noaa} en {args.time}")
    args.harp = int(rows.iloc[0]["HARPNUM"])
    print(f"NOAA {args.noaa} -> HARP {args.harp}")

stamp = args.time.replace("-", "").replace(":", "").replace("T", "_")[:13]
data_dir = f"data/harp{args.harp}_{stamp}"
if args.harp == 377 and args.time.startswith("2011-02-15T00:00") and os.path.isdir("data/harp377"):
    data_dir = "data/harp377"  # compatibility with earlier downloads
if not os.path.isdir(data_dir) or not any(f.endswith("Br.fits") for f in os.listdir(data_dir)):
    if not args.email:
        sys.exit("Los datos no están descargados: añade --email <registrado en JSOC>")
    sff.download_sharp(args.harp, args.time, email=args.email, out_dir=data_dir)

raw = sff.load_sharp_cea(data_dir)
rebin = args.rebin or max(1, math.ceil(max(raw.shape) / 190))
mag = raw.rebin(rebin).balance_flux()
nz = args.nz or int(min(96, max(24, 4 * round(min(mag.shape) / 8))))  # ~half of the shorter side, multiple of 4
print(mag, mag.meta.get("T_REC"), "NOAA", mag.meta.get("NOAA_ARS"), f"| rebin {rebin}, nz {nz}")

cache = f"{data_dir}_r{rebin}_nz{nz}_mc{args.mc}.pkl"
if data_dir == "data/harp377" and rebin == 4 and nz == 48:
    cache = f"data/harp377_study_mc{args.mc}.pkl"  # name used by 08, 09, 10, 12 and the revision scripts
t0 = time.time()
if os.path.exists(cache):
    study, mc = pickle.load(open(cache, "rb"))
    print(f"Cargado de {cache}")
else:
    study = sff.run_study(mag, nz=nz, nlfff_kw={"max_iter": 4000}, n_seeds=100, name=f"HARP {args.harp}")
    print(f"Extrapolaciones: {time.time() - t0:.0f} s")
    mc = sff.monte_carlo(study, n=args.mc) if args.mc else None
    if mc:
        print(f"Monte Carlo: {time.time() - t0:.0f} s")
    pickle.dump((study, mc), open(cache, "wb"))

if not args.no_grad_rubin and not getattr(study, "grad_rubin", None):
    sff.add_grad_rubin(study)
    print(f"Grad-Rubin: {time.time() - t0:.0f} s")
    pickle.dump((study, mc), open(cache, "wb"))
if args.no_report:
    sys.exit(0)

noaa = mag.meta.get("NOAA_ARS") or args.noaa or "?"
out = args.out or os.path.join("docs", "informes", f"harp{args.harp}" + ("" if data_dir == "data/harp377" else f"_{stamp}"))
aia = None
# AIA validation of this time (examples/08_aia_validation.py --out <data_dir>_aia.pkl);
# HARP 377 at 00:00 keeps its old name.
aia_pkl = f"{data_dir}_aia.pkl"
if not os.path.exists(aia_pkl) and data_dir == "data/harp377":
    aia_pkl = "data/aia_validation_harp377.pkl"
if os.path.exists(aia_pkl):
    aia = pickle.load(open(aia_pkl, "rb"))
# Results of the three regions of the paper (examples/12 and 15), if they already exist
res_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "resultados")
regions = None
try:
    with open(os.path.join(res_dir, "regions_summary.json"), encoding="utf-8") as fh_s,             open(os.path.join(res_dir, "flare_energy.json"), encoding="utf-8") as fh_f:
        regions = {"summary": json.load(fh_s), "flares": json.load(fh_f)}
except (OSError, ValueError):
    pass
sff.generate_report(study, out, monte_carlo=mc, aia=aia, regions=regions,
                    title=f"NOAA {noaa} (HARP {args.harp}) --- {mag.meta.get('T_REC', args.time)}")
print(f"Total: {time.time() - t0:.0f} s")
