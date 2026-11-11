"""Automatic PDF report (Spanish and English) with everything explained.

Uses the Low & Lou solution with noise (like a real magnetogram) to go through
every section: preprocessing, alpha fit, NLFFF, comparison with the exact
solution and material for the paper.

    python examples/03_report.py

For HMI data just replace the first two lines by::

    mag = sff.load_sharp_cea("data/harp377", rebin=4).balance_flux()
    study = sff.run_study(mag, nz=48)
"""

import solarff as sff

mag, exact = sff.low_lou_magnetogram(noise=30, nx=48, ny=48, nz=32)
study = sff.run_study(mag, nz=32, exact=exact)

sff.generate_report(study, "docs/informes/lowlou")
# -> docs/informes/lowlou/informe_es.pdf, docs/informes/lowlou/report_en.pdf,
#    docs/informes/lowlou/paper/ (methods, LaTeX tables, BibTeX, JSON)
