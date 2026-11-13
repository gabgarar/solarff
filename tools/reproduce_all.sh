#!/bin/sh
# Reproduce todas las cifras, tablas y figuras del artículo, en orden.
# Uso: sh tools/reproduce_all.sh EMAIL [fase]
#   EMAIL: email registrado en JSOC (sólo se pasa a las descargas; no se guarda). Puede ir
#          vacío ("") si los datos de SHARP y AIA ya están en data/.
#   fase:  calib | regiones | analisis | revision | pruebas | todo (por defecto: todo)
# Tiempos orientativos en un portátil de 8 núcleos. Se escriben: data/ (no versionado),
# solarff/*.json (lo que lee la librería), docs/resultados/, docs/revision/resultados/,
# docs/informes/ y docs/paper/figuras/.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python.exe}
export PYTHONIOENCODING=utf-8
EMAIL=$1
FASE=${2:-todo}

calib() {   # ~5-7 h; campos exactos: calibración, sesgo, robustez, cuerdas
  # las 108 cajas de proporción estándar (la batería de la versión 1.1) y, con --boxes, las 54 de otras proporciones:
  # el modelo publicado (cinco términos) se ajusta con las 162
  $PY examples/04_calibration_suite.py --extra-fields TD-relax
  $PY examples/04_calibration_suite.py --extra-fields TD-relax --boxes --write-package
  $PY examples/04_calibration_suite.py --extra-fields TD-relax --sizes 32 48 --mc 4
  $PY examples/05_bias_validation.py
  $PY examples/06_robustness.py
  $PY examples/07_flux_rope_test.py --field TD
  $PY examples/07_flux_rope_test.py --field TD-relax
  $PY examples/17_external_families.py   # tres familias externas nuevas (LL-IV, LFFF-delta, TD3-relax), ~1 h
  $PY examples/23_mhs_test.py            # dos familias magnetohidrostáticas exactas
  $PY examples/11_divergence_weight_exact.py --weights 1 4 --package
}

regiones() {   # ~6-8 h; descarga SHARP de JSOC si no está en data/ (HARP 377 = NOAA 11158,
               # 1449 = NOAA 11429, 7115 = NOAA 12673)
  # Antes y después de cada fulguración: extrapolación, Monte Carlo, Grad-Rubin e informe
  for job in "377 2011-02-15T00:00" "377 2011-02-15T03:00" "1449 2012-03-06T23:36"              "1449 2012-03-07T02:00" "7115 2017-09-06T11:36" "7115 2017-09-06T13:00"; do
    set -- $job
    $PY examples/02_hmi_sharp.py --harp $1 --time $2 --email "$EMAIL" --mc 10
  done
  # Instante de referencia y variación natural previa: sólo la extrapolación
  for job in "377 2011-02-14T21:00" "377 2011-02-14T22:00" "377 2011-02-14T23:00"              "1449 2012-03-06T20:36" "1449 2012-03-06T21:36" "1449 2012-03-06T22:36"              "7115 2017-09-06T09:48" "7115 2017-09-06T10:24" "7115 2017-09-06T11:00"; do
    set -- $job
    $PY examples/02_hmi_sharp.py --harp $1 --time $2 --email "$EMAIL" --mc 0 --no-grad-rubin --no-report
  done
  $PY examples/09_divergence_weight.py --weights 1 2 3 4 8 --package   # barrido de w_d en HARP 377
  $PY examples/10_tracing_step.py --package                             # paso del trazador en HARP 377
}

analisis() {   # ~1-2 h; AIA, fulguraciones, tabla de regiones y figuras del artículo
  $PY examples/08_aia_validation.py --email "$EMAIL"
  for d in data/harp1449_20120306_2336 data/harp1449_20120307_0200 data/harp377_20110215_0300 \
           data/harp7115_20170906_1136 data/harp7115_20170906_1300; do
    $PY examples/08_aia_validation.py --email "$EMAIL" --cache "$(ls ${d}_r*_mc10.pkl)" \
        --sharp "$(ls $d/*Br.fits)" --time-utc auto --out ${d}_aia.pkl --region "$(basename $d)"
  done
  $PY examples/13_aia_null_test.py --n 200
  $PY examples/13_aia_null_test.py --n 200 --shift-mm 5 15 --no-save \
      --json docs/revision/resultados/r5_aia_null_5_15.json
  $PY examples/14_aia_npoints_bias.py
  $PY examples/12_flare_energy.py
  $PY examples/revision/r1_flare_energy.py     # energía por zonas y su sensibilidad (figura 6)
  $PY examples/revision/r3_natural_variation.py
  $PY examples/15_regions_table.py
  $PY examples/19_preprocess_change.py   # cambio del magnetograma por el preprocesado (regiones y base exacta), ~2 min
  $PY examples/20_systematic_base.py   # errores sistemáticos en la base (escala, altura, azimut), ~40 min
  # los informes incluyen la sección de las tres regiones: se regeneran con la tabla ya escrita
  for job in "377 2011-02-15T00:00" "377 2011-02-15T03:00" "1449 2012-03-06T23:36" \
             "1449 2012-03-07T02:00" "7115 2017-09-06T11:36" "7115 2017-09-06T13:00"; do
    set -- $job
    $PY examples/02_hmi_sharp.py --harp $1 --time $2 --mc 10
  done
}

revision() {   # revisión de 9-nov-2026 (docs/revision/): estadística, física e indicadores
  $PY examples/revision/r0_suite.py
  $PY examples/18_applicability.py       # indicador global de aplicabilidad: necesita 04, 17, r0 y los pkl de las regiones, ~1.5 h
  $PY examples/revision/r2_stats.py --tag original --suite data/calibration_suite_s32-48-64_r2_mc0_plus_TD-relax.pkl \
      --features sin_jb alpha_spread buffer apex length length_sq      # el modelo de cinco indicadores de los primeros borradores
  $PY examples/revision/r2_stats.py --tag nuevo
  $PY examples/revision/r2_stats.py --tag nuevo_l2_0 --no-ablation --l2 0
  $PY examples/revision/r4_physics.py
  $PY examples/revision/r7_models.py
  $PY examples/revision/r8_indicators.py
  $PY examples/revision/r9_coefficients.py
  $PY examples/revision/r10_energy_identity.py
  $PY examples/revision/r11_label_nested.py
  $PY examples/revision/r12_helicity_check.py
  $PY examples/revision/r13_mc_indicator.py
  $PY examples/revision/r14_baselines.py        # líneas base, particiones y equilibrio del diseño
  $PY examples/revision/r15_extra_checks.py     # ajuste sin penalizar, umbral de igual riesgo, dejar fuera ruido o resolución
  $PY examples/revision/r16_decision_checks.py  # variantes de calibración, intervalos del riesgo, etiqueta, correlaciones (necesita 17 y los pkl de AIA)
  $PY examples/revision/r17_external_coverage.py  # familias externas: riesgo según el umbral y tamaño de los errores aceptados
  $PY examples/16_paper_figures.py   # figuras del artículo: necesitan r1 (fase analisis) y r7 (esta fase)
  $PY examples/21_line_table.py      # tabla por línea de la calibración: necesita 04, 17 y r0
}

pruebas() {   # pruebas del 12 y 13 de noviembre de 2026; las extrapolaciones nuevas suman ~8 h
  # sensibilidad a la resolución: cada región al doble y a la mitad de la resolución publicada
  for r in 11158 11429 12673; do
    $PY examples/22_resolution_check.py --part region --region $r
    $PY examples/22_resolution_check.py --part region --region $r --coarser
  done
  $PY examples/22_resolution_check.py --part region --region 11158 --physical-smoothing   # alisado del preprocesado fijado en Mm
  $PY examples/revision/r18_equal_height_robustness.py
  $PY examples/revision/r18_equal_height_robustness.py --v1
  $PY examples/revision/r19_stability_vs_aia.py
  $PY examples/revision/r19_stability_vs_aia.py --v1
  $PY examples/25_observed_flux_fields.py                                      # campos exactos sobre el flujo observado
  $PY examples/25_observed_flux_fields.py --alpha-frac 0.5 --fields LFFF-obs   # los mismos con cizalla moderada
  for s in pre post first; do $PY examples/28_model_as_truth.py --set $s; done # modelos de las regiones tomados como verdad
  $PY examples/29_boundary_sensitivity.py                                      # caras no observadas: potencial frente a LFFF
  # históricos: cómo se llegó al quinto término. Usan el modelo de la versión 1.1 (CALIBRATION_V1)
  $PY examples/24_ensemble_indicator.py
  $PY examples/27_box_proportion.py
  $PY examples/27_box_proportion.py --families TD-relax LFFF-delta
  $PY examples/26_box_height_length.py
  $PY examples/30_recalibration_boxes.py
  $PY examples/revision/r20_height_rule_regions.py
  $PY examples/revision/r21_m3_regions.py
}

case $FASE in
  calib) calib ;; regiones) regiones ;; analisis) analisis ;; revision) revision ;; pruebas) pruebas ;;
  todo) calib; regiones; analisis; revision; pruebas ;;
esac
