"""Benchmark against the analytical solution of Low & Lou (1990).

The vector magnetogram at z = 0 of the exact solution is extrapolated with the
potential, LFFF and NLFFF methods and compared with the exact field. This is the
way to check that the library works before using HMI data.

    python examples/01_lowlou_benchmark.py
"""

import matplotlib.pyplot as plt

import solarff as sff

NX, NZ = 48, 32

mag, exact = sff.low_lou_magnetogram(nx=NX, ny=NX, nz=NZ)
print(mag)
print("Criterios de Aly:", sff.aly_criteria(mag))

pot = sff.potential_field(mag, nz=NZ)

alpha = sff.estimate_alpha(mag, method="fit")
print(f"alpha (ajuste Bt) = {alpha:.4g} Mm^-1   (alpha_max = {sff.alpha_max(mag):.4g})")
lin = sff.lfff(mag, alpha=alpha, nz=NZ)

res = sff.nlfff(mag, nz=NZ, return_result=True)
nl = res.field

models = {"Exacto": exact, "Potencial": pot, "LFFF": lin, "NLFFF": nl}
seeds = sff.seeds_from_magnetogram(mag, n=80)   # grid over the positive polarity

print("\n=== Comparación (excluyendo la capa de amortiguamiento) ===")
table = sff.compare_models(models, reference="Exacto", potential="Potencial",
                           margin=nl.meta["nd"], seeds=seeds)
sff.print_table(table)

# Lines coloured by height (common scale) in 3D, top view and side view
sff.plot_comparison(models, seeds, save="lowlou_lines.png")
# Same numbered seeds in every model: connectivity comparison
sff.plot_comparison(models, seeds, color_by="seed", views=("3d", "top"),
                    save="lowlou_connectivity.png")
# local α: constant in the LFFF, variable in the NLFFF
sff.plot_alpha_maps(models, z=5.0, save="lowlou_alpha_maps.png")
sff.plot_alpha_histogram(models, seeds, reference="Exacto", save="lowlou_alpha_hist.png")
# Where each model fails with respect to the exact one
sff.plot_error_maps(models, reference="Exacto", heights=(5, 15, 30), save="lowlou_errors.png")

fig, ax = plt.subplots(figsize=(6, 4))
it, L, _ = zip(*res.history)
ax.semilogy(it, L)
ax.set_xlabel("iteración")
ax.set_ylabel("L")
ax.set_title("Convergencia NLFFF")
fig.tight_layout()
fig.savefig("lowlou_convergence.png", dpi=120)
print("\nFiguras: lowlou_comparison.png, lowlou_convergence.png")
