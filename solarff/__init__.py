"""solarff: extrapolation of coronal magnetic fields from magnetograms.

Typical workflow:

    import solarff as sff

    mag = sff.load_sharp_cea("Br.fits", "Bp.fits", "Bt.fits").rebin(4)
    pot = sff.potential_field(mag, nz=64)
    alpha = sff.estimate_alpha(mag)
    lin = sff.lfff(mag, alpha=alpha, nz=64)
    nl = sff.nlfff(sff.preprocess(mag), nz=64)

    seeds = sff.seeds_from_magnetogram(mag, n=200)
    sff.plot_comparison({"Potential": pot, "LFFF": lin, "NLFFF": nl}, seeds)
    sff.print_table(sff.compare_models({"Potential": pot, "LFFF": lin, "NLFFF": nl}))

Conventions
-----------
* 3D arrays are indexed ``[ix, iy, iz]``; 2D ``[ix, iy]``.
* ``z`` is the height above the photosphere; ``z = 0`` is the magnetogram.
* Distances in Mm, field in Gauss, alpha in Mm^-1.
"""

from .field import VectorField3D
from .magnetogram import Magnetogram, aly_criteria
from .potential import potential_field
from .lfff import lfff, alpha_max
from .alpha import estimate_alpha, alpha_from_jz, alpha_best_fit
from .nlfff import nlfff, NLFFFResult
from .nlfff_torch import nlfff_lbfgs
from .preprocess import preprocess
from .tracing import trace_field_lines, seeds_from_magnetogram, FieldLine
from .metrics import (
    force_free_metrics,
    energy,
    compare_vector_fields,
    compare_connectivity,
    compare_models,
    print_table,
)
from .io import load_sharp_cea, download_sharp, search_sharp, from_sunpy, to_sunpy
from .analytic import low_lou_field, low_lou_magnetogram
from .plotting import (
    plot_field_lines, plot_comparison, plot_magnetogram,
    plot_alpha_maps, plot_alpha_histogram, plot_error_maps,
    line_alpha_stats, spread_indices, set_language,
)
from .study import Study, run_study, add_grad_rubin, reference_potential
from .grad_rubin import nlfff_grad_rubin
from .topology import twist_map, twist_number, squashing_q
from .stability import decay_index, decay_index_profile, pil_mask
from .validation import run_validation
from .utils import keep_awake
from .montecarlo import monte_carlo
from .helicity import (volume_potential, energy_decomposition, relative_helicity,
                       helicity_report, poisson_neumann)
from .confidence import (assess_lines, confidence_map, reliable_volume, line_indicators,
                         line_errors, calibrate)
from .report import generate_report

__version__ = "1.2.0"

__all__ = [
    "VectorField3D", "Magnetogram", "aly_criteria",
    "potential_field", "lfff", "alpha_max",
    "estimate_alpha", "alpha_from_jz", "alpha_best_fit",
    "nlfff", "nlfff_lbfgs", "NLFFFResult", "preprocess",
    "trace_field_lines", "seeds_from_magnetogram", "FieldLine",
    "force_free_metrics", "energy", "compare_vector_fields",
    "compare_connectivity", "compare_models", "print_table",
    "load_sharp_cea", "download_sharp", "search_sharp",
    "low_lou_field", "low_lou_magnetogram",
    "plot_field_lines", "plot_comparison", "plot_magnetogram",
    "plot_alpha_maps", "plot_alpha_histogram", "plot_error_maps",
    "line_alpha_stats", "spread_indices", "set_language",
    "Study", "run_study", "add_grad_rubin", "reference_potential", "nlfff_grad_rubin", "generate_report",
    "twist_map", "twist_number", "squashing_q",
    "decay_index", "decay_index_profile", "pil_mask", "run_validation", "monte_carlo", "keep_awake",
    "volume_potential", "energy_decomposition", "relative_helicity", "helicity_report", "poisson_neumann",
    "assess_lines", "confidence_map", "reliable_volume", "line_indicators", "line_errors", "calibrate",
]
