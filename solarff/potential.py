r"""Potential field ($\nabla\times\mathbf{B} = 0$): the $\alpha = 0$ case of the LFFF."""

from __future__ import annotations

from .field import VectorField3D
from .lfff import lfff
from .magnetogram import Magnetogram


def potential_field(mag: Magnetogram, nz: int, pad: float = 1.0) -> VectorField3D:
    """Potential extrapolation by FFT from ``mag.bz``.

    It is the minimum-energy state for the given Bz distribution; the
    energy difference with the NLFFF is the *free energy*.
    """
    return lfff(mag, alpha=0.0, nz=nz, pad=pad)
