"""
========================================
Radar Retrievals (:mod:`pyart.retrieve`)
========================================

.. currentmodule:: pyart.retrieve

Radar retrievals.

Radar retrievals
================

.. autosummary::
    :toctree: generated/

    kdp_maesaka
    calculate_snr_from_reflectivity
    fetch_radar_time_profile
    map_profile_to_gates
    steiner_conv_strat
    texture_of_complex_phase
    grid_displacement_pc
    grid_shift
    add_grids

"""

from .kdp_proc import kdp_maesaka
from .echo_class import steiner_conv_strat
from .gate_id import map_profile_to_gates, fetch_radar_time_profile
from .simple_moment_calculations import calculate_snr_from_reflectivity

try:
    from .advection import  grid_displacement_pc, grid_shift, add_grids
    _ADVECTION_AVAILABLE = True
except:
    _ADVECTION_AVAILABLE = False

__all__ = [s for s in dir() if not s.startswith('_')]
