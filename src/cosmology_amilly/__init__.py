"""
Cosmology Amilly Package

A Python package for cosmological simulations and analysis.

Main features:
- K-space grid generation
- Fourier transforms for particle distributions
- Power spectrum calculation
"""

__version__ = "0.3.2"
__author__ = "Amilly Wang"
__email__ = "amillyforph@outlook.com"


from .gaussian_random_field import Gaussian_Random_Field
from .k_space_grid import K_Space_Grid
from .linear_growth import Linear_Growth_Factor, Linear_Growth_Rate
from .power_spectrum_estimator import Power_Spectrum_Estimator
from .power_spectrum import Power_Spectrum


# Public Interface
__all__ = [
    "Gaussian_Random_Field",
    "K_Space_Grid",
    "Linear_Growth_Factor",
    "Linear_Growth_Rate",
    "Power_Spectrum_Estimator",
    "Power_Spectrum",
]
