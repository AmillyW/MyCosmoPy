import numpy as np
import pyfftw
from functools import cached_property
from .power_spectrum import Power_Spectrum
from .power_spectrum_estimator import Power_Spectrum_Estimator
from .k_space_grid import K_Space_Grid


class Gaussian_Random_Field:
    def __init__(self, L, N, z, file_path):
        self.z = z  # Redshift of the simulation
        self.file_path = file_path  # Path to the power spectrum data file

        self.grid = K_Space_Grid(L_box=L, N_grid=N)

        # ---Power Spectrum Interpolator---
        self.PL_fit = Power_Spectrum(z, file_path).Pk_interp

    def __call__(self):
        def Real_GRF(N, mu, sigma):
            Gaussian_Field_real = pyfftw.empty_aligned((N, N, N))
            Gaussian_Field_real[:] = np.random.normal(mu, sigma, (N, N, N))
            return Gaussian_Field_real

        return Real_GRF(self.grid.N_grid, 0, 1)

    def k_discretization(self):
        k = np.zeros_like(self.grid.n_k, dtype=float)
        k = self.grid.n_k * self.grid.k_F
        return k

    # ---Get the array result of the Fourier modes of the Gaussian Random Field using pyfftw---
    @cached_property
    def get_fourier_mode(self):
        a = self()
        self._fourier_mode = pyfftw.interfaces.numpy_fft.rfftn(a) * (self.grid.H**3)
        return self._fourier_mode

    @cached_property
    def get_rescaled_fourier_mode(self):
        delta_k = self.get_fourier_mode
        k_safe = np.where(self.grid.k == 0, 1, self.grid.k)
        scaling = np.where(
            self.grid.k == 0, 0, np.sqrt(self.PL_fit(k_safe) / (self.grid.H**3))
        )
        self._rescaled_fourier_mode = scaling * delta_k
        return self._rescaled_fourier_mode

    def binned_ps_grf(self, rescale=True):
        if rescale:
            fourier_modes = self.get_rescaled_fourier_mode
        else:
            fourier_modes = self.get_fourier_mode

        estimator_grf = Power_Spectrum_Estimator(
            self.grid.L_box, self.grid.N_grid, fourier_modes
        )
        P = estimator_grf.binned_power_spectrum()

        return P

    def binned_correction(self):
        delta_k = np.sqrt(self.PL_fit(self.grid.k) * self.grid.V)
        P = Power_Spectrum_Estimator(
            self.grid.L_box, self.grid.N_grid, delta_k
        ).binned_power_spectrum()
        return P
