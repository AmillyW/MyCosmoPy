import numpy as np
import pyfftw
from functools import cached_property
import cosmology_amilly.power_spectrum as ps
from cosmology_amilly.power_spectrum_estimator import Power_Spectrum_Estimator


class Gaussian_Random_Field:
    def __init__(self, L, N, z, file_path):
        # ---Parameters of the simulation---
        self.L = L  # Box size in Mpc/h
        self.N = N  # Grid points per dimension
        self.z = z  # Redshift of the simulation
        self.file_path = file_path  # Path to the power spectrum data file

        self.H = self.L / self.N  # Grid spacing in Mpc/h
        self.V = self.L**3  # Volume of the box in Mpc/h^3
        self.k_F = 2 * np.pi / self.L  # Fundamental frequency in h/Mpc
        self.k_N = np.pi / self.H  # Nyquist frequency in h/Mpc
        self.n_k = np.arange(0, self.N // 2)  # Array of natural numbers

        # ---Construct the k-space grid---
        kx = np.fft.fftfreq(self.N, d=self.H) * 2 * np.pi
        ky = np.fft.fftfreq(self.N, d=self.H) * 2 * np.pi
        kz = np.fft.rfftfreq(self.N, d=self.H) * 2 * np.pi
        self.kX, self.kY, self.kZ = np.meshgrid(kx, ky, kz, indexing="ij")
        self.k = np.sqrt(self.kX * self.kX + self.kY * self.kY + self.kZ * self.kZ)
        self.discrete_k = self.k_discretization()

        # ---Power Spectrum Interpolator---
        self.PL_fit = ps.Power_Spectrum(z, file_path).Pk_interp

    def __call__(self):
        def Real_GRF(N, mu, sigma):
            Gaussian_Field_real = pyfftw.empty_aligned((N, N, N))
            Gaussian_Field_real[:] = np.random.normal(mu, sigma, (N, N, N))
            return Gaussian_Field_real

        return Real_GRF(self.N, 0, 1)

    def k_discretization(self):
        k = np.zeros_like(self.n_k, dtype=float)
        k = self.n_k * self.k_F
        return k

    # ---Get the array result of the Fourier modes of the Gaussian Random Field using pyfftw---
    @cached_property
    def get_fourier_mode(self):
        a = self()
        self._fourier_mode = pyfftw.interfaces.numpy_fft.rfftn(a) * (self.H**3)
        return self._fourier_mode

    @cached_property
    def get_rescaled_fourier_mode(self):
        delta_k = self.get_fourier_mode
        k_safe = np.where(self.k == 0, 1, self.k)
        scaling = np.where(self.k == 0, 0, np.sqrt(self.PL_fit(k_safe) / (self.H**3)))
        self._rescaled_fourier_mode = scaling * delta_k
        return self._rescaled_fourier_mode

    def binned_ps_grf(self, rescale=True):
        if rescale:
            fourier_modes = self.get_rescaled_fourier_mode
        else:
            fourier_modes = self.get_fourier_mode

        estimator_grf = Power_Spectrum_Estimator(self.L, self.N, fourier_modes)
        P = estimator_grf.binned_power_spectrum()

        return P

    def binned_correction(self):
        delta_k = np.sqrt(self.PL_fit(self.k) * self.V)
        P = Power_Spectrum_Estimator(self.L, self.N, delta_k).binned_power_spectrum()
        return P
