import numpy as np


class Power_Spectrum_Estimator:
    def __init__(self, L, N, delta_k):
        self.L = L  # Box size in Mpc/h
        self.N = N  # Grid points per dimension
        self.delta_k = delta_k  # The field delta_k behind the power spectrum field

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

    def k_discretization(self):
        k = np.zeros_like(self.n_k, dtype=float)
        k = self.n_k * self.k_F
        return k

    def binned_power_spectrum(self):
        power_field = np.abs(self.delta_k) * np.abs(self.delta_k)

        P = np.zeros_like(self.n_k, dtype=float)

        for i, n_i in enumerate(self.n_k):  # n_k[i] = n_i, k_i = n_i * k_F
            k_i = n_i * self.k_F
            mask = (self.k >= k_i - self.k_F / 2) & (self.k <= k_i + self.k_F / 2)
            if np.any(mask):
                P[i] = np.mean(power_field[mask]) / self.V

        return P
