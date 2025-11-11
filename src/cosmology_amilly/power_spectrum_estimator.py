import numpy as np
from .k_space_grid import K_Space_Grid


class Power_Spectrum_Estimator:
    def __init__(self, L, N, delta_k):
        self.delta_k = delta_k  # The field delta_k behind the power spectrum field

        self.grid = K_Space_Grid(L_box=L, N_grid=N)

    def binned_power_spectrum(self):
        power_field = np.abs(self.delta_k) * np.abs(self.delta_k)

        P = np.zeros_like(self.grid.n_k, dtype=float)

        for i, n_i in enumerate(self.grid.n_k):  # n_k[i] = n_i, k_i = n_i * k_F
            k_i = n_i * self.grid.k_F
            mask = (self.grid.k >= k_i - self.grid.k_F / 2) & (
                self.grid.k <= k_i + self.grid.k_F / 2
            )
            if np.any(mask):
                P[i] = np.mean(power_field[mask]) / self.grid.V

        return P
