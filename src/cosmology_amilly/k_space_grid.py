import numpy as np


class K_Space_Grid:
    def __init__(self, L_box, N_grid):
        # ---Parameters of the simulation---
        self.L_box = L_box  # Box size in Mpc/h
        self.N_grid = N_grid  # Grid points per dimension
        self.H = self.L_box / self.N_grid  # Grid spacing in Mpc/h
        self.V = self.L_box**3  # Volume of the box in Mpc/h^3
        self.k_F = 2 * np.pi / self.L_box  # Fundamental frequency in h/Mpc
        self.k_N = np.pi / self.H  # Nyquist frequency in h/Mpc
        self.n_k = np.arange(0, self.N_grid // 2)  # Array of natural numbers
        self._construct_k_grid()

    def _construct_k_grid(self):
        # ---Construct the k-space grid---
        kx = np.fft.fftfreq(self.N_grid, d=self.H) * 2 * np.pi
        ky = np.fft.fftfreq(self.N_grid, d=self.H) * 2 * np.pi
        kz = np.fft.rfftfreq(self.N_grid, d=self.H) * 2 * np.pi

        self.kx_axis = kx
        self.ky_axis = ky
        self.kz_axis = kz

        self.kX, self.kY, self.kZ = np.meshgrid(kx, ky, kz, indexing="ij")
        self.k = np.sqrt(self.kX * self.kX + self.kY * self.kY + self.kZ * self.kZ)
        self.discrete_k = self.k_discretization()

    def k_discretization(self):
        k = np.zeros_like(self.n_k, dtype=float)
        k = self.n_k * self.k_F
        return k
