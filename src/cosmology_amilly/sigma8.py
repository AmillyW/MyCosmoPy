import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad


def read_power_spectrum(file_path):
    data = np.loadtxt(file_path)
    k_vals = data[:, 0]
    P_vals = data[:, 1]
    return k_vals, P_vals


def build_Pk_interpolator(k1, P1):
    return interp1d(k1, P1)


def dimensionless_power_spectrum(k2, P2):
    triangle2 = k2**3 * P2 / (2 * np.pi * np.pi)
    return triangle2


def Window_Function_Fourier(k3, R3):
    W_R_k = 3 * (np.sin(k3 * R3) - k3 * R3 * np.cos(k3 * R3)) / (k3 * R3) ** 3
    return W_R_k


class Power_Spectrum:
    def __init__(self, z, file_path):
        self.z = z
        self.k_vals, self.P_vals = read_power_spectrum(file_path)
        self.Pk_interp = build_Pk_interpolator(self.k_vals, self.P_vals)
        self.dimensionless_Pk = dimensionless_power_spectrum(self.k_vals, self.P_vals)

    def sigma_R(self, R):
        self.R = R

        def integrand(k):
            W = Window_Function_Fourier(k, self.R)
            Pk = self.Pk_interp(k)
            Delta2 = dimensionless_power_spectrum(k, Pk)
            return W**2 * Delta2 / k

        result, _ = quad(integrand, 0, self.k_vals[-1])
        return np.sqrt(result)

    def sigma_8(self):
        return self.sigma_R(8)
