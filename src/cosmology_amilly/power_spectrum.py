import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad


def read_power_spectrum(file_path):
    data = np.loadtxt(file_path)
    k_vals = data[:, 0]
    P_vals = data[:, 1]
    return k_vals, P_vals


def build_log_k_grid(k_min, k_max, n_k=512):
    if n_k % 2 != 0:
        raise ValueError("n_k must be even for FAST-PT compatibility.")
    return np.logspace(np.log10(k_min), np.log10(k_max), n_k)


def build_Pk_interpolator(k1, P1):
    log_k1 = np.log(k1)
    log_P1 = np.log(P1)
    interp_func = interp1d(
        log_k1, log_P1, kind="cubic", bounds_error=False, fill_value="extrapolate"
    )

    def Pk_func(k):
        k = np.asarray(k)
        is_scalar = k.ndim == 0
        k_arr = np.atleast_1d(k)

        res = np.zeros_like(k_arr, dtype=float)
        mask = k_arr > 0
        res[mask] = np.exp(interp_func(np.log(k_arr[mask])))

        return res[0] if is_scalar else res

    return Pk_func


def dimensionless_power_spectrum(k2, P2):
    triangle2 = k2**3 * P2 / (2 * np.pi * np.pi)
    return triangle2


def Window_Function_Fourier(k3, R3):
    W_R_k = 3 * (np.sin(k3 * R3) - k3 * R3 * np.cos(k3 * R3)) / (k3 * R3) ** 3
    return W_R_k


class Power_Spectrum:
    def __init__(self, z, file_path, k_Log=None, n_k=512, use_log_grid=True):
        self.z = z
        self.k_raw, self.P_raw = read_power_spectrum(file_path)
        self.Pk_interp = build_Pk_interpolator(self.k_raw, self.P_raw)

        if k_Log is not None:
            self.k_vals = np.asarray(k_Log, dtype=float)
            if self.k_vals.size % 2 != 0:
                raise ValueError("k_Log must contain an even number of points.")
        elif use_log_grid:
            self.k_vals = build_log_k_grid(self.k_raw[0], self.k_raw[-1], n_k=n_k)
        else:
            self.k_vals = self.k_raw

        self.P_vals = self.Pk_interp(self.k_vals)
        self.dimensionless_Pk = dimensionless_power_spectrum(self.k_vals, self.P_vals)

    def sigma_R(self, R):
        self.R = R

        def integrand(k):
            W = Window_Function_Fourier(k, self.R)
            Pk = self.Pk_interp(k)
            Delta2 = dimensionless_power_spectrum(k, Pk)
            return W**2 * Delta2 / k

        result, _ = quad(integrand, self.k_raw[0], self.k_raw[-1])
        return np.sqrt(result)

    def sigma_8(self):
        return self.sigma_R(8)
