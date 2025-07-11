import numpy as np
from scipy.integrate import quad


class Linear_Growth:
    def __init__(self, z, Omega_m, Omega_r, Omega_Lambda, Omega_K):
        self.z = z
        self.a = 1 / (1 + self.z)
        self.Omega_m = Omega_m
        self.Omega_r = Omega_r
        self.Omega_Lambda = Omega_Lambda
        self.Omega_K = Omega_K

    def Friedmann_E(self, a1):  # E = H / H_0
        E = np.sqrt(
            self.Omega_m / a1**3
            + self.Omega_r / a1**4
            + self.Omega_K / a1**2
            + self.Omega_Lambda
        )
        return E

    def integral_function(self, a3):
        def integrand(a2):
            return 1 / (self.Friedmann_E(a2) * a2) ** 3

        result, _ = quad(integrand, 0, a3)
        return result

    def Growing_mode(self, a4):
        return self.Friedmann_E(a4) * self.integral_function(a4)

    def Growth_factor(self):
        return self.Growing_mode(self.a) / self.Growing_mode(1)

    def Growth_rate(self):
        f = (
            -3 * self.Omega_m / self.a**3
            - 4 * self.Omega_r / self.a**4
            - 2 * self.Omega_K / self.a**2
        ) / (2 * self.Friedmann_E(self.a) ** 2) + 1 / (
            self.a**2 * self.Friedmann_E(self.a) ** 3 * self.integral_function(self.a)
        )
        return f

    def Sound_Horizon(self, H0, a7):
        def sound_speed_radiation(a5):
            C_s = 1 / (np.sqrt(3 * (a5 * (3 * self.Omega_m) / (4 * self.Omega_r) + 1)))
            return C_s

        def integrand_SH(a6):
            return sound_speed_radiation(a6) / (a6 * a6 * H0 * self.Friedmann_E(a6))

        SH, _ = quad(integrand_SH, 0, a7)
        return SH


def Linear_Growth_Factor(z, Omega_m, Omega_r, Omega_Lambda, Omega_K):
    vectorized1 = np.vectorize(
        lambda z_: Linear_Growth(
            z_, Omega_m, Omega_r, Omega_Lambda, Omega_K
        ).Growth_factor()
    )
    return vectorized1(z)


def Linear_Growth_Rate(z, Omega_m, Omega_r, Omega_Lambda, Omega_K):
    vectorized2 = np.vectorize(
        lambda z_: Linear_Growth(
            z_, Omega_m, Omega_r, Omega_Lambda, Omega_K
        ).Growth_rate()
    )
    return vectorized2(z)
