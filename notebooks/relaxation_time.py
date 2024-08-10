"""This notebook fits a function to the relaxation time of GeSe and extrapolates it
to experimentally non-measured temperatures.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# %%
# GeSe relaxation times extracted from fig. 2 in
# https://pubs.acs.org/doi/abs/10.1021/acs.chemmater.6b01164
GeSe = [
    (299.131, 28.2573),
    (349.719, 20.9370),
    (399.924, 16.1142),
    (449.735, 13.0339),
    (500.116, 10.8378),
    (550.153, 8.48201),
    (600.281, 6.57917),
    (651.861, 4.83469),
    (650.719, 4.33288),
    (700.109, 3.90518),
    (750.905, 3.67718),
]
GeSe = pd.DataFrame(GeSe, columns=["temp", "tau"])
GeSe.tau = GeSe.tau * 1e-15


# %%
def GeSe_tau_decay(temp, a, b, c):
    # 1/x + linear fit.
    return a / temp + b * temp + c


popt, pcov = curve_fit(GeSe_tau_decay, GeSe.temp, GeSe.tau)


# %%
plt.plot(np.arange(100, 1000, 50), GeSe_tau_decay(np.arange(100, 1000, 50), *popt))
plt.scatter(*GeSe.values.T)


# %%
# Generate dataframe with GeSe relaxation times extrapolated from 100 to 1000 K.
temps = np.linspace(100, 1000, 10, dtype=int)
GeSe_extra = pd.DataFrame(
    [temps, GeSe_tau_decay(temps, *popt)], index=["temp", "tau"]
).T
