import numpy as np
import scipy
from scipy import signal
import matplotlib.pyplot as plt

rng = np.random.default_rng()

fs_Hz = 1e4
N = 1e5
amp = 2 * np.sqrt(2)
freq = 1234.0
noise_power = 0.001 * fs_Hz / 2
time = np.arange(N) / fs_Hz

x_V = amp * np.sin(2 * np.pi * freq * time)
x_V += rng.normal(scale=np.sqrt(noise_power), size=time.shape)

f_low = 1e1

nperseg = int(fs_Hz / f_low)
print(nperseg)

f, Pxx_den = scipy.signal.welch(
    x=x_V,
    fs=fs_Hz,
    nperseg=nperseg,
    scaling="density",
    average="mean",
)
f_med, Pxx_den_med = scipy.signal.welch(
    x=x_V,
    fs=fs_Hz,
    nperseg=nperseg,
    scaling="density",
    average="median",
)


plt.figure()
plt.plot(time[0:nperseg], x_V[0:nperseg])
plt.xlabel("time / s")
plt.ylabel("signal / V")
plt.savefig("welch_signal.jpg")
plt.close("all")

plt.figure()
plt.semilogy(f, Pxx_den, "b")
plt.semilogy(f_med, Pxx_den_med, "r--")
plt.ylim([0.5e-3, 1])
plt.xlabel("frequency / Hz")
plt.ylabel("power spectral density / V$^2$ Hz$^{-1}$")
plt.savefig("welch_spectrum.jpg")
plt.close("all")
