import numpy as np
import matplotlib.pyplot as plt

# ======================
# Parámetros del radar
# ======================
fc = 10e9                 # Frecuencia portadora (Hz)
c = 3e8                   # Velocidad de la luz
lam = c / fc              # Longitud de onda

fs = 10e3                 # Frecuencia de muestreo (Hz)
T = 0.1                   # Duración (s)
N = int(fs * T)
t = np.arange(N) / fs

# ======================
# Entradas del modelo
# ======================
SNR_dB = 10               # SNR en dB
v_mean = 30               # Velocidad promedio (m/s)
doppler_bw = 50           # Ancho espectral Doppler (Hz)

# ======================
# Doppler medio
# ======================
f_d = 2 * v_mean / lam    # Doppler (Hz)

# ======================
# Eco radar complejo
# ======================
phase = 2 * np.pi * f_d * t
signal = np.exp(1j * phase)

# Ensanchamiento espectral (dispersión Doppler)
doppler_noise = np.exp(
    1j * 2 * np.pi * np.random.normal(0, doppler_bw, N) * t
)

signal = signal * doppler_noise

# ======================
# Ajuste de SNR
# ======================
signal_power = np.mean(np.abs(signal)**2)
noise_power = signal_power / (10**(SNR_dB / 10))

noise = np.sqrt(noise_power/2) * (
    np.random.randn(N) + 1j*np.random.randn(N)
)

rx_signal = signal + noise

# ======================
# Visualización
# ======================
plt.figure()
plt.plot(t, np.real(rx_signal), label='I')
plt.plot(t, np.imag(rx_signal), label='Q')
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.title("Eco Radar I/Q Simulado")
plt.legend()
plt.grid()
plt.show()
