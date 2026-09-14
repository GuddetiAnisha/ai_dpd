import numpy as np

def qam16_symbols(n, rng):
    levels = np.array([-3, -1, 1, 3], dtype=float)
    i = rng.choice(levels, size=n)
    q = rng.choice(levels, size=n)
    x = i + 1j * q
    return x / np.sqrt(np.mean(np.abs(x) ** 2))

def generate_ofdm(n_symbols=300, n_fft=256, occupied=160, seed=7, rms=0.25):
    if occupied >= n_fft or occupied % 2 != 0:
        raise ValueError("occupied must be even and smaller than n_fft")
    rng = np.random.default_rng(seed)
    half = occupied // 2
    out = []
    for _ in range(n_symbols):
        freq = np.zeros(n_fft, dtype=np.complex128)
        data = qam16_symbols(occupied, rng)
        freq[1:half+1] = data[:half]
        freq[-half:] = data[half:]
        td = np.fft.ifft(freq) * np.sqrt(n_fft)
        out.append(td)
    x = np.concatenate(out)
    x = x / np.sqrt(np.mean(np.abs(x) ** 2) + 1e-12) * rms
    return x.astype(np.complex128)
