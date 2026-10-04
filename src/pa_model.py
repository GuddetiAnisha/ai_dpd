import numpy as np

DEFAULT_COEFFS = {
    (0, 1): 1.00 + 0.00j,
    (0, 3): -0.40 + 0.08j,
    (0, 5): 0.12 - 0.03j,
    (1, 1): 0.06 - 0.02j,
    (1, 3): -0.05 + 0.02j,
    (2, 1): 0.02 + 0.01j,
}

def _delay_same_length(x, m):
    if m == 0:
        return x
    if m >= len(x):
        return np.zeros_like(x)
    return np.concatenate([np.zeros(m, dtype=x.dtype), x[:-m]])

def memory_polynomial_pa(x, coeffs=None):
    coeffs = DEFAULT_COEFFS if coeffs is None else coeffs
    y = np.zeros_like(x, dtype=np.complex128)
    for (m, p), a in coeffs.items():
        xm = _delay_same_length(x, m)
        y += a * xm * (np.abs(xm) ** (p - 1))
    return y

def pa_with_soft_limit(x, limit=1.2):
    y = memory_polynomial_pa(x)
    mag = np.abs(y)
    phase = np.angle(y)
    compressed = limit * np.tanh(mag / max(limit, 1e-9))
    return compressed * np.exp(1j * phase)
