import numpy as np

def _delay(x, m):
    if m == 0:
        return x
    return np.concatenate([np.zeros(m, dtype=x.dtype), x[:-m]])

def design_matrix(x, memory_depth=4, orders=(1,3,5,7)):
    cols = []
    for m in range(memory_depth):
        xm = _delay(x, m)
        for p in orders:
            cols.append(xm * (np.abs(xm) ** (p - 1)))
    return np.column_stack(cols)

class MemoryPolynomialDPD:
    def __init__(self, memory_depth=4, orders=(1,3,5,7), ridge=1e-6):
        self.memory_depth = memory_depth
        self.orders = tuple(orders)
        self.ridge = ridge
        self.coef_ = None

    @property
    def parameter_count(self):
        return self.memory_depth * len(self.orders)

    def fit(self, pa_output, desired_input):
        X = design_matrix(pa_output, self.memory_depth, self.orders)
        A = X.conj().T @ X + self.ridge * np.eye(X.shape[1], dtype=np.complex128)
        b = X.conj().T @ desired_input
        self.coef_ = np.linalg.solve(A, b)
        return self

    def predict(self, x):
        if self.coef_ is None:
            raise RuntimeError("Model not fitted")
        return design_matrix(x, self.memory_depth, self.orders) @ self.coef_

    def approx_macs_per_sample(self):
        return self.parameter_count

    def approx_memory_bytes(self):
        return self.parameter_count * 16
