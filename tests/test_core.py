import numpy as np
from src.signal_gen import generate_ofdm
from src.pa_model import pa_with_soft_limit
from src.memory_polynomial import MemoryPolynomialDPD, design_matrix
from src.metrics import nmse_db, evm_percent, aclr_db

def test_ofdm_generation():
    x = generate_ofdm(n_symbols=8,n_fft=64,occupied=32,seed=1)
    assert np.iscomplexobj(x)
    assert len(x) == 512
    assert np.isfinite(x).all()

def test_pa_is_nonlinear():
    x = np.array([0.05+0j,0.8+0j],dtype=np.complex128)
    y = pa_with_soft_limit(x)
    g1 = abs(y[0])/abs(x[0])
    g2 = abs(y[1])/abs(x[1])
    assert abs(g1-g2) > 1e-3

def test_design_matrix_shape():
    x = np.ones(100,dtype=np.complex128)
    X = design_matrix(x,memory_depth=3,orders=(1,3,5))
    assert X.shape == (100,9)

def test_memory_polynomial_fit_predict():
    x = generate_ofdm(n_symbols=10,n_fft=64,occupied=32,seed=2)
    y = pa_with_soft_limit(x)
    model = MemoryPolynomialDPD(memory_depth=3,orders=(1,3,5))
    model.fit(y,x)
    pred = model.predict(y)
    assert pred.shape == x.shape
    assert np.isfinite(pred).all()

def test_metrics_are_finite():
    x = generate_ofdm(n_symbols=6,n_fft=64,occupied=32,seed=3)
    y = pa_with_soft_limit(x)
    assert np.isfinite(nmse_db(x,y))
    assert np.isfinite(evm_percent(x,y))
    assert np.isfinite(aclr_db(y))
