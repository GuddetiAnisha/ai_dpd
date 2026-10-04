import numpy as np

def align_reference(reference, observed):
    denom = np.vdot(reference, reference) + 1e-18
    gain = np.vdot(reference, observed) / denom
    return gain * reference, gain

def nmse_db(reference, observed):
    ref, _ = align_reference(reference, observed)
    err = observed - ref
    return float(10*np.log10((np.sum(np.abs(err)**2)+1e-18)/(np.sum(np.abs(ref)**2)+1e-18)))

def evm_percent(reference, observed):
    ref, _ = align_reference(reference, observed)
    err = observed - ref
    return float(100*np.sqrt((np.sum(np.abs(err)**2)+1e-18)/(np.sum(np.abs(ref)**2)+1e-18)))

def aclr_db(signal, occupied_fraction=0.625, nfft=8192):
    n = min(len(signal), nfft)
    s = signal[:n]
    if n < nfft:
        s = np.pad(s, (0, nfft-n))
    S = np.fft.fftshift(np.fft.fft(s*np.hanning(nfft)))
    psd = np.abs(S)**2
    f = np.linspace(-0.5, 0.5, nfft, endpoint=False)
    half = occupied_fraction/2
    main = np.abs(f) <= half
    adj = (np.abs(f) > half)
    pmain = np.sum(psd[main]) + 1e-18
    padj = np.sum(psd[adj]) + 1e-18
    return float(10*np.log10(padj/pmain))
