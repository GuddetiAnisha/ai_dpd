import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .signal_gen import generate_ofdm
from .pa_model import pa_with_soft_limit
from .memory_polynomial import MemoryPolynomialDPD
from .hybrid_dpd import HybridResidualDPD
from .metrics import nmse_db, evm_percent, aclr_db

def rms(x):
    return np.sqrt(np.mean(np.abs(x)**2) + 1e-12)

def normalize_drive(x, target_rms):
    return x / rms(x) * target_rms

def evaluate(reference, observed):
    return {
        "NMSE_dB": nmse_db(reference, observed),
        "EVM_percent": evm_percent(reference, observed),
        "ACLR_dB": aclr_db(observed),
    }

def save_psd_plot(signals, out_path):
    plt.figure(figsize=(10,6))
    nfft = 8192
    for label, sig in signals.items():
        s = sig[:nfft]
        if len(s) < nfft:
            s = np.pad(s, (0, nfft-len(s)))
        S = np.fft.fftshift(np.fft.fft(s*np.hanning(nfft)))
        p = 20*np.log10(np.abs(S)/(np.max(np.abs(S))+1e-12)+1e-12)
        f = np.linspace(-0.5,0.5,nfft,endpoint=False)
        plt.plot(f,p,label=label,linewidth=1)
    plt.xlabel("Normalized frequency")
    plt.ylabel("Normalized magnitude (dB)")
    plt.title("PA Output Spectrum")
    plt.grid(True,alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path,dpi=160)
    plt.close()

def save_error_plot(reference, outputs, out_path):
    plt.figure(figsize=(10,5))
    n = min(2500, len(reference))
    for label, sig in outputs.items():
        err = np.abs(sig[:n]-reference[:n])
        plt.plot(err,label=label,linewidth=0.8)
    plt.xlabel("Sample")
    plt.ylabel("Absolute complex error")
    plt.title("Output Error Comparison")
    plt.grid(True,alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path,dpi=160)
    plt.close()

def run(output_dir="outputs", seed=7):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    x = generate_ofdm(n_symbols=220, n_fft=256, occupied=160, seed=seed, rms=0.28)
    split = int(0.6*len(x))
    x_train, x_test = x[:split], x[split:]

    y_train = pa_with_soft_limit(x_train)
    y_no = pa_with_soft_limit(x_test)

    mp = MemoryPolynomialDPD(memory_depth=4, orders=(1,3,5,7), ridge=1e-5)
    mp.fit(y_train, x_train)

    target_rms = rms(x_test)
    x_mp = mp.predict(x_test)
    y_mp = pa_with_soft_limit(x_mp)

    hybrid = HybridResidualDPD(mp, memory_depth=4, hidden=(48,48), epochs=8,
                               batch_size=1024, seed=seed, max_train_samples=30000,
                               residual_scale=0.35)
    hybrid.fit(y_train, x_train)
    x_h = hybrid.predict(x_test)
    y_h = pa_with_soft_limit(x_h)

    rows = []
    rows.append({"Method":"No DPD", **evaluate(x_test,y_no), "Parameters":0,
                 "Approx_Memory_Bytes":0, "Approx_MACs_per_sample":0})
    rows.append({"Method":"Memory Polynomial DPD", **evaluate(x_test,y_mp),
                 "Parameters":mp.parameter_count,
                 "Approx_Memory_Bytes":mp.approx_memory_bytes(),
                 "Approx_MACs_per_sample":mp.approx_macs_per_sample()})
    rows.append({"Method":"Hybrid AI DPD", **evaluate(x_test,y_h),
                 "Parameters":hybrid.parameter_count,
                 "Approx_Memory_Bytes":hybrid.approx_memory_bytes(),
                 "Approx_MACs_per_sample":hybrid.approx_macs_per_sample()})

    df = pd.DataFrame(rows)
    df.to_csv(out/"metrics.csv", index=False)
    (out/"metrics.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")

    save_psd_plot({"No DPD":y_no,"Memory Polynomial":y_mp,"Hybrid AI":y_h},
                  out/"spectrum_comparison.png")
    save_error_plot(x_test,{"No DPD":y_no,"Memory Polynomial":y_mp,"Hybrid AI":y_h},
                    out/"error_comparison.png")

    best_nmse = df.loc[df["NMSE_dB"].idxmin(),"Method"]
    best_evm = df.loc[df["EVM_percent"].idxmin(),"Method"]
    best_aclr = df.loc[df["ACLR_dB"].idxmin(),"Method"]

    summary = "AI-Enhanced DPD Prototype Results\n\n" + df.to_string(index=False)
    summary += f"\n\nBest NMSE: {best_nmse}\nBest EVM: {best_evm}\nBest ACLR-like metric: {best_aclr}"
    summary += "\n\nLower NMSE/EVM is better; more negative ACLR-like metric is better."
    summary += "\nSimulation/offline prototype only."
    (out/"summary.txt").write_text(summary,encoding="utf-8")

    return df
