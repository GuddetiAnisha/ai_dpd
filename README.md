# AI-Enhanced Digital Predistortion for Wideband Power Amplifiers

A simulation/offline research prototype inspired by next-generation wideband DPD research.

## What this project does

This project simulates a complex-baseband wideband transmitter chain and compares:

1. No DPD
2. Memory Polynomial DPD — conventional baseline
3. Hybrid AI DPD — memory-polynomial initialization plus neural residual correction

The simulated PA includes nonlinear behavior and memory effects.

Evaluation includes:
- NMSE
- EVM
- ACLR-like adjacent-channel leakage / spectral regrowth
- parameter count
- approximate memory footprint
- approximate multiply-add cost per sample

> Important: This is a simulation/offline prototype. It does not claim validation on Ericsson data, 1 GHz RF hardware, or a production radio implementation.

## Structure

```text
ai_dpd_thesis_prototype/
├─ src/
│  ├─ signal_gen.py
│  ├─ pa_model.py
│  ├─ metrics.py
│  ├─ memory_polynomial.py
│  ├─ hybrid_dpd.py
│  └─ experiment.py
├─ tests/
│  └─ test_core.py
├─ outputs/
├─ run_experiment.py
├─ requirements.txt
├─ report.md
└─ README.md
```

## Setup

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python run_experiment.py
```

Results are saved under `outputs/`.

## Test

```bash
pytest -q
```

## Thesis context

The prototype works in normalized complex-baseband simulation. In a real 1 GHz+ RF study, these algorithms would be trained/evaluated with measured PA data at a defined sample rate and later validated on RF hardware.
