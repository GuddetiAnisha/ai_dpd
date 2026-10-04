# AI DPD — Final Validation Results

## Validation status

The project was validated locally after correcting the Python package layout so that the implementation lives under `src/` and the tests under `tests/`.

Final automated test result:

```text
5 passed in 0.73s
```

The end-to-end experiment was then executed successfully with:

```powershell
python run_experiment.py
```

The run completed and generated result artifacts under `outputs/`.

## Validated experiment results

| Method | NMSE (dB) | EVM (%) | ACLR (dB) | Parameters | Approx. Memory (Bytes) | Approx. MACs/sample |
|---|---:|---:|---:|---:|---:|---:|
| No DPD | -21.368033 | 8.542762 | -20.278952 | 0 | 0 | 0 |
| Memory Polynomial DPD | -33.781514 | 2.046088 | -19.879738 | 16 | 256 | 16 |
| Hybrid AI DPD | -31.080101 | 2.792511 | -19.837889 | 3234 | 13128 | 3136 |

## Interpretation

Both DPD approaches reduced in-band error compared with the no-DPD baseline in this experiment. The Memory Polynomial DPD achieved the best NMSE and EVM while using substantially fewer parameters, less memory, and fewer approximate MACs per sample than the Hybrid AI DPD.

The Hybrid AI DPD still improved NMSE and EVM over the no-DPD baseline, but did not outperform the Memory Polynomial DPD in this validated run.

ACLR did not improve in this run, so the results should not be interpreted as showing across-the-board spectral improvement.

## Generated artifacts

The validated local run produced:

```text
outputs/error_comparison.png
outputs/metrics.csv
outputs/metrics.json
outputs/spectrum_comparison.png
outputs/summary.txt
```

## Reproduce locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest -q
python run_experiment.py
```

Expected test result:

```text
5 passed
```

## Scope and limitations

These results validate the bundled synthetic thesis prototype and its included tests. They do not establish real-RF hardware performance, over-the-air performance, production deployment readiness, or superiority of AI-based DPD in general. The comparison applies to this implementation, configuration, data generation process, PA model, and experiment run.
