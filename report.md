# Technical Report

## Title
AI-Enhanced Next-Generation Digital Predistortion for Wideband Power Amplifiers

## Objective
Investigate whether a hybrid AI/ML model can improve the linearization of a nonlinear PA with memory effects while retaining interpretability and efficiency from conventional DPD.

## Signal
A synthetic OFDM-like waveform is generated from random 16-QAM symbols and transformed to time-domain complex baseband. The waveform has a high peak-to-average ratio, making nonlinear PA effects visible.

## PA model
The PA uses a complex-baseband memory-polynomial model:

y[n] = sum over memory taps and nonlinear orders of
a[m,p] * x[n-m] * |x[n-m]|^(p-1)

A soft limiting stage adds additional compression.

## Baseline DPD
The conventional model uses indirect learning:
1. send training data through the PA,
2. learn a post-distorter mapping PA output back to desired PA input,
3. reuse that inverse model as the predistorter.

The memory-polynomial coefficients are estimated using ridge-regularized least squares.

## Hybrid AI DPD
The hybrid model starts from the memory-polynomial inverse and adds a small neural residual correction. Features include current/delayed I/Q samples, magnitudes, and the conventional DPD output.

The goal is to let domain knowledge handle the main inverse behavior while ML learns the remaining mismatch.

## Evaluation
The project compares:
- No DPD
- Memory Polynomial DPD
- Hybrid AI DPD

Metrics:
- NMSE (dB)
- EVM (%)
- ACLR-like adjacent leakage (dB)
- model parameter count
- estimated model memory
- approximate MACs/sample

## Limitations
This implementation does not contain:
- Ericsson proprietary/measured data,
- actual 1 GHz RF waveform capture,
- DAC/ADC impairments,
- RF test-equipment calibration,
- online adaptation,
- FPGA/ASIC implementation,
- compliance-grade ACLR measurement.

Therefore the results are algorithmic simulation results only.

## Strong next steps
- train on measured PA input/output pairs,
- add generalized memory-polynomial and LUT baselines,
- test recurrent/temporal neural architectures,
- study quantization and pruning,
- test multiple output powers and temperatures,
- implement adaptive/online learning,
- profile FPGA-friendly models,
- measure latency, throughput, power, and real ACLR/EVM in hardware.
