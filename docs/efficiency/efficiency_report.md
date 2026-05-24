# Efficiency Evaluation of the HPO-Optimized Architectures

## What was done

**Metrics and method**
- **FLOPs (M):** floating-point operations for one 224×224 image = 2 × MACs, where MACs are
  measured with `thop` (its parameter count was cross-checked against the direct count for
  every model). The *FLOPs ratio* is relative to the reference (lower = more efficient).
- **Model size (MB):** in-memory footprint of parameters + buffers.
- **Peak memory (MB):** peak CUDA memory during a forward pass (steady state).
- **GPU/CPU time and throughput:** mean (± std) forward latency over many warmed-up, timed
  passes (CUDA events on GPU, `perf_counter` on CPU; `cudnn.benchmark` enabled) and the
  corresponding throughput (images/second).

**Batch size.** Latency/throughput/memory were measured at batch sizes 128
**Hardware / software:** NVIDIA GeForce RTX 4070 Laptop GPU, CPU x86_64, PyTorch 2.5.1, input
(3, 224, 224).

## Efficiency table (batch 128)

| Config | Acc (%) | FLOPs (M) | FLOPs ratio | Model size (MB) | Peak mem (MB) | GPU time (ms) | GPU thr. (img/s) | CPU time (ms) | CPU thr. (img/s) |
|---|---|---|---|---|---|---|---|---|---|
| Reference | — | 226.6 | 1.00× | 5.88 | 333.1 | 28.51±0.67 | 4490 | 509.62±32.63 | 251 |
| 5 blocks | 83.00 | 171.1 | 0.76× | 2.77 | 329.4 | 24.04±0.40 | 5324 | 384.21±26.97 | 333 |
| 7 blocks | 85.78 | 190.4 | 0.84× | 2.99 | 329.7 | 26.46±0.73 | 4838 | 478.49±28.67 | 268 |
| 9 blocks | 89.17 | 213.8 | 0.94× | 4.08 | 330.8 | 25.85±0.32 | 4951 | 410.83±16.93 | 312 |