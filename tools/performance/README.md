# Experimental CUDA matrix tuning

This optional tool searches cuBLASLt kernels for dense FP16/BF16/TF32 matrix multiplication with FP32 accumulation. It compares the selected kernel with the ordinary cuBLAS path on the same unlocked GPU. It is a user-space experiment, not an additional GPU unlock or an application-wide accelerator.

The v2.1.9 EFI transaction already writes the full-rate selectors. There is no faster selector value to apply. This fork adds no firmware writes, overclock, power-limit change, replacement driver, automatic boot entry, or VBIOS modification. Its EFI source is unchanged from upstream v2.1.9.

## Build and run

Requires Windows, Python 3.9+, the CUDA Toolkit with cuBLAS/cuBLASLt, and a CUDA-compatible x64 MSVC Developer Command Prompt. The tested combination is CUDA/cuBLAS 12.6 and MSVC 14.29. The build targets Ada `sm_89`.

From the repository root in that command prompt:

```bat
tools\performance\build.cmd
python tools\performance\run.py
```

Run only one size with `python tools\performance\run.py --size 8192`. The runner samples temperature, power, clocks and utilization; it terminates the child at 85 C or 120 seconds, or if telemetry becomes unavailable. These are sampled limits, not hardware safety guarantees. It does not change hardware settings. Close other GPU workloads for useful comparisons.

JSON results and local machine telemetry are saved under `tools/performance/results/` (ignored by Git). The executable can be invoked directly, but only the Python runner provides those execution limits.

## Measurement and correctness

- Deterministic, nonzero dense inputs; FP32 output and accumulation. TF32 is explicitly a reduced-input-precision mode, reported separately.
- Up to 64 cuBLASLt heuristic candidates, capped at 64 MiB workspace. Unsupported candidates are skipped. Actual CUDA execution errors abort.
- Candidate search timings are discarded. Nine new alternating baseline/candidate and candidate/baseline measurement pairs provide the score. Each batch contains 100 GEMMs at size 4096 or 20 at size 8192; CUDA events measure elapsed GPU-stream time, including any stream gaps.
- Every output is compared with the same-precision cuBLAS baseline at `abs(error) <= 0.0002 * (1 + abs(reference))`. Nonfinite outputs reject the result. Sixteen independent dot products are also checked against CPU double-precision accumulation of the quantized inputs.
- A candidate is recommended only if both the ratio of median times and the median paired speedup exceed 3%, with improvements in at least seven of nine pairs. Otherwise `recommended_backend` remains `cublas`. This is a conservative screening rule, not a statistical confidence interval.

Algorithm IDs are diagnostic, not portable profiles. A real application must integrate and cache selection for its exact dimensions, layouts, precision, library version and device. This tool does not inject kernels into ComfyUI, Ollama, PyTorch, or another process. A benchmark gain does not establish an application gain.

## Measured scope

See [the hardware experiment](../../docs/evidence/rtx4070-super-performance-confirmation/README.md). On the tested 220 W RTX 4070 SUPER, smaller FP16/BF16 GEMMs sometimes benefited. Large 8192 GEMMs did not show a convincing improvement. No claim is made for faster image generation, LLMs, training, or gaming.

The ordinary upstream USB unlock instructions still apply. There is no new EFI binary to install for this experiment.
