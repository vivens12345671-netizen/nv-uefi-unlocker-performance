# RTX 4070 SUPER kernel-selection experiment

Tests compare ordinary cuBLAS and empirically selected cuBLASLt kernels on the same already-unlocked 220 W RTX 4070 SUPER, with CUDA/cuBLAS 12.6 and FP32 accumulation.

Two longer confirmation runs found improvements for some 4096 square FP16/BF16 operations. The 8192 square cases showed no convincing improvement. All selected kernels passed full-output comparisons and independent numerical spot checks.

These are user-space matrix-kernel results, not evidence of faster EFI firmware or application-wide acceleration. Application integration and end-to-end testing remain necessary.
