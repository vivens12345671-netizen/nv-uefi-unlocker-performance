# Experimental CUDA matrix tuning

An opt-in user-space cuBLASLt kernel tuner for the already-unlocked RTX 4070 SUPER. It compares FP16, BF16 and TF32 matrix operations against ordinary cuBLAS with numerical checks.

This branch starts from upstream v2.1.9 (08b7bc6). Its EFI unlock implementation is unchanged. Matrix-kernel gains do not establish faster ComfyUI, LLMs, training or gaming.

Build in a CUDA-compatible x64 MSVC Developer Command Prompt with tools\performance\build.cmd, then run python tools\performance\run.py.
