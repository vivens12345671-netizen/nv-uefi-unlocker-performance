// SPDX-License-Identifier: GPL-2.0-only
// Opt-in user-space kernel selection. Does not write GPU clocks or registers.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_OK(call) do { const auto status_ = (call); if (status_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error at line %d: %s\n", __LINE__, cudaGetErrorString(status_)); exit(2); } } while (0)
#define BLAS_OK(call) do { const auto status_ = (call); if (status_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error at line %d: %d\n", __LINE__, int(status_)); exit(3); } } while (0)

__host__ __device__ float input(unsigned index, unsigned seed) {
    unsigned x = index + seed;
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return float(x & 65535) / 65536.0f - 0.5f;
}
template<class T> __global__ void initialize(T* a, T* b, size_t count) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) { a[i] = T(input(unsigned(i), 123)); b[i] = T(input(unsigned(i), 987)); }
}
struct Comparison { unsigned mismatches, nonfinite, max_absolute_bits; };
__global__ void compare(const float* reference, const float* actual, size_t count, Comparison* result) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    if (!isfinite(reference[i]) || !isfinite(actual[i])) { atomicAdd(&result->nonfinite, 1u); return; }
    const float error = fabsf(reference[i] - actual[i]);
    atomicMax(&result->max_absolute_bits, __float_as_uint(error));
    if (error > 0.0002f * (1.0f + fabsf(reference[i]))) atomicAdd(&result->mismatches, 1u);
}
template<class Run> double milliseconds(Run run, int repetitions) {
    cudaEvent_t begin, end;
    CUDA_OK(cudaEventCreate(&begin)); CUDA_OK(cudaEventCreate(&end));
    CUDA_OK(cudaEventRecord(begin));
    for (int i = 0; i < repetitions; ++i) run();
    CUDA_OK(cudaEventRecord(end)); CUDA_OK(cudaEventSynchronize(end));
    float elapsed; CUDA_OK(cudaEventElapsedTime(&elapsed, begin, end));
    CUDA_OK(cudaEventDestroy(begin)); CUDA_OK(cudaEventDestroy(end));
    return elapsed / repetitions;
}
double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end()); return samples[samples.size() / 2];
}
// These finite inputs are in [-0.5, 0.5] on a 2^-16 grid, so their TF32
// round-to-nearest conversion equals a round-to-nearest half conversion.
template<class T> float quantized(float value) { return float(T(value)); }
template<> float quantized<float>(float value) { return __half2float(__float2half_rn(value)); }

template<class T> void tune(cublasHandle_t baseline_handle, cublasLtHandle_t lt,
                          const char* mode, int n, cudaDataType_t type, cublasComputeType_t compute) {
    const size_t count = size_t(n) * n, workspace_bytes = 64ull << 20;
    T *a, *b; float *baseline, *candidate; void* workspace; Comparison* device_comparison;
    CUDA_OK(cudaMalloc(&a, count * sizeof(T))); CUDA_OK(cudaMalloc(&b, count * sizeof(T)));
    CUDA_OK(cudaMalloc(&baseline, count * sizeof(float))); CUDA_OK(cudaMalloc(&candidate, count * sizeof(float)));
    CUDA_OK(cudaMalloc(&workspace, workspace_bytes)); CUDA_OK(cudaMalloc(&device_comparison, sizeof(Comparison)));
    initialize<<<unsigned((count + 255) / 256), 256>>>(a, b, count); CUDA_OK(cudaDeviceSynchronize());
    cublasLtMatmulDesc_t operation; cublasLtMatrixLayout_t ab_layout, cd_layout; cublasLtMatmulPreference_t preference;
    BLAS_OK(cublasLtMatmulDescCreate(&operation, compute, CUDA_R_32F));
    BLAS_OK(cublasLtMatrixLayoutCreate(&ab_layout, type, n, n, n));
    BLAS_OK(cublasLtMatrixLayoutCreate(&cd_layout, CUDA_R_32F, n, n, n));
    BLAS_OK(cublasLtMatmulPreferenceCreate(&preference));
    BLAS_OK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                               &workspace_bytes, sizeof(workspace_bytes)));
    std::vector<cublasLtMatmulHeuristicResult_t> choices(64); int returned = 0;
    BLAS_OK(cublasLtMatmulAlgoGetHeuristic(lt, operation, ab_layout, ab_layout, cd_layout, cd_layout,
                                          preference, int(choices.size()), choices.data(), &returned));
    const float alpha = 1, beta = 0;
    auto run_baseline = [&]() {
        BLAS_OK(cublasGemmEx(baseline_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                            a, type, n, b, type, n, &beta, baseline, CUDA_R_32F, n, compute, CUBLAS_GEMM_DEFAULT));
    };
    auto candidate_status = [&](int i) {
        return cublasLtMatmul(lt, operation, &alpha, a, ab_layout, b, ab_layout, &beta,
                              candidate, cd_layout, candidate, cd_layout, &choices[i].algo,
                              workspace, workspace_bytes, nullptr);
    };
    // Search measurements select a candidate; none are used as the final score.
    double search_best = 1e30; int best = -1, valid = 0;
    for (int i = 0; i < returned; ++i) {
        if (choices[i].state != CUBLAS_STATUS_SUCCESS) continue;
        if (candidate_status(i) != CUBLAS_STATUS_SUCCESS) continue;
        CUDA_OK(cudaDeviceSynchronize());
        const double time = milliseconds([&]() { BLAS_OK(candidate_status(i)); }, 5);
        ++valid;
        if (time < search_best) { search_best = time; best = i; }
    }
    if (best < 0) { fprintf(stderr, "No supported algorithms for %s/%d\n", mode, n); exit(4); }
    auto run_candidate = [&]() { BLAS_OK(candidate_status(best)); };
    for (int i = 0; i < 5; ++i) { run_baseline(); run_candidate(); }
    CUDA_OK(cudaDeviceSynchronize());
    // Alternating AB/BA order reduces bias from clocks and thermal drift.
    std::vector<double> base_times, tuned_times, paired_speedups;
    const int repetitions = n == 4096 ? 100 : 20;
    for (int trial = 0; trial < 9; ++trial) {
        double base_ms, tuned_ms;
        if (trial % 2 == 0) {
            base_ms = milliseconds(run_baseline, repetitions); tuned_ms = milliseconds(run_candidate, repetitions);
        } else {
            tuned_ms = milliseconds(run_candidate, repetitions); base_ms = milliseconds(run_baseline, repetitions);
        }
        base_times.push_back(base_ms); tuned_times.push_back(tuned_ms); paired_speedups.push_back(base_ms / tuned_ms);
    }
    // Check EVERY output against the same-precision cuBLAS baseline, and
    // independently sample 16 dot products using double precision on the CPU.
    CUDA_OK(cudaMemset(device_comparison, 0, sizeof(Comparison)));
    compare<<<unsigned((count + 255) / 256), 256>>>(baseline, candidate, count, device_comparison);
    Comparison comparison; CUDA_OK(cudaMemcpy(&comparison, device_comparison, sizeof(comparison), cudaMemcpyDeviceToHost));
    double sample_max_error = 0;
    for (int t = 0; t < 16; ++t) {
        const int row = (t * 193 + 17) % n, col = (t * 317 + 29) % n;
        float actual; CUDA_OK(cudaMemcpy(&actual, candidate + row + size_t(col) * n, sizeof(float), cudaMemcpyDeviceToHost));
        double reference = 0;
        for (int k = 0; k < n; ++k) reference += double(quantized<T>(input(row + k * n, 123))) * quantized<T>(input(k + col * n, 987));
        const double error = fabs(actual - reference); sample_max_error = std::max(sample_max_error, error);
        if (!std::isfinite(actual) || error > 0.002 * (1 + fabs(reference))) {
            fprintf(stderr, "CPU-reference check failed for %s/%d\n", mode, n); exit(5);
        }
    }
    if (comparison.mismatches || comparison.nonfinite) {
        fprintf(stderr, "Full-output check failed for %s/%d: %u mismatches, %u nonfinite\n",
                mode, n, comparison.mismatches, comparison.nonfinite); exit(6);
    }
    int algorithm_id = -1; size_t written = 0;
    BLAS_OK(cublasLtMatmulAlgoConfigGetAttribute(&choices[best].algo, CUBLASLT_ALGO_CONFIG_ID,
                                               &algorithm_id, sizeof(algorithm_id), &written));
    const double base_ms = median(base_times), tuned_ms = median(tuned_times), paired = median(paired_speedups);
    const int improved = int(std::count_if(paired_speedups.begin(), paired_speedups.end(), [](double x) { return x > 1; }));
    const bool recommend_tuned = base_ms / tuned_ms > 1.03 && paired > 1.03 && improved >= 7;
    printf("{\"kind\":\"comparison\",\"mode\":\"%s\",\"n\":%d,\"valid_candidates\":%d,\"algorithm_id\":%d,"
           "\"workspace_bytes\":%zu,\"baseline_ms\":%.6f,\"tuned_ms\":%.6f,\"baseline_tflops\":%.4f,"
           "\"tuned_tflops\":%.4f,\"paired_median_speedup\":%.6f,\"improved_pairs\":%d,\"pairs\":9,"
           "\"repetitions_per_batch\":%d,\"recommended_backend\":\"%s\","
           "\"compared_outputs\":%zu,\"mismatches\":%u,\"nonfinite\":%u,\"sample_max_abs_error\":%.9g}\n",
           mode, n, valid, algorithm_id, choices[best].workspaceSize, base_ms, tuned_ms,
           2.0 * n * n * n / (base_ms * 1e9), 2.0 * n * n * n / (tuned_ms * 1e9), paired,
           improved, repetitions, recommend_tuned ? "cublasLt" : "cublas",
           count, comparison.mismatches, comparison.nonfinite, sample_max_error);
    fflush(stdout);
    BLAS_OK(cublasLtMatmulPreferenceDestroy(preference)); BLAS_OK(cublasLtMatrixLayoutDestroy(ab_layout));
    BLAS_OK(cublasLtMatrixLayoutDestroy(cd_layout)); BLAS_OK(cublasLtMatmulDescDestroy(operation));
    CUDA_OK(cudaFree(a)); CUDA_OK(cudaFree(b)); CUDA_OK(cudaFree(baseline)); CUDA_OK(cudaFree(candidate));
    CUDA_OK(cudaFree(workspace)); CUDA_OK(cudaFree(device_comparison));
}
int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--help") == 0) {
        printf("Usage: matmul-tune.exe [4096|8192]\n"); return 0;
    }
    if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "4096") != 0 && std::strcmp(argv[1], "8192") != 0)) {
        fprintf(stderr, "Usage: matmul-tune.exe [4096|8192]\n"); return 1;
    }
    cudaDeviceProp device; CUDA_OK(cudaGetDeviceProperties(&device, 0));
    cublasHandle_t base; cublasLtHandle_t lt; BLAS_OK(cublasCreate(&base)); BLAS_OK(cublasLtCreate(&lt));
    BLAS_OK(cublasSetMathMode(base, cublasMath_t(CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
    int cublas_version; BLAS_OK(cublasGetVersion(base, &cublas_version));
    printf("{\"kind\":\"environment\",\"device\":\"%s\",\"sms\":%d,\"cublas_version\":%d,\"workspace_limit_mib\":64}\n",
           device.name, device.multiProcessorCount, cublas_version); fflush(stdout);
    const std::vector<int> sizes = argc == 2 ? std::vector<int>{std::atoi(argv[1])} : std::vector<int>{4096, 8192};
    for (int n : sizes) {
        tune<__half>(base, lt, "FP16_FP32_accumulate", n, CUDA_R_16F, CUBLAS_COMPUTE_32F);
        tune<__nv_bfloat16>(base, lt, "BF16_FP32_accumulate", n, CUDA_R_16BF, CUBLAS_COMPUTE_32F);
        tune<float>(base, lt, "TF32_FP32_accumulate", n, CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_TF32);
    }
    BLAS_OK(cublasLtDestroy(lt)); BLAS_OK(cublasDestroy(base)); return 0;
}
