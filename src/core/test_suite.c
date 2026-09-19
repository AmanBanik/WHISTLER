#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "whistler_core.h"

// Helper to compute RMS
float compute_rms(const float *arr, int n) {
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) sum_sq += arr[i] * arr[i];
    return sqrtf(sum_sq / n);
}

// 1. FFT/IFFT Round Trip Validation
int test_fft_ifft_roundtrip() {
    printf("[TEST] FFT/IFFT Round Trip Validation\n");
    int n = 4096;
    float *x = (float*)SAFE_CALLOC(n, sizeof(float));
    float *re = (float*)SAFE_CALLOC(n, sizeof(float));
    float *im = (float*)SAFE_CALLOC(n, sizeof(float));
    
    for (int i = 0; i < n; i++) {
        x[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        re[i] = x[i];
    }
    
    fft(re, im, n);
    ifft(re, im, n);
    
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float err = fabsf(x[i] - re[i]);
        if (err > max_err) max_err = err;
    }
    
    free(x); free(re); free(im);
    printf("  Max Error: %e\n", max_err);
    if (max_err < 1e-4f) { printf("  -> PASSED\n\n"); return 0; }
    else { printf("  -> FAILED\n\n"); return 1; }
}

// 2. CPU vs CUDA Spectrogram Comparison
int test_spectrogram_equivalence() {
    printf("[TEST] CPU vs CUDA Spectrogram Equivalence\n");
#if !ENABLE_CUDA_TESTS
    printf("  -> SKIPPED (CUDA disabled in CPU CI)\n\n");
    return 0;
#else
    int n = 16384;
    float fs = 10000.0f;
    float *re_cpu = (float*)SAFE_CALLOC(n, sizeof(float));
    float *im_cpu = (float*)SAFE_CALLOC(n, sizeof(float));
    float *re_gpu = (float*)SAFE_CALLOC(n, sizeof(float));
    float *im_gpu = (float*)SAFE_CALLOC(n, sizeof(float));
    
    generate_damped_pulse(re_cpu, 200, fs, 1.0f, 500.0f, 2000.0f, 0.0f);
    for (int i = 0; i < n; i++) re_gpu[i] = re_cpu[i];
    
    fft(re_cpu, im_cpu, n);
    fft(re_gpu, im_gpu, n);
    
    apply_dispersion(re_cpu, im_cpu, n, fs, 0.1f, 15.0f);
    apply_dispersion_cuda(re_gpu, im_gpu, n, fs, 0.1f, 15.0f);
    
    ifft(re_cpu, im_cpu, n);
    ifft(re_gpu, im_gpu, n);
    
    int frames_cpu, frames_gpu;
    float *spec_cpu = compute_stft(re_cpu, n, 256, 64, &frames_cpu);
    float *spec_gpu = compute_stft(re_gpu, n, 256, 64, &frames_gpu);
    
    float max_err = 0.0f;
    float sum_sq_err = 0.0f;
    float sum_sq_ref = 0.0f;
    for (int i = 0; i < frames_cpu * 128; i++) {
        float err = fabsf(spec_cpu[i] - spec_gpu[i]);
        if (err > max_err) max_err = err;
        sum_sq_err += err * err;
        sum_sq_ref += spec_cpu[i] * spec_cpu[i];
    }
    float rms_err = sqrtf(sum_sq_err / (frames_cpu * 128));
    float rel_err = (sum_sq_ref > 0.0f) ? sqrtf(sum_sq_err / sum_sq_ref) : 0.0f;
    
    free(re_cpu); free(im_cpu); free(re_gpu); free(im_gpu);
    free(spec_cpu); free(spec_gpu);
    
    printf("  Spectrogram Max Error: %e\n", max_err);
    printf("  Spectrogram RMS Error: %e\n", rms_err);
    printf("  Spectrogram Rel Error: %e\n", rel_err);
    if (max_err < 1e-4f && rel_err < 1e-3f) { printf("  -> PASSED\n\n"); return 0; }
    else { printf("  -> FAILED\n\n"); return 1; }
#endif
}

// 3. Low-Frequency Stability (CPU & CUDA)
int test_low_freq_stability() {
    printf("[TEST] Low-Frequency Stability (DC and near-DC cutoff)\n");
    int n = 1024;
    float fs = 100.0f; // very low fs
    float *re_cpu = (float*)SAFE_CALLOC(n, sizeof(float));
    float *im_cpu = (float*)SAFE_CALLOC(n, sizeof(float));
    
    re_cpu[0] = 1.0f; re_cpu[1] = 1.0f;
    apply_dispersion(re_cpu, im_cpu, n, fs, 0.1f, 15.0f);
    
    int failed = 0;
    for (int i = 0; i < n; i++) {
        if (isnan(re_cpu[i]) || isinf(re_cpu[i])) failed = 1;
    }
    
#if ENABLE_CUDA_TESTS
    float *re_gpu = (float*)SAFE_CALLOC(n, sizeof(float));
    float *im_gpu = (float*)SAFE_CALLOC(n, sizeof(float));
    re_gpu[0] = 1.0f; re_gpu[1] = 1.0f;
    apply_dispersion_cuda(re_gpu, im_gpu, n, fs, 0.1f, 15.0f);
    for (int i = 0; i < n; i++) {
        if (isnan(re_gpu[i]) || isinf(re_gpu[i])) failed = 1;
    }
    free(re_gpu); free(im_gpu);
#endif

    free(re_cpu); free(im_cpu);
    if (!failed) { printf("  -> PASSED (No NaN/Inf detected)\n\n"); return 0; }
    else { printf("  -> FAILED (NaN/Inf detected)\n\n"); return 1; }
}

// 4. Deterministic Pipeline Regression (Multipath + AWGN)
#include <stdint.h>

// Simple portable LCG for deterministic cross-platform noise
static uint32_t lcg_state = 42;
static inline float lcg_randf() {
    lcg_state = lcg_state * 1664525 + 1013904223;
    return (float)lcg_state / (float)0xFFFFFFFF;
}

int test_pipeline_regression() {
    printf("[TEST] Deterministic Full-Pipeline Regression\n");
    int n = 4096;
    float fs = 10000.0f;
    float *re = (float*)SAFE_CALLOC(n, sizeof(float));
    float *im = (float*)SAFE_CALLOC(n, sizeof(float));
    
    // Generate base pulse
    generate_damped_pulse(re, 100, fs, 1.0f, 500.0f, 2000.0f, 0.0f);
    
    // Multipath (add a delayed echo)
    int delay = (int)(0.05f * fs); // 50ms delay
    for (int i = n - 1; i >= delay; i--) {
        re[i] += 0.5f * re[i - delay];
    }
    
    fft(re, im, n);
#if ENABLE_CUDA_TESTS
    apply_dispersion_cuda(re, im, n, fs, 0.1f, 15.0f);
#else
    apply_dispersion(re, im, n, fs, 0.1f, 15.0f);
#endif
    ifft(re, im, n);
    
    float *combined = (float*)SAFE_CALLOC(n, sizeof(float));
    for (int i = 0; i < n; i++) combined[i] = re[i];
    
    // Deterministic Box-Muller AWGN using portable LCG
    for (int i = 0; i < n; i += 2) {
        float u1 = lcg_randf();
        float u2 = lcg_randf();
        if (u1 < 1e-6f) u1 = 1e-6f;
        float mag = sqrtf(-2.0f * logf(u1));
        float z0 = mag * cosf(2.0f * (float)M_PI * u2);
        float z1 = mag * sinf(2.0f * (float)M_PI * u2);
        combined[i] += z0 * 0.05f;
        if (i + 1 < n) combined[i + 1] += z1 * 0.05f;
    }
    
    // STFT
    int frames;
    float *spec = compute_stft(combined, n, 256, 64, &frames);
    
    // Compute checksum (sum of log-scaled spectrogram)
    float sum_spec = 0.0f;
    for (int i = 0; i < frames * 128; i++) {
        sum_spec += spec[i];
    }
    
    printf("  Final Spectrogram Sum: %f\n", sum_spec);
    
    free(re); free(im); free(combined); free(spec);
    
    // Check against known reference (allowing minor floating-point divergence)
    float expected_sum = 3031.597656f; 
    if (!isnan(sum_spec) && fabsf(sum_spec - expected_sum) < 5.0f) { 
        printf("  -> PASSED\n\n"); 
        return 0; 
    }
    else { 
        printf("  -> FAILED\n\n"); 
        return 1; 
    }
}

// 5. FFT Size Sweep
int test_fft_size_sweep() {
    printf("[TEST] CUDA Dispersion Kernel Scaling vs FFT Size\n");
#if !ENABLE_CUDA_TESTS
    printf("  -> SKIPPED (CUDA disabled in CPU CI)\n\n");
    return 0;
#else
    int sizes[] = {1024, 2048, 4096, 8192, 16384};
    int iters = 10000;
    
    for (int i = 0; i < 5; i++) {
        int n = sizes[i];
        float *re = (float*)SAFE_CALLOC(n, sizeof(float));
        float *im = (float*)SAFE_CALLOC(n, sizeof(float));
        
        double time_taken = apply_dispersion_cuda_benchmark(re, im, n, 10000.0f, 0.1f, 15.0f, iters);
        
        printf("  Size: %5d | Time for %d iters: %.4f s\n", n, iters, time_taken);
        free(re); free(im);
    }
    printf("  -> PASSED\n\n");
    return 0;
#endif
}

int main() {
    printf("==========================================\n");
    printf("WHISTLER Automated Test Suite\n");
    printf("==========================================\n\n");
    
    int fails = 0;
    fails += test_fft_ifft_roundtrip();
    fails += test_spectrogram_equivalence();
    fails += test_low_freq_stability();
    fails += test_pipeline_regression();
    fails += test_fft_size_sweep();
    
    if (fails == 0) {
        printf("All automated C tests completed SUCCESSFULLY.\n");
        printf("==========================================\n");
        return 0;
    } else {
        printf("%d TEST(S) FAILED.\n", fails);
        printf("==========================================\n");
        return 1;
    }
}
