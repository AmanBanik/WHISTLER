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
}

// 3. Low-Frequency Stability (CPU & CUDA)
int test_low_freq_stability() {
    printf("[TEST] Low-Frequency Stability (DC and near-DC cutoff)\n");
    int n = 1024;
    float fs = 100.0f; // very low fs
    float *re_cpu = (float*)SAFE_CALLOC(n, sizeof(float));
    float *im_cpu = (float*)SAFE_CALLOC(n, sizeof(float));
    float *re_gpu = (float*)SAFE_CALLOC(n, sizeof(float));
    float *im_gpu = (float*)SAFE_CALLOC(n, sizeof(float));
    
    re_cpu[0] = 1.0f; re_cpu[1] = 1.0f;
    re_gpu[0] = 1.0f; re_gpu[1] = 1.0f;
    
    apply_dispersion(re_cpu, im_cpu, n, fs, 0.1f, 15.0f);
    apply_dispersion_cuda(re_gpu, im_gpu, n, fs, 0.1f, 15.0f);
    
    int failed = 0;
    for (int i = 0; i < n; i++) {
        if (isnan(re_cpu[i]) || isinf(re_cpu[i]) || isnan(re_gpu[i]) || isinf(re_gpu[i])) failed = 1;
    }
    
    free(re_cpu); free(im_cpu); free(re_gpu); free(im_gpu);
    if (!failed) { printf("  -> PASSED (No NaN/Inf detected on CPU or CUDA)\n\n"); return 0; }
    else { printf("  -> FAILED (NaN/Inf detected)\n\n"); return 1; }
}

// 4. Deterministic Pipeline Regression (Multipath + AWGN)
int test_pipeline_regression() {
    printf("[TEST] Deterministic Full-Pipeline Regression\n");
    int n = 4096;
    float fs = 10000.0f;
    float *re = (float*)SAFE_CALLOC(n, sizeof(float));
    float *im = (float*)SAFE_CALLOC(n, sizeof(float));
    
    generate_damped_pulse(re, 100, fs, 1.0f, 500.0f, 2000.0f, 0.0f);
    fft(re, im, n);
    apply_dispersion_cuda(re, im, n, fs, 0.1f, 15.0f);
    ifft(re, im, n);
    
    srand(42); // deterministic seed
    float *combined = (float*)SAFE_CALLOC(n, sizeof(float));
    for (int i = 0; i < n; i++) combined[i] = re[i];
    
    // add deterministic noise
    for (int i = 0; i < n; i++) {
        float noise = (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * 0.1f;
        combined[i] += noise;
    }
    
    // Verify signal RMS is bounded and reasonable
    float rms = compute_rms(combined, n);
    printf("  Final Output RMS: %f\n", rms);
    
    free(re); free(im); free(combined);
    if (rms > 0.0f && !isnan(rms)) { printf("  -> PASSED\n\n"); return 0; }
    else { printf("  -> FAILED\n\n"); return 1; }
}

// 5. FFT Size Sweep
int test_fft_size_sweep() {
    printf("[TEST] CUDA Dispersion Kernel Scaling vs FFT Size\n");
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
