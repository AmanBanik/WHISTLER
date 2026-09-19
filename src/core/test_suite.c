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
void test_fft_ifft_roundtrip() {
    printf("[TEST] FFT/IFFT Round Trip Validation\n");
    int n = 4096;
    float *x = (float*)calloc(n, sizeof(float));
    float *re = (float*)calloc(n, sizeof(float));
    float *im = (float*)calloc(n, sizeof(float));
    
    // Generate random signal
    for (int i = 0; i < n; i++) {
        x[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        re[i] = x[i];
    }
    
    fft(re, im, n);
    ifft(re, im, n);
    
    float max_err = 0.0f;
    float sum_sq_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float err = fabsf(x[i] - re[i]);
        if (err > max_err) max_err = err;
        sum_sq_err += err * err;
    }
    float rms_err = sqrtf(sum_sq_err / n);
    
    printf("  Max Error: %e\n", max_err);
    printf("  RMS Error: %e\n", rms_err);
    if (max_err < 1e-4f) printf("  -> PASSED\n\n");
    else printf("  -> FAILED\n\n");
    
    free(x); free(re); free(im);
}

// 2. CPU vs CUDA Spectrogram Comparison
void test_spectrogram_equivalence() {
    printf("[TEST] CPU vs CUDA Spectrogram Equivalence\n");
    int n = 16384;
    float fs = 10000.0f;
    float *re_cpu = (float*)calloc(n, sizeof(float));
    float *im_cpu = (float*)calloc(n, sizeof(float));
    float *re_gpu = (float*)calloc(n, sizeof(float));
    float *im_gpu = (float*)calloc(n, sizeof(float));
    
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
    int total_bins = frames_cpu * 128;
    for (int i = 0; i < total_bins; i++) {
        float err = fabsf(spec_cpu[i] - spec_gpu[i]);
        if (err > max_err) max_err = err;
    }
    
    printf("  Spectrogram Max Error: %e\n", max_err);
    if (max_err < 1e-4f) printf("  -> PASSED\n\n");
    else printf("  -> FAILED\n\n");
    
    free(re_cpu); free(im_cpu); free(re_gpu); free(im_gpu);
    free(spec_cpu); free(spec_gpu);
}

// 3. Low-Frequency Stability
void test_low_freq_stability() {
    printf("[TEST] Low-Frequency Stability (DC and near-DC cutoff)\n");
    int n = 1024;
    float fs = 100.0f; // very low fs to hit low bins
    float *re = (float*)calloc(n, sizeof(float));
    float *im = (float*)calloc(n, sizeof(float));
    
    re[0] = 1.0f; // DC
    re[1] = 1.0f; // Near DC (0.097 Hz)
    
    apply_dispersion(re, im, n, fs, 0.1f, 15.0f);
    
    printf("  DC Output Magnitude: %f\n", sqrtf(re[0]*re[0] + im[0]*im[0]));
    printf("  Bin 1 (0.097Hz) Output Magnitude: %f\n", sqrtf(re[1]*re[1] + im[1]*im[1]));
    
    int nan_found = 0;
    for (int i = 0; i < n; i++) {
        if (isnan(re[i]) || isnan(im[i]) || isinf(re[i]) || isinf(im[i])) {
            nan_found = 1;
        }
    }
    if (!nan_found) printf("  -> PASSED (No NaN/Inf detected)\n\n");
    else printf("  -> FAILED (NaN/Inf detected)\n\n");
    
    free(re); free(im);
}

// 4. FFT Size Sweep
void test_fft_size_sweep() {
    printf("[TEST] FFT Size Scaling (Compute-Path benchmark)\n");
    int sizes[] = {1024, 2048, 4096, 8192, 16384};
    int num_sizes = 5;
    float fs = 10000.0f;
    int iters = 10000;
    
    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        float *re = (float*)calloc(n, sizeof(float));
        float *im = (float*)calloc(n, sizeof(float));
        
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        apply_dispersion_cuda_benchmark(re, im, n, fs, 0.1f, 15.0f, iters);
        clock_gettime(CLOCK_MONOTONIC, &end);
        
        double time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("  Size: %5d | Time for %d iters: %.4f s\n", n, iters, time);
        
        free(re); free(im);
    }
    printf("\n");
}

int main() {
    printf("==========================================\n");
    printf("WHISTLER Automated Test Suite\n");
    printf("==========================================\n\n");
    
    test_fft_ifft_roundtrip();
    test_spectrogram_equivalence();
    test_low_freq_stability();
    test_fft_size_sweep();
    
    printf("All automated C tests completed.\n");
    printf("==========================================\n");
    return 0;
}
