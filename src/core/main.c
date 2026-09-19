#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <stdint.h>
#include "whistler_core.h"
#include "io.h"

int main() {
    float fs = 10000.0f; 
    int nx = 200; 
    float *x = (float*)SAFE_MALLOC(nx * sizeof(float));
    
    // Generate pulse
    generate_damped_pulse(x, nx, fs, 1.0f, 500.0f, 2000.0f, 0.0f);
    
    int n_fft = 16384; 
    
    // ============================================
    // CPU vs GPU NUMERICAL VALIDATION
    // ============================================
    float *val_re_cpu = (float*)SAFE_CALLOC(n_fft, sizeof(float));
    float *val_im_cpu = (float*)SAFE_CALLOC(n_fft, sizeof(float));
    float *val_re_gpu = (float*)SAFE_CALLOC(n_fft, sizeof(float));
    float *val_im_gpu = (float*)SAFE_CALLOC(n_fft, sizeof(float));
    
    for (int i = 0; i < nx; i++) {
        val_re_cpu[i] = x[i];
        val_re_gpu[i] = x[i];
    }
    
    fft(val_re_cpu, val_im_cpu, n_fft);
    fft(val_re_gpu, val_im_gpu, n_fft);
    
    // Apply on CPU
    apply_dispersion(val_re_cpu, val_im_cpu, n_fft, fs, 0.1f, 15.0f);
    
    // Apply on GPU
    apply_dispersion_cuda(val_re_gpu, val_im_gpu, n_fft, fs, 0.1f, 15.0f);
    
    float max_err = 0.0f;
    for (int i = 0; i < n_fft; i++) {
        float err_re = fabsf(val_re_cpu[i] - val_re_gpu[i]);
        float err_im = fabsf(val_im_cpu[i] - val_im_gpu[i]);
        if (err_re > max_err) max_err = err_re;
        if (err_im > max_err) max_err = err_im;
    }
    
    // ============================================
    // BENCHMARKING CPU vs GPU (100,000 Iterations - Compute Only)
    // ============================================
    struct timespec start, end;
    int ITERS = 100000;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int iter = 0; iter < ITERS; iter++) {
        apply_dispersion(val_re_cpu, val_im_cpu, n_fft, fs, 0.1f, 15.0f);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double cpu_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    // Uses CUDA events internally for precise kernel-only timing
    double gpu_time = apply_dispersion_cuda_benchmark(val_re_gpu, val_im_gpu, n_fft, fs, 0.1f, 15.0f, ITERS);
    
    // ============================================
    // END-TO-END BENCHMARK (Including PCIe Overhead)
    // ============================================
    int E2E_ITERS = 1000;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int iter = 0; iter < E2E_ITERS; iter++) {
        apply_dispersion(val_re_cpu, val_im_cpu, n_fft, fs, 0.1f, 15.0f);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double e2e_cpu_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int iter = 0; iter < E2E_ITERS; iter++) {
        apply_dispersion_cuda(val_re_gpu, val_im_gpu, n_fft, fs, 0.1f, 15.0f);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double e2e_gpu_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("======================================\n");
    printf("HARDWARE & METHODOLOGY RECORD\n");
    printf("--------------------------------------\n");
    print_gpu_info();
    #ifdef __VERSION__
    printf("Host Compiler Version: GCC %s\n", __VERSION__);
    #endif
    printf("FFT Size: %d bins\n", n_fft);
    printf("Warm-up Iterations: 0\n");
    printf("Compute Benchmark Iterations: %d\n", ITERS);
    printf("E2E Benchmark Iterations: %d\n", E2E_ITERS);
    printf("Kernel Timing: cudaEvent_t\n");
    printf("E2E Timing: clock_gettime(CLOCK_MONOTONIC)\n");
    
    printf("======================================\n");
    printf("CPU vs CUDA Validation Max Error: %e\n", max_err);
    if (max_err < 1e-4f) {
        printf("Validation PASSED! Hardware yields identical physics.\n");
    } else {
        printf("Validation FAILED!\n");
    }
    printf("--------------------------------------\n");
    printf("Compute-Path Performance (100,000 iters of %d bins):\n", n_fft);
    printf("CPU Time: %f seconds\n", cpu_time);
    printf("GPU Time (Resident): %f seconds\n", gpu_time);
    printf("Compute Speedup: %.2fx\n", cpu_time / gpu_time);
    printf("--------------------------------------\n");
    printf("End-to-End Performance (1,000 iters w/ PCIe & malloc overhead):\n");
    printf("CPU Time: %f seconds\n", e2e_cpu_time);
    printf("GPU Time (H2D + Compute + D2H): %f seconds\n", e2e_gpu_time);
    printf("E2E Speedup: %.2fx\n", e2e_cpu_time / e2e_gpu_time);
    printf("======================================\n\n");
    
    free(val_re_cpu); free(val_im_cpu); free(val_re_gpu); free(val_im_gpu);
    
    // ============================================
    // PIPELINE EXECUTION (M7: Multipath & Noise)
    // ============================================
    float *re1 = (float*)SAFE_CALLOC(n_fft, sizeof(float));
    float *im1 = (float*)SAFE_CALLOC(n_fft, sizeof(float));
    float *re2 = (float*)SAFE_CALLOC(n_fft, sizeof(float));
    float *im2 = (float*)SAFE_CALLOC(n_fft, sizeof(float));
    float *combined = (float*)SAFE_CALLOC(n_fft, sizeof(float));
    
    // Path 1 (Strong direct path)
    for (int i = 0; i < nx; i++) re1[i] = x[i];
    fft(re1, im1, n_fft);
    apply_dispersion_cuda(re1, im1, n_fft, fs, 0.1f, 15.0f); 
    ifft(re1, im1, n_fft);
    
    // Path 2 (Weaker, highly dispersed multipath)
    for (int i = 0; i < nx; i++) re2[i] = x[i];
    fft(re2, im2, n_fft);
    apply_dispersion_cuda(re2, im2, n_fft, fs, 0.15f, 25.0f); 
    ifft(re2, im2, n_fft);
    
    // Combine paths
    for (int i = 0; i < n_fft; i++) {
        combined[i] = re1[i] + (0.5f * re2[i]); 
    }
    
    // Add noise
    add_awgn(combined, n_fft, 0.005f);
    
    // STFT
    int stft_n_fft = 256;
    int hop = 64;
    int num_frames = 0;
    float *spectrogram = compute_stft(combined, n_fft, stft_n_fft, hop, &num_frames);
    
    FILE *f = fopen("data/spectrogram.bin", "wb");
    if (!f) {
        fprintf(stderr, "FATAL: Failed to open data/spectrogram.bin\n");
        exit(EXIT_FAILURE);
    }
    int32_t num_bins_val = stft_n_fft / 2;
    int32_t num_frames_val = num_frames;
    int32_t version = 1;
    fwrite("SPEC", 1, 4, f);
    write_le32(f, version);
    write_le_float(f, fs);
    write_le32(f, num_frames_val);
    write_le32(f, num_bins_val);
    write_le_float_array(f, spectrogram, num_frames_val * num_bins_val);
    fclose(f);
    printf("Pipeline complete. Wrote spectrogram.\n");
    
    FILE *fa = fopen("data/audio.bin", "wb");
    if (!fa) {
        fprintf(stderr, "FATAL: Failed to open data/audio.bin\n");
        exit(EXIT_FAILURE);
    }
    int32_t audio_len = n_fft;
    int32_t version_a = 1;
    fwrite("WAVA", 1, 4, fa);
    write_le32(fa, version_a);
    write_le_float(fa, fs);
    write_le32(fa, audio_len);
    write_le_float_array(fa, combined, audio_len);
    fclose(fa);
    printf("Saved audio waveform to data/audio.bin\n");
    
    free(x); free(re1); free(im1); free(re2); free(im2); free(combined); free(spectrogram);
    return 0;
}
