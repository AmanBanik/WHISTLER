#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <stdint.h>
#include "whistler_core.h"

int main(int argc, char **argv) {
    int num_paths = 2;
    float noise_level = 0.005f;
    float base_D = 15.0f;
    
    // Simple argument parsing
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) num_paths = atoi(argv[++i]);
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) noise_level = atof(argv[++i]);
        if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) base_D = atof(argv[++i]);
    }
    
    if (num_paths < 1) {
        fprintf(stderr, "FATAL: num_paths must be >= 1\n");
        exit(EXIT_FAILURE);
    }
    if (noise_level < 0.0f) {
        fprintf(stderr, "FATAL: noise_level must be >= 0\n");
        exit(EXIT_FAILURE);
    }
    if (base_D <= 0.0f) {
        fprintf(stderr, "FATAL: dispersion constant must be > 0\n");
        exit(EXIT_FAILURE);
    }

    printf("Configuration -> Paths: %d, Noise Level: %f, Base Dispersion: %f\n", num_paths, noise_level, base_D);

    float fs = 10000.0f; 
    int nx = 200; 
    float *x = (float*)malloc(nx * sizeof(float));
    
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
    printf("Warm-up Iterations: 0 (Implicitly covered by pipeline syncs)\n");
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
    float *combined = (float*)SAFE_CALLOC(n_fft, sizeof(float));
    
    for (int p = 0; p < num_paths; p++) {
        float *re = (float*)SAFE_CALLOC(n_fft, sizeof(float));
        float *im = (float*)SAFE_CALLOC(n_fft, sizeof(float));
        
        for (int i = 0; i < nx; i++) re[i] = x[i];
        fft(re, im, n_fft);
        
        float t0 = 0.1f + p * 0.05f;       // Incremental delay for echoes
        float D = base_D + p * 10.0f;      // Incremental dispersion for echoes
        float attenuation = 1.0f / (p + 1.0f); // Amplitude falloff
        
        apply_dispersion_cuda(re, im, n_fft, fs, t0, D); 
        ifft(re, im, n_fft);
        
        // Combine path into final buffer
        for (int i = 0; i < n_fft; i++) {
            combined[i] += attenuation * re[i]; 
        }
        
        free(re);
        free(im);
    }
    
    // Add noise
    add_awgn(combined, n_fft, noise_level);
    
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
    if (fwrite("SPEC", 1, 4, f) != 4 ||
        fwrite(&version, sizeof(int32_t), 1, f) != 1 ||
        fwrite(&fs, sizeof(float), 1, f) != 1 ||
        fwrite(&num_frames_val, sizeof(int32_t), 1, f) != 1 ||
        fwrite(&num_bins_val, sizeof(int32_t), 1, f) != 1 ||
        fwrite(spectrogram, sizeof(float), num_frames_val * num_bins_val, f) != (size_t)(num_frames_val * num_bins_val)) {
        fprintf(stderr, "FATAL: Failed to write to data/spectrogram.bin\n");
        exit(EXIT_FAILURE);
    }
    fclose(f);
    printf("Pipeline complete. Wrote spectrogram.\n");
    
    FILE *fa = fopen("data/audio.bin", "wb");
    if (!fa) {
        fprintf(stderr, "FATAL: Failed to open data/audio.bin\n");
        exit(EXIT_FAILURE);
    }
    int32_t audio_len = n_fft;
    int32_t version_a = 1;
    if (fwrite("WAVA", 1, 4, fa) != 4 ||
        fwrite(&version_a, sizeof(int32_t), 1, fa) != 1 ||
        fwrite(&fs, sizeof(float), 1, fa) != 1 ||
        fwrite(&audio_len, sizeof(int32_t), 1, fa) != 1 ||
        fwrite(combined, sizeof(float), audio_len, fa) != (size_t)audio_len) {
        fprintf(stderr, "FATAL: Failed to write to data/audio.bin\n");
        exit(EXIT_FAILURE);
    }
    fclose(fa);
    printf("Saved audio waveform to data/audio.bin\n");
    
    free(x); free(combined); free(spectrogram);
    return 0;
}
