#ifndef WHISTLER_CORE_H
#define WHISTLER_CORE_H

#include <stdlib.h>
#include <stdio.h>

#define SAFE_CALLOC(n, size) \
    ({ \
        void *ptr = calloc(n, size); \
        if (!ptr) { \
            fprintf(stderr, "FATAL: Memory allocation failed at %s:%d\n", __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
        ptr; \
    })

#define SAFE_MALLOC(size) \
    ({ \
        void *ptr = malloc(size); \
        if (!ptr) { \
            fprintf(stderr, "FATAL: Memory allocation failed at %s:%d\n", __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
        ptr; \
    })

#ifdef __cplusplus
extern "C" {
#endif

// Generate a damped broadband pulse
// x(t) = A * exp(-alpha * t) * sin(2 * pi * f_c * t + phi)
void generate_damped_pulse(float *out, int length, float fs, float A, float alpha, float fc, float phi);


// Add Additive White Gaussian Noise
void add_awgn(float *signal, int len, float noise_std_dev);

// In-place Radix-2 FFT (n must be a power of 2)
void fft(float *real, float *imag, int n);

// Inverse Radix-2 FFT (n must be a power of 2)
void ifft(float *real, float *imag, int n);

// Calculate magnitude spectrum
void compute_spectrum(const float *real, const float *imag, float *magnitude, int n);

// Apply frequency-dependent delay: tau(f) = t0 + D * f^(-1/2)
void apply_dispersion(float *real, float *imag, int n, float fs, float t0, float D);

// Frequency domain phase dispersion using CUDA
void apply_dispersion_cuda(float *real, float *imag, int n, float fs, float t0, float D);

// Benchmark: loops kernel in CUDA to ignore PCIe overhead
double apply_dispersion_cuda_benchmark(float *real, float *imag, int n, float fs, float t0, float D, int iters);

// Apply a window function to a frame
void apply_window(float *frame, int n);

// Compute STFT and return a flat spectrogram matrix
float* compute_stft(const float *signal, int sig_len, int n_fft, int hop, int *out_frames);

#ifdef __cplusplus
}
#endif

#endif // WHISTLER_CORE_H
