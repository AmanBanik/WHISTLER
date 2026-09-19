#include "whistler_core.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Swap function for bit-reversal
static void swap(float *a, float *b) {
    float temp = *a;
    *a = *b;
    *b = temp;
}

// In-place Radix-2 DIT FFT
// Assumes n is a power of 2
void fft(float *real, float *imag, int n) {
    if (n <= 0 || (n & (n - 1)) != 0) {
        fprintf(stderr, "FATAL: FFT size must be a positive power of 2 (got %d)\n", n);
        exit(EXIT_FAILURE);
    }
    
    // Bit-reversal permutation
    int j = 0;
    for (int i = 0; i < n - 1; i++) {
        if (i < j) {
            swap(&real[i], &real[j]);
            swap(&imag[i], &imag[j]);
        }
        int k = n / 2;
        while (k <= j) {
            j -= k;
            k /= 2;
        }
        j += k;
    }

    // Cooley-Tukey decimation-in-time radix-2 FFT
    for (int len = 2; len <= n; len <<= 1) {
        float angle = -2.0f * (float)M_PI / len;
        float wlen_real = cosf(angle);
        float wlen_imag = sinf(angle);
        
        for (int i = 0; i < n; i += len) {
            float w_real = 1.0f;
            float w_imag = 0.0f;
            for (int j = 0; j < len / 2; j++) {
                int u = i + j;
                int v = i + j + len / 2;
                
                // (w_real + i*w_imag) * (real[v] + i*imag[v])
                float t_real = w_real * real[v] - w_imag * imag[v];
                float t_imag = w_real * imag[v] + w_imag * real[v];
                
                real[v] = real[u] - t_real;
                imag[v] = imag[u] - t_imag;
                
                real[u] = real[u] + t_real;
                imag[u] = imag[u] + t_imag;
                
                // Update twiddle factor
                float next_w_real = w_real * wlen_real - w_imag * wlen_imag;
                float next_w_imag = w_real * wlen_imag + w_imag * wlen_real;
                w_real = next_w_real;
                w_imag = next_w_imag;
            }
        }
    }
}

// Calculate the magnitude spectrum
void compute_spectrum(const float *real, const float *imag, float *magnitude, int n) {
    for (int i = 0; i < n; i++) {
        magnitude[i] = sqrtf(real[i] * real[i] + imag[i] * imag[i]);
    }
}

// Inverse Radix-2 FFT
void ifft(float *real, float *imag, int n) {
    // Conjugate the input
    for (int i = 0; i < n; i++) {
        imag[i] = -imag[i];
    }
    
    // Forward FFT
    fft(real, imag, n);
    
    // Conjugate output and scale by 1/n
    float scale = 1.0f / n;
    for (int i = 0; i < n; i++) {
        imag[i] = -imag[i] * scale;
        real[i] = real[i] * scale;
    }
}
