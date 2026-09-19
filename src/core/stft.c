#include "whistler_core.h"
#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// Compute STFT: returns a flat array of size num_frames * (n_fft/2)
// representing the spectrogram (magnitude squared).
float* compute_stft(const float *signal, int sig_len, int n_fft, int hop, int *out_frames) {
    int num_frames = 1 + (sig_len - n_fft) / hop;
    if (num_frames <= 0) return NULL;
    
    *out_frames = num_frames;
    int num_bins = n_fft / 2;
    float *spectrogram = (float*)malloc(num_frames * num_bins * sizeof(float));
    
    float *re = (float*)malloc(n_fft * sizeof(float));
    float *im = (float*)malloc(n_fft * sizeof(float));
    
    // Precompute Hann window for crisp STFT
    float *window = (float*)malloc(n_fft * sizeof(float));
    for (int i = 0; i < n_fft; i++) {
        window[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (n_fft - 1)));
    }
    
    for (int m = 0; m < num_frames; m++) {
        int start = m * hop;
        for (int i = 0; i < n_fft; i++) {
            if (start + i < sig_len) {
                re[i] = signal[start + i] * window[i];
            } else {
                re[i] = 0.0f;
            }
            im[i] = 0.0f;
        }
        
        fft(re, im, n_fft);
        
        for (int k = 0; k < num_bins; k++) {
            spectrogram[m * num_bins + k] = re[k] * re[k] + im[k] * im[k];
        }
    }
    
    free(window);
    free(re);
    free(im);
    return spectrogram;
}
