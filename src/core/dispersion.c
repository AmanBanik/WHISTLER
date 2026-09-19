#include "whistler_core.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Apply frequency-dependent delay: tau(f) = t0 + D * f^(-1/2)
void apply_dispersion(float *real, float *imag, int n, float fs, float t0, float D) {
    for (int k = 0; k < n; k++) {
        // Calculate the actual frequency for this bin
        float f = 0.0f;
        if (k <= n / 2) {
            f = (float)k * fs / n;
        } else {
            f = (float)(k - n) * fs / n;
        }
        
        float abs_f = fabsf(f);
        
        // Calculate delay for this frequency
        float tau = t0;
        
        // Bounded effective frequency to prevent singularity and smooth cutoff
        float f_eff = fmaxf(abs_f, 1.0f);
        tau += D / sqrtf(f_eff);
        
        // Phase shift: phi = -2 * pi * f * tau
        float phase = -2.0f * (float)M_PI * f * tau;
        
        // Apply phase shift: X(f) * exp(j * phi)
        float cos_phi = cosf(phase);
        float sin_phi = sinf(phase);
        
        float re = real[k];
        float im = imag[k];
        
        real[k] = re * cos_phi - im * sin_phi;
        imag[k] = re * sin_phi + im * cos_phi;
    }
}
