#include "whistler_core.h"
#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Generate standard normal distributed random number (Box-Muller)
static float rand_normal() {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    if (u1 <= 1e-7f) u1 = 1e-7f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

void add_awgn(float *signal, int len, float noise_std_dev) {
    for (int i = 0; i < len; i++) {
        signal[i] += noise_std_dev * rand_normal();
    }
}

