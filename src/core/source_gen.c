#include "whistler_core.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void generate_damped_pulse(float *out, int length, float fs, float A, float alpha, float fc, float phi) {
    for (int i = 0; i < length; ++i) {
        float t = (float)i / fs;
        out[i] = A * expf(-alpha * t) * sinf(2.0f * (float)M_PI * fc * t + phi);
    }
}
