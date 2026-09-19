#include "io.h"

void write_le32(FILE *stream, int32_t value) {
    uint8_t bytes[4] = {
        (uint8_t)(value & 0xFF),
        (uint8_t)((value >> 8) & 0xFF),
        (uint8_t)((value >> 16) & 0xFF),
        (uint8_t)((value >> 24) & 0xFF)
    };
    fwrite(bytes, 1, 4, stream);
}

void write_le_float(FILE *stream, float value) {
    union { float f; int32_t i; } u;
    u.f = value;
    write_le32(stream, u.i);
}

void write_le_float_array(FILE *stream, const float *values, size_t count) {
    for (size_t i = 0; i < count; i++) {
        write_le_float(stream, values[i]);
    }
}
