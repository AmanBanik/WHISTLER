#ifndef IO_H
#define IO_H

#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void write_le32(FILE *stream, int32_t value);
void write_le_float(FILE *stream, float value);
void write_le_float_array(FILE *stream, const float *values, size_t count);

#ifdef __cplusplus
}
#endif

#endif // IO_H
