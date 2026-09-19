CC = gcc
NVCC = nvcc
CFLAGS = -Wall -Wextra -O3 -I./include
NVCCFLAGS = -O3 -I./include
LDFLAGS = -lm

SRC_C = src/core/main.c src/core/source_gen.c src/core/signal_math.c src/core/fft.c src/core/dispersion.c src/core/stft.c
SRC_CU = src/cuda/dispersion.cu

OBJ_C = $(SRC_C:.c=.o)
OBJ_CU = $(SRC_CU:.cu=.o)
OBJ = $(OBJ_C) $(OBJ_CU)

TARGET = whistler_m1

all: $(TARGET)

$(TARGET): $(OBJ)
	$(NVCC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)
