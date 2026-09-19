CC = gcc
NVCC = nvcc
CFLAGS = -Wall -Wextra -O3 -I./include
NVCCFLAGS = -O3 -I./include -arch=sm_75
LDFLAGS = -lm

SRC_C = src/core/main.c src/core/source_gen.c src/core/signal_math.c src/core/fft.c src/core/dispersion.c src/core/stft.c
CLI_SRC_C = src/core/main_cli.c src/core/source_gen.c src/core/signal_math.c src/core/fft.c src/core/dispersion.c src/core/stft.c
TEST_SRC_C = src/core/test_suite.c src/core/source_gen.c src/core/signal_math.c src/core/fft.c src/core/dispersion.c src/core/stft.c
SRC_CU = src/cuda/dispersion.cu

OBJ_C = $(SRC_C:.c=.o)
CLI_OBJ_C = $(CLI_SRC_C:.c=.o)
TEST_OBJ_C = $(TEST_SRC_C:.c=.o)
OBJ_CU = $(SRC_CU:.cu=.o)

TARGET = whistler_m1
CLI_TARGET = whistler_cli
TEST_TARGET = whistler_test

all: $(TARGET) $(CLI_TARGET) $(TEST_TARGET)

test: $(TEST_TARGET)
	./$(TEST_TARGET)

$(TARGET): $(OBJ_C) $(OBJ_CU)
	$(NVCC) -o $@ $^ $(LDFLAGS)

$(CLI_TARGET): $(CLI_OBJ_C) $(OBJ_CU)
	$(NVCC) -o $@ $^ $(LDFLAGS)

$(TEST_TARGET): $(TEST_OBJ_C) $(OBJ_CU)
	$(NVCC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f src/core/*.o src/cuda/*.o $(TARGET) $(CLI_TARGET) $(TEST_TARGET)
