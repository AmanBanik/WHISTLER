#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// CUDA kernel to apply dispersion
__global__ void apply_dispersion_kernel(float *real, float *imag, int n, float fs, float t0, float D) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (k < n) {
        float f = 0.0f;
        if (k <= n / 2) {
            f = (float)k * fs / n;
        } else {
            f = (float)(k - n) * fs / n;
        }
        
        float abs_f = fabsf(f);
        float tau = t0;
        
        // Bounded effective frequency to prevent singularity and smooth cutoff
        float f_eff = fmaxf(abs_f, 1.0f);
        tau += D / sqrtf(f_eff);

        float phase = -2.0f * (float)M_PI * f * tau;
        
        float cos_phi = cosf(phase);
        float sin_phi = sinf(phase);
        
        float re = real[k];
        float im = imag[k];
        
        real[k] = re * cos_phi - im * sin_phi;
        imag[k] = re * sin_phi + im * cos_phi;
    }
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                    __FILE__, __LINE__, err, cudaGetErrorString(err), #call); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

extern "C" void apply_dispersion_cuda(float *real, float *imag, int n, float fs, float t0, float D) {
    float *d_real = NULL, *d_imag = NULL;
    size_t size = n * sizeof(float);
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_real, size));
    CUDA_CHECK(cudaMalloc((void**)&d_imag, size));
    
    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_real, real, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_imag, imag, size, cudaMemcpyHostToDevice));
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    apply_dispersion_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_real, d_imag, n, fs, t0, D);
    CUDA_CHECK(cudaGetLastError());
    
    // Wait for kernel to finish and check for errors
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(real, d_real, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(imag, d_imag, size, cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_real));
    CUDA_CHECK(cudaFree(d_imag));
}

extern "C" {
    double apply_dispersion_cuda_benchmark(float *real, float *imag, int n, float fs, float t0, float D, int iters) {
        size_t size = n * sizeof(float);
        float *d_real = NULL, *d_imag = NULL;
        
        CUDA_CHECK(cudaMalloc((void**)&d_real, size));
        CUDA_CHECK(cudaMalloc((void**)&d_imag, size));
        
        CUDA_CHECK(cudaMemcpy(d_real, real, size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_imag, imag, size, cudaMemcpyHostToDevice));
        
        int blockSize = 256;
        int numBlocks = (n + blockSize - 1) / blockSize;
        
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start));
        // Loop the kernel execution (no memory transfer overhead)
        for(int i=0; i<iters; i++) {
            apply_dispersion_kernel<<<numBlocks, blockSize>>>(d_real, d_imag, n, fs, t0, D);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventSynchronize(stop)); // wait for all kernel loops to finish
        
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        
        CUDA_CHECK(cudaMemcpy(real, d_real, size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(imag, d_imag, size, cudaMemcpyDeviceToHost));
        
        CUDA_CHECK(cudaFree(d_real));
        CUDA_CHECK(cudaFree(d_imag));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        
        return (double)milliseconds / 1000.0;
    }
}

extern "C" void print_gpu_info() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    int driverVersion = 0, runtimeVersion = 0;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    
    printf("GPU Model: %s\n", prop.name);
    printf("GPU Architecture: sm_%d%d\n", prop.major, prop.minor);
    printf("CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);
    printf("NVIDIA Driver Version: %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10);
    #ifdef __CUDACC_VER_MAJOR__
    printf("NVCC Compiler Version: %d.%d\n", __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__);
    #endif
}
