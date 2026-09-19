#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

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
        
        if (abs_f > 1.0f) {
            tau += D / sqrtf(abs_f); 
        }
        
        float phase = -2.0f * (float)M_PI * f * tau;
        
        float cos_phi = cosf(phase);
        float sin_phi = sinf(phase);
        
        float re = real[k];
        float im = imag[k];
        
        real[k] = re * cos_phi - im * sin_phi;
        imag[k] = re * sin_phi + im * cos_phi;
    }
}

extern "C" void apply_dispersion_cuda(float *real, float *imag, int n, float fs, float t0, float D) {
    float *d_real, *d_imag;
    size_t size = n * sizeof(float);
    
    // Allocate device memory
    cudaMalloc((void**)&d_real, size);
    cudaMalloc((void**)&d_imag, size);
    
    // Copy data from host to device
    cudaMemcpy(d_real, real, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_imag, imag, size, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    apply_dispersion_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_real, d_imag, n, fs, t0, D);
    
    // Wait for kernel to finish and check for errors
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaMemcpy(real, d_real, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(imag, d_imag, size, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_real);
    cudaFree(d_imag);
}

extern "C" {
    void apply_dispersion_cuda_benchmark(float *real, float *imag, int n, float fs, float t0, float D, int iters) {
        size_t size = n * sizeof(float);
        float *d_real, *d_imag;
        
        cudaMalloc((void**)&d_real, size);
        cudaMalloc((void**)&d_imag, size);
        
        cudaMemcpy(d_real, real, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_imag, imag, size, cudaMemcpyHostToDevice);
        
        int blockSize = 256;
        int numBlocks = (n + blockSize - 1) / blockSize;
        
        // Loop the kernel execution (no memory transfer overhead)
        for(int i=0; i<iters; i++) {
            apply_dispersion_kernel<<<numBlocks, blockSize>>>(d_real, d_imag, n, fs, t0, D);
        }
        
        cudaDeviceSynchronize(); // wait for all kernel loops to finish
        
        cudaMemcpy(real, d_real, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(imag, d_imag, size, cudaMemcpyDeviceToHost);
        
        cudaFree(d_real);
        cudaFree(d_imag);
    }
}
