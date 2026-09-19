# ⚡ WHISTLER: Computational Modeling of Lightning-Generated Whistlers

> **Date:** September 2026  
> **Repository:** [WHISTLER Core](.)  

---

## 1. Abstract
This paper presents a simplified computational model for the synthesis and analysis of **lightning-generated whistlers**. A broadband electromagnetic transient is simulated and passed through a simplified magnetized-plasma propagation model, where the phase delay is frequency-dependent. The output is processed using a Short-Time Fourier Transform (STFT) to produce a time-frequency trace that characteristically descends with time, a hallmark of whistler waves. Our approach provides a small, inspectable signal pipeline wherein the unusual natural phenomenon emerges from explicit signal processing operations rather than black-box models. 

---

## 2. Introduction & Physical Context
Lightning-generated whistlers are radio-frequency (RF) emissions that become highly dispersed while propagating through the Earth's magnetized plasma, particularly within the plasmasphere [1]. In a spectrogram, this dispersion manifests as a characteristic descending tone. 

While full-wave plasma solvers can accurately predict these signatures, they are often computationally intensive and opaque. The engineering objective of the WHISTLER project is to construct a computational model that recreates this phenomenon through an explicit, heterogeneous digital signal processing (DSP) pipeline.

When a lightning strike occurs, a portion of its broadband electromagnetic energy escapes into the magnetosphere [2]. As these Very Low Frequency (VLF) waves travel along the Earth's magnetic field lines, they interact with the cold plasma of the plasmasphere. Because the plasma is a dispersive medium, higher frequencies travel faster than lower frequencies, leading to a frequency-dependent arrival time at a distant receiver [3]. 

---

## 3. Mathematical Model & Assumptions

By framing the problem as a discrete-time signal propagating through a linear time-invariant (LTI) system with a frequency-dependent phase delay, we isolate the fundamental physics and signal operations from complex environmental variables. This is a simplified engineering approximation of a full plasma-wave solver.

### 3.1 Source Transient and Channel Response
The lightning strike is approximated as a damped broadband pulse in the time domain:

$$
x(t) = A e^{-\alpha t} \sin(2\pi f_c t + \phi)
$$

While the propagation channel can be formulated conceptually as a time-domain convolution $y[n] = (x * h)[n]$, explicitly computing this convolution is computationally prohibitive. Instead, the implementation utilizes the mathematical equivalence of frequency-domain multiplication $Y[k] = X[k] H[k]$. By transforming the signal into the frequency domain via an FFT, the physical dispersion is efficiently applied as an element-wise phase rotation before an IFFT returns the signal to the time domain.

### 3.2 Dispersion Approximation
To capture the physics of plasma dispersion for VLF whistler waves, we apply a frequency-dependent phase delay $\tau(f)$. We use a simplified engineering approximation that captures the qualitative dispersion behavior:

$$
\tau(f) = t_0 + D f^{-1/2}
$$

where $t_0$ is the constant propagation delay and $D$ is the dispersion constant. The phase response is derived directly as $\phi(f) = -2\pi f \tau(f)$. This guarantees that lower frequencies experience greater phase wrapping, yielding the desired group delay characteristics and generating the descending tone signature.

### 3.3 Noise and Multipath Effects
The received signal $r[n]$ is subject to Additive White Gaussian Noise (AWGN) $n[n]$, configurable to test system robustness under varying Signal-to-Noise Ratios (SNR):

$$
r[n] = y[n] + n[n]
$$

Additionally, multipath propagation—where energy travels along multiple field-aligned ducts—is modeled by superimposing multiple instances of $y[n]$ with varying attenuation and dispersion constants.

### 3.4 Time-Frequency Analysis
To visualize the descending tone, the signal is processed using the Short-Time Fourier Transform (STFT). Using a sliding Hann window $w[n]$ of length $N$ and hop size $H$, the spectrogram $S(m, k)$ is computed as:

$$
S(m,k) = \left|\sum_{n=0}^{N-1} r[n+mH] w[n] e^{-j2\pi kn/N}\right|^2
$$

---

## 4. Heterogeneous Implementation Architecture

The WHISTLER system employs a heterogeneous computing architecture to maximize throughput and minimize latency. 

### 4.1 Hardware Target & Data Flow
The codebase is designed to be compiled via `nvcc` targeting modern NVIDIA GPU architectures (SM_75+) while the host code targets a standard x86_64 or ARM64 CPU. The pipeline executes sequentially as follows:

```mermaid
graph TD
    A[Pulse Generation C] --> B[FFT C]
    B --> C{PCIe Transfer to VRAM}
    C --> D[apply_dispersion_kernel CUDA]
    D --> E{PCIe Transfer to RAM}
    E --> F[IFFT C]
    F --> G[Multipath + AWGN C]
    G --> H[STFT Hann Window C]
    H --> I[(Binary Data Dump)]
    I --> J[Python plot.py]
    J --> K[waterfall.png / .gif]
    J --> L[whistler.wav]
    
    style D fill:#76B900,stroke:#333,stroke-width:2px,color:#fff
    style A fill:#00599C,stroke:#333,stroke-width:2px,color:#fff
    style J fill:#3776AB,stroke:#333,stroke-width:2px,color:#fff
```

### 4.2 Core CUDA Implementation
In contrast to the CPU orchestrating logic, the highly parallelizable task of applying the frequency-dependent dispersion delay is offloaded to the GPU via CUDA. For each frequency bin in the transformed signal, the kernel computes a non-linear phase shift involving square roots and trigonometric functions. 

> **Note on Tensor Cores:** Specialized Tensor Cores were explicitly avoided for this task. Tensor Cores are designed primarily for matrix multiply-accumulate (MMA) operations and often operate at reduced precision. The dispersion calculation requires independent, element-wise transcendental math at full single-precision (FP32), making standard CUDA ALUs the optimal and necessary choice to maintain physical fidelity.

```cpp
__global__ void apply_dispersion_kernel(float *real, float *imag, int n, float fs, float t0, float D) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    
    // Handle both positive and negative frequencies
    float f = (k <= n / 2) ? ((float)k * fs / n) : ((float)(k - n) * fs / n);
    float abs_f = fabsf(f);
    
    float tau = t0;
    // Bounded effective frequency to prevent singularity and smooth cutoff
    float f_eff = fmaxf(abs_f, 1.0f);
    tau += D / sqrtf(f_eff);
    
    float phase = -2.0f * (float)M_PI * f * tau;

    // Full precision ALU execution (Avoids fast-math drift)
    float cos_phi = cosf(phase);
    float sin_phi = sinf(phase);
    
    float re = real[k];
    float im = imag[k];
    
    real[k] = re * cos_phi - im * sin_phi;
    imag[k] = re * sin_phi + im * cos_phi;
}
```

### 4.3 Binary Serialization
To ensure cross-platform reproducibility and decoupled visualization, the pipeline serializes the final outputs into explicit 32-bit little-endian binary blobs (`data/spectrogram.bin` and `data/audio.bin`). Each file begins with a 4-byte magic header (`"SPEC"` or `"WAVA"`) followed by tightly packed metadata (`version`, `fs`, `N`, `M`) and a flat array of IEEE-754 single-precision (FP32) floats. This enforces strict numeric portability when ingesting the data into higher-level analytical tools like Python.

---

## 5. Validation & Benchmarking

To ensure that the hardware acceleration does not compromise the physical accuracy of the simulation, strict numerical validation was performed between the CPU and GPU implementations. 

* **Physical Parity (Validation):** By applying the dispersion algorithm to identical input pulses, the maximum absolute error between the C and CUDA outputs was measured at **$1.9 \times 10^{-6}$**. This error is on the order expected for single-precision (FP32) numerical floating-point operations.
* **Performance Speedup (Benchmarking):** For a problem size of 16,384 FFT bins evaluated over 100,000 iterations, the purely CPU-based implementation required **16.32 seconds** to complete. The equivalent CUDA benchmark completed the same computational workload in just **0.94 seconds** (measuring raw kernel throughput while keeping data resident on the GPU). This translates to a massive **17.3x compute-path speedup**.

---

## 6. Experiments & Visual Results

The simulation pipeline successfully outputs a time-domain signal corresponding to a whistler wave with a secondary multipath echo. The characteristic descending tone is clearly visible in the STFT spectrograms below.

### 🖼️ Static High-Resolution Spectrogram
![Static Spectrogram](../data/waterfall_static.png)

### 🎥 Dynamic Real-Time Sweeping Waterfall
![Animated Spectrogram](../data/waterfall_animated.gif)

### 🔊 Listen to the Waveform
Because the frequencies involved overlap with the human auditory range, the binary waveform is normalized into 16-bit PCM format. You can play the raw physical output directly in the browser:

<audio controls>
  <source src="../data/whistler.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>

*(If the player is not supported, you can [download whistler.wav](../data/whistler.wav) directly)*

---

## 7. Conclusion
This project successfully developed a simplified computational model for lightning-generated whistlers using an explicit DSP pipeline. The heterogeneous architecture effectively balances control logic on the CPU and parallelizable, transcendental element-wise operations on the GPU. The result is a highly efficient, inspectable computational approximation that produces physically motivated signatures of whistler waves, enabling further study.

### 7.1 Limitations
The current model makes several explicit engineering assumptions:
* The dispersion and phase delay law is a simplified approximation and not derived from a full magnetospheric cold-plasma solver.
* Multipath and noise effects are basic additive constructs.
* The model output has not yet been rigorously validated against raw satellite or ground-based VLF recordings.
* At $1.0\text{ Hz}$, the frequency is artificially bounded (`fmaxf`) to prevent a singularity at DC, reflecting a continuous mathematical cutoff rather than an exact physical propagation cutoff boundary.

### References & Core Concepts

#### Magnetospheric Physics References
1. NASA, *The Plasmasphere*, 2023.
2. NASA, *Lightning and Whistlers*, 2022.
3. NASA, *Structure of the Magnetosphere*, 2021.

#### Signals & Systems Concepts Utilized
* **Linear Time-Invariant (LTI) Systems:** The magnetospheric propagation channel is mathematically framed as a discrete LTI system where the received signal is the convolution of the lightning transient and the channel impulse response $h[n]$.
* **Frequency-Dependent Phase Delay:** Unlike ideal communication channels with constant delay, whistler dispersion relies on a phase delay law $\tau(f)$ changing as a function of frequency, heavily delaying lower spectral components to generate characteristic group delay signatures.
* **Short-Time Fourier Transform (STFT):** A time-frequency analysis technique used to isolate the descending tone. By taking the FFT of overlapping, windowed segments, it bypasses the time-resolution loss inherent to a standard global Fourier Transform.
* **Hann Windowing:** Applied prior to the STFT to taper the ends of the time-domain chunks to zero, drastically reducing spectral leakage and high-frequency artifacts.
* **Additive White Gaussian Noise (AWGN):** A foundational probabilistic noise model added to the channel to simulate thermal receiver noise and general background radiation.
