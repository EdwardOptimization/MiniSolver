// Standalone exploratory benchmark for batched short-horizon scalar Riccati.
//
// Unlike MPX/PCR scans over one long horizon, this route assigns one CUDA
// thread to one independent short Riccati recursion. It tests whether GPU work
// is more plausible for many MPC samples/guesses/problems than for one horizon.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#define CUDA_CHECK(expr)                                                                           \
    do {                                                                                           \
        const cudaError_t err__ = (expr);                                                          \
        if (err__ != cudaSuccess) {                                                                \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "                   \
                      << cudaGetErrorString(err__) << std::endl;                                   \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (false)

namespace {

struct Stage {
    double a;
    double b;
    double q;
    double r;
};

template <typename F> float time_cuda_ms(F&& fn)
{
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    fn();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
}

inline void consume_for_benchmark(double value)
{
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "g"(value) : "memory");
#else
    volatile double sink = value;
    (void)sink;
#endif
}

void generate_stage_batch(std::vector<Stage>& stages, int horizon, int batch)
{
    stages.resize(static_cast<std::size_t>(horizon * batch));
    std::mt19937 rng(9090 + 13 * horizon + batch);
    std::uniform_real_distribution<double> a_dist(0.92, 1.02);
    std::uniform_real_distribution<double> b_dist(0.05, 0.18);
    std::uniform_real_distribution<double> q_dist(0.5, 2.0);
    std::uniform_real_distribution<double> r_dist(0.2, 2.0);

    for (auto& s : stages) {
        s.a = a_dist(rng);
        s.b = b_dist(rng);
        s.q = q_dist(rng);
        s.r = r_dist(rng);
    }
}

__host__ __device__ double riccati_step(const Stage& s, double next)
{
    return s.q + (s.a * s.a * s.r * next) / (s.r + s.b * s.b * next);
}

void cpu_batched_riccati(const std::vector<Stage>& stages, std::vector<double>& output, int horizon,
    int batch, double terminal)
{
    output.resize(static_cast<std::size_t>(batch));
    for (int i = 0; i < batch; ++i) {
        double p = terminal;
        const Stage* problem = stages.data() + static_cast<std::size_t>(i * horizon);
        for (int k = horizon - 1; k >= 0; --k) {
            p = riccati_step(problem[k], p);
        }
        output[static_cast<std::size_t>(i)] = p;
    }
}

__global__ void batched_riccati_kernel(
    const Stage* stages, double* output, int horizon, int batch, double terminal)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch) {
        return;
    }
    double p = terminal;
    const Stage* problem = stages + static_cast<std::size_t>(idx * horizon);
    for (int k = horizon - 1; k >= 0; --k) {
        p = riccati_step(problem[k], p);
    }
    output[idx] = p;
}

float gpu_batched_riccati(const std::vector<Stage>& stages, std::vector<double>& output,
    int horizon, int batch, int repeats, double terminal)
{
    Stage* d_stages = nullptr;
    double* d_output = nullptr;
    const std::size_t stage_bytes = stages.size() * sizeof(Stage);
    const std::size_t output_bytes = static_cast<std::size_t>(batch) * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_stages, stage_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, output_bytes));
    CUDA_CHECK(cudaMemcpy(d_stages, stages.data(), stage_bytes, cudaMemcpyHostToDevice));

    const int threads = 128;
    const int blocks = (batch + threads - 1) / threads;
    batched_riccati_kernel<<<blocks, threads>>>(d_stages, d_output, horizon, batch, terminal);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const float total_ms = time_cuda_ms([&]() {
        for (int r = 0; r < repeats; ++r) {
            batched_riccati_kernel<<<blocks, threads>>>(
                d_stages, d_output, horizon, batch, terminal);
            CUDA_CHECK(cudaGetLastError());
        }
    });

    output.resize(static_cast<std::size_t>(batch));
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, output_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_stages));
    CUDA_CHECK(cudaFree(d_output));

    return total_ms / static_cast<float>(repeats);
}

double max_abs_error(const std::vector<double>& expected, const std::vector<double>& actual)
{
    double err = 0.0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        err = std::max(err, std::abs(expected[i] - actual[i]));
    }
    return err;
}

void run_case(int horizon, int batch, int repeats)
{
    constexpr double terminal = 1.0;
    std::vector<Stage> stages;
    std::vector<double> cpu_output;
    std::vector<double> gpu_output;
    generate_stage_batch(stages, horizon, batch);

    cpu_batched_riccati(stages, cpu_output, horizon, batch, terminal);
    const auto cpu_start = std::chrono::steady_clock::now();
    for (int r = 0; r < repeats; ++r) {
        cpu_batched_riccati(stages, cpu_output, horizon, batch, terminal);
        consume_for_benchmark(cpu_output.back());
    }
    const auto cpu_end = std::chrono::steady_clock::now();
    const double cpu_us = std::chrono::duration<double, std::micro>(cpu_end - cpu_start).count()
        / static_cast<double>(repeats);

    const float gpu_ms = gpu_batched_riccati(stages, gpu_output, horizon, batch, repeats, terminal);
    const double gpu_us = 1000.0 * static_cast<double>(gpu_ms);
    const double err = max_abs_error(cpu_output, gpu_output);

    std::cout << std::setw(4) << horizon << " " << std::setw(7) << batch << " " << std::setw(4)
              << repeats << " " << std::setw(12) << std::fixed << std::setprecision(2) << cpu_us
              << " " << std::setw(12) << gpu_us << " " << std::setw(9) << std::setprecision(2)
              << (cpu_us / gpu_us) << " " << std::scientific << std::setprecision(2)
              << std::setw(11) << err << "\n";
}

} // namespace

int main()
{
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp props {};
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));

    std::cout << "MiniSolver CUDA batched scalar Riccati microbenchmark\n";
    std::cout << "Device: " << props.name << "\n";
    std::cout << "Metric: device-resident recursion time; host<->device transfers excluded\n";
    std::cout << "Kernel: one CUDA thread solves one independent scalar Riccati horizon\n";
    std::cout << "   N   batch    R       CPU_us       GPU_us   GPU_spd         err\n";

    for (const int horizon : { 32, 128 }) {
        for (const int batch : { 1, 256, 4096 }) {
            const int repeats = (batch <= 256) ? 100 : 20;
            run_case(horizon, batch, repeats);
        }
    }

    return 0;
}
