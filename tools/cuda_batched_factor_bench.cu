// Standalone exploratory benchmark for batched small-matrix factorization.
//
// This benchmark compares a CPU sequential Cholesky decomposition against a
// simple CUDA batched Cholesky kernel. It is not a solver backend and is not
// optimized enough to justify Backend::GPU_* support by itself.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <thread>
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

template <int DIM> void generate_spd_batch(std::vector<double>& batch, int count)
{
    batch.assign(static_cast<std::size_t>(count * DIM * DIM), 0.0);
    std::mt19937 rng(2026 + DIM * 31 + count);
    std::uniform_real_distribution<double> offdiag(-0.02, 0.02);
    std::uniform_real_distribution<double> diag_noise(0.0, 0.2);

    for (int m = 0; m < count; ++m) {
        double* A = batch.data() + static_cast<std::size_t>(m * DIM * DIM);
        for (int r = 0; r < DIM; ++r) {
            double row_sum = 0.0;
            for (int c = 0; c < DIM; ++c) {
                if (r == c) {
                    continue;
                }
                const double v = offdiag(rng);
                A[r * DIM + c] = v;
                A[c * DIM + r] = v;
                row_sum += std::abs(v);
            }
            A[r * DIM + r] = 1.0 + row_sum + diag_noise(rng);
        }
    }
}

template <int DIM> __host__ __device__ bool cholesky_one(const double* A, double* L)
{
    for (int i = 0; i < DIM * DIM; ++i) {
        L[i] = 0.0;
    }

    for (int j = 0; j < DIM; ++j) {
        double diag = A[j * DIM + j];
        for (int k = 0; k < j; ++k) {
            const double v = L[j * DIM + k];
            diag -= v * v;
        }
        if (!(diag > 0.0)) {
            return false;
        }
        L[j * DIM + j] = sqrt(diag);

        for (int i = j + 1; i < DIM; ++i) {
            double acc = A[i * DIM + j];
            for (int k = 0; k < j; ++k) {
                acc -= L[i * DIM + k] * L[j * DIM + k];
            }
            L[i * DIM + j] = acc / L[j * DIM + j];
        }
    }
    return true;
}

template <int DIM>
__global__ void cholesky_batch_kernel(const double* input, double* output, int* info, int count)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    const double* A = input + static_cast<std::size_t>(idx * DIM * DIM);
    double* L = output + static_cast<std::size_t>(idx * DIM * DIM);
    info[idx] = cholesky_one<DIM>(A, L) ? 0 : 1;
}

template <int DIM>
void cpu_factor_batch(const std::vector<double>& input, std::vector<double>& output, int count)
{
    output.resize(static_cast<std::size_t>(count * DIM * DIM));
    for (int m = 0; m < count; ++m) {
        const double* A = input.data() + static_cast<std::size_t>(m * DIM * DIM);
        double* L = output.data() + static_cast<std::size_t>(m * DIM * DIM);
        if (!cholesky_one<DIM>(A, L)) {
            std::cerr << "CPU Cholesky failed at matrix " << m << "\n";
            std::exit(1);
        }
    }
}

template <int DIM>
void cpu_factor_batch_range(
    const std::vector<double>& input, std::vector<double>& output, int begin, int end)
{
    for (int m = begin; m < end; ++m) {
        const double* A = input.data() + static_cast<std::size_t>(m * DIM * DIM);
        double* L = output.data() + static_cast<std::size_t>(m * DIM * DIM);
        if (!cholesky_one<DIM>(A, L)) {
            std::cerr << "CPU threaded Cholesky failed at matrix " << m << "\n";
            std::exit(1);
        }
    }
}

template <int DIM>
void cpu_factor_batch_threaded(
    const std::vector<double>& input, std::vector<double>& output, int count, int num_threads)
{
    output.resize(static_cast<std::size_t>(count * DIM * DIM));
    if (num_threads <= 1 || count <= 1) {
        cpu_factor_batch_range<DIM>(input, output, 0, count);
        return;
    }

    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(num_threads));
    const int chunk = (count + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; ++t) {
        const int begin = t * chunk;
        const int end = std::min(count, begin + chunk);
        if (begin >= end) {
            break;
        }
        workers.emplace_back([&input, &output, begin, end]() {
            cpu_factor_batch_range<DIM>(input, output, begin, end);
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
}

inline int benchmark_thread_count(int count)
{
    const unsigned int hardware_threads = std::thread::hardware_concurrency();
    const int max_threads = hardware_threads == 0u ? 1 : static_cast<int>(hardware_threads);
    constexpr int min_matrices_per_thread = 512;
    return std::max(1, std::min(max_threads, count / min_matrices_per_thread));
}

template <int DIM>
double time_cpu_factor_batch_threaded_us(const std::vector<double>& input,
    std::vector<double>& output, int count, int num_threads, int repeats)
{
    output.resize(static_cast<std::size_t>(count * DIM * DIM));
    const auto start = std::chrono::steady_clock::now();
    if (num_threads <= 1 || count <= 1) {
        for (int r = 0; r < repeats; ++r) {
            cpu_factor_batch_range<DIM>(input, output, 0, count);
        }
    } else {
        std::vector<std::thread> workers;
        workers.reserve(static_cast<std::size_t>(num_threads));
        const int chunk = (count + num_threads - 1) / num_threads;
        for (int t = 0; t < num_threads; ++t) {
            const int begin = t * chunk;
            const int end = std::min(count, begin + chunk);
            if (begin >= end) {
                break;
            }
            workers.emplace_back([&input, &output, begin, end, repeats]() {
                for (int r = 0; r < repeats; ++r) {
                    cpu_factor_batch_range<DIM>(input, output, begin, end);
                }
            });
        }
        for (auto& worker : workers) {
            worker.join();
        }
    }
    consume_for_benchmark(output.back());
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count()
        / static_cast<double>(repeats);
}

template <int DIM>
double max_l_error(const std::vector<double>& expected, const std::vector<double>& actual)
{
    double err = 0.0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        err = std::max(err, std::abs(expected[i] - actual[i]));
    }
    return err;
}

template <int DIM>
double max_reconstruction_error(
    const std::vector<double>& input, const std::vector<double>& factors, int count)
{
    double err = 0.0;
    for (int m = 0; m < count; ++m) {
        const double* A = input.data() + static_cast<std::size_t>(m * DIM * DIM);
        const double* L = factors.data() + static_cast<std::size_t>(m * DIM * DIM);
        for (int r = 0; r < DIM; ++r) {
            for (int c = 0; c < DIM; ++c) {
                double recon = 0.0;
                const int limit = (r < c) ? r : c;
                for (int k = 0; k <= limit; ++k) {
                    recon += L[r * DIM + k] * L[c * DIM + k];
                }
                err = std::max(err, std::abs(A[r * DIM + c] - recon));
            }
        }
    }
    return err;
}

template <int DIM>
float gpu_factor_batch(
    const std::vector<double>& input, std::vector<double>& output, int count, int repeats)
{
    double* d_input = nullptr;
    double* d_output = nullptr;
    int* d_info = nullptr;
    const std::size_t bytes = static_cast<std::size_t>(count * DIM * DIM) * sizeof(double);

    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMalloc(&d_info, static_cast<std::size_t>(count) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));

    const int threads = 128;
    const int blocks = (count + threads - 1) / threads;
    cholesky_batch_kernel<DIM><<<blocks, threads>>>(d_input, d_output, d_info, count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const float total_ms = time_cuda_ms([&]() {
        for (int r = 0; r < repeats; ++r) {
            cholesky_batch_kernel<DIM><<<blocks, threads>>>(d_input, d_output, d_info, count);
            CUDA_CHECK(cudaGetLastError());
        }
    });

    output.resize(static_cast<std::size_t>(count * DIM * DIM));
    std::vector<int> info(static_cast<std::size_t>(count));
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(info.data(), d_info, static_cast<std::size_t>(count) * sizeof(int),
        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_info));

    for (int i = 0; i < count; ++i) {
        if (info[static_cast<std::size_t>(i)] != 0) {
            std::cerr << "GPU Cholesky failed at matrix " << i << "\n";
            std::exit(1);
        }
    }

    return total_ms / static_cast<float>(repeats);
}

template <int DIM> void run_case(int count, int repeats)
{
    std::vector<double> input;
    generate_spd_batch<DIM>(input, count);
    std::vector<double> cpu_output;
    std::vector<double> threaded_output;
    std::vector<double> gpu_output;

    cpu_factor_batch<DIM>(input, cpu_output, count);
    const auto cpu_start = std::chrono::steady_clock::now();
    for (int r = 0; r < repeats; ++r) {
        cpu_factor_batch<DIM>(input, cpu_output, count);
        consume_for_benchmark(cpu_output.back());
    }
    const auto cpu_end = std::chrono::steady_clock::now();
    const double cpu_us = std::chrono::duration<double, std::micro>(cpu_end - cpu_start).count()
        / static_cast<double>(repeats);

    const int num_threads = benchmark_thread_count(count);
    cpu_factor_batch_threaded<DIM>(input, threaded_output, count, num_threads);
    const double threaded_us = time_cpu_factor_batch_threaded_us<DIM>(
        input, threaded_output, count, num_threads, repeats);

    const float gpu_ms = gpu_factor_batch<DIM>(input, gpu_output, count, repeats);
    const double gpu_us = 1000.0 * static_cast<double>(gpu_ms);
    const double l_err = max_l_error<DIM>(cpu_output, gpu_output);
    const double threaded_err = max_l_error<DIM>(cpu_output, threaded_output);
    const double recon_err = max_reconstruction_error<DIM>(input, gpu_output, count);

    std::cout << std::setw(3) << DIM << " " << std::setw(7) << count << " " << std::setw(4)
              << repeats << " " << std::setw(3) << num_threads << " " << std::setw(12) << std::fixed
              << std::setprecision(2) << cpu_us << " " << std::setw(12) << threaded_us << " "
              << std::setw(12) << gpu_us << " " << std::setw(9) << std::setprecision(2)
              << (cpu_us / gpu_us) << " " << std::setw(10) << (threaded_us / gpu_us) << " "
              << std::scientific << std::setprecision(2) << std::setw(11) << l_err << " "
              << std::setw(11) << threaded_err << " " << std::setw(11) << recon_err << "\n";
}

template <int DIM> void run_dimension_suite()
{
    for (const int count : { 1, 16, 256, 4096, 65536 }) {
        const int repeats = (count <= 256) ? 100 : 20;
        run_case<DIM>(count, repeats);
    }
}

} // namespace

int main()
{
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp props {};
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));

    std::cout << "MiniSolver CUDA batched Cholesky microbenchmark\n";
    std::cout << "Device: " << props.name << "\n";
    std::cout << "Metric: device-resident factorization time; host<->device transfers excluded\n";
    std::cout << "Kernel: one CUDA thread factorizes one SPD matrix; baseline only\n";
    std::cout << "DIM   batch    R   T       CPU_us    CPUthr_us       GPU_us   GPU_spd"
                 "  GPU_spd_thr       L_err   Thr_L_err   recon_err\n";

    run_dimension_suite<4>();
    run_dimension_suite<8>();
    run_dimension_suite<12>();
    run_dimension_suite<16>();

    return 0;
}
