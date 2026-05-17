// Standalone exploratory benchmark for block linear-fractional transform scans.
//
// Block Riccati recursions can be represented through linear-fractional
// operators. This benchmark only measures composition of those operators with
// MPX/PCR-style scans; it is intentionally not a full Riccati solver backend.

#include <cuda_runtime.h>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

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

inline void consume_for_benchmark(double value)
{
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "g"(value) : "memory");
#else
    volatile double sink = value;
    (void)sink;
#endif
}

template <int NX> struct BlockLftOp {
    static constexpr int DIM = 2 * NX;
    double M[DIM * DIM];
};

template <int NX>
__host__ __device__ BlockLftOp<NX> compose_after(
    const BlockLftOp<NX>& after, const BlockLftOp<NX>& before)
{
    BlockLftOp<NX> result;
    constexpr int DIM = BlockLftOp<NX>::DIM;
    for (int r = 0; r < DIM; ++r) {
        for (int c = 0; c < DIM; ++c) {
            double acc = 0.0;
            for (int k = 0; k < DIM; ++k) {
                acc += after.M[r * DIM + k] * before.M[k * DIM + c];
            }
            result.M[r * DIM + c] = acc;
        }
    }
    return result;
}

template <int NX> struct LftCompose {
    __host__ __device__ BlockLftOp<NX> operator()(
        const BlockLftOp<NX>& left, const BlockLftOp<NX>& right) const
    {
        return compose_after<NX>(right, left);
    }
};

template <int NX>
__global__ void pcr_scan_step_kernel(
    const BlockLftOp<NX>* in, BlockLftOp<NX>* out, int n, int stride)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    if (idx >= stride) {
        out[idx] = compose_after<NX>(in[idx], in[idx - stride]);
    } else {
        out[idx] = in[idx];
    }
}

template <int NX> std::vector<BlockLftOp<NX>> make_ops(int n)
{
    std::vector<BlockLftOp<NX>> ops(static_cast<std::size_t>(n));
    std::mt19937 rng(9017 + NX * 31 + n);
    std::uniform_real_distribution<double> noise(-1.0e-5, 1.0e-5);
    constexpr int DIM = BlockLftOp<NX>::DIM;

    for (auto& op : ops) {
        for (int r = 0; r < DIM; ++r) {
            for (int c = 0; c < DIM; ++c) {
                op.M[r * DIM + c] = ((r == c) ? 1.0 : 0.0) + noise(rng);
            }
        }
    }
    return ops;
}

template <int NX>
void cpu_scan(const std::vector<BlockLftOp<NX>>& input, std::vector<BlockLftOp<NX>>& output)
{
    output[0] = input[0];
    for (std::size_t i = 1; i < input.size(); ++i) {
        output[i] = compose_after<NX>(input[i], output[i - 1]);
    }
}

template <int NX>
double max_abs_error(
    const std::vector<BlockLftOp<NX>>& expected, const std::vector<BlockLftOp<NX>>& actual)
{
    double err = 0.0;
    constexpr int DIM = BlockLftOp<NX>::DIM;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        for (int j = 0; j < DIM * DIM; ++j) {
            err = std::max(err, std::abs(expected[i].M[j] - actual[i].M[j]));
        }
    }
    return err;
}

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

template <int NX>
float run_mpx_like_scan(
    const std::vector<BlockLftOp<NX>>& input, std::vector<BlockLftOp<NX>>& gpu_output, int repeats)
{
    thrust::device_vector<BlockLftOp<NX>> d_input(input.begin(), input.end());
    thrust::device_vector<BlockLftOp<NX>> d_output(input.size());

    thrust::inclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), LftCompose<NX> {});
    CUDA_CHECK(cudaDeviceSynchronize());

    const float total_ms = time_cuda_ms([&]() {
        for (int r = 0; r < repeats; ++r) {
            thrust::inclusive_scan(
                d_input.begin(), d_input.end(), d_output.begin(), LftCompose<NX> {});
        }
    });

    thrust::copy(d_output.begin(), d_output.end(), gpu_output.begin());
    CUDA_CHECK(cudaDeviceSynchronize());
    return total_ms / static_cast<float>(repeats);
}

template <int NX>
float run_pcr_like_scan(
    const std::vector<BlockLftOp<NX>>& input, std::vector<BlockLftOp<NX>>& gpu_output, int repeats)
{
    thrust::device_vector<BlockLftOp<NX>> d_input(input.begin(), input.end());
    thrust::device_vector<BlockLftOp<NX>> d_work(input.size());
    thrust::device_vector<BlockLftOp<NX>> d_tmp(input.size());
    const int n = static_cast<int>(input.size());
    const int threads = 128;
    const int blocks = (n + threads - 1) / threads;

    auto run_once = [&]() {
        const BlockLftOp<NX>* const input_ptr = thrust::raw_pointer_cast(d_input.data());
        BlockLftOp<NX>* const work_ptr = thrust::raw_pointer_cast(d_work.data());
        BlockLftOp<NX>* const tmp_ptr = thrust::raw_pointer_cast(d_tmp.data());
        const BlockLftOp<NX>* src = input_ptr;

        for (int stride = 1; stride < n; stride <<= 1) {
            BlockLftOp<NX>* const dst = (src == work_ptr) ? tmp_ptr : work_ptr;
            pcr_scan_step_kernel<NX><<<blocks, threads>>>(src, dst, n, stride);
            CUDA_CHECK(cudaGetLastError());
            src = dst;
        }
        return src;
    };

    const BlockLftOp<NX>* final_src = run_once();
    CUDA_CHECK(cudaDeviceSynchronize());

    const float total_ms = time_cuda_ms([&]() {
        for (int r = 0; r < repeats; ++r) {
            final_src = run_once();
        }
    });

    auto final_device = thrust::device_pointer_cast(final_src);
    thrust::copy(final_device, final_device + input.size(), gpu_output.begin());
    CUDA_CHECK(cudaDeviceSynchronize());
    return total_ms / static_cast<float>(repeats);
}

template <int NX> void run_case(int n, int repeats)
{
    const auto input = make_ops<NX>(n);
    std::vector<BlockLftOp<NX>> cpu_output(input.size());
    std::vector<BlockLftOp<NX>> mpx_output(input.size());
    std::vector<BlockLftOp<NX>> pcr_output(input.size());

    cpu_scan<NX>(input, cpu_output);
    const auto cpu_start = std::chrono::steady_clock::now();
    for (int r = 0; r < repeats; ++r) {
        cpu_scan<NX>(input, cpu_output);
        consume_for_benchmark(cpu_output.back().M[0]);
    }
    const auto cpu_end = std::chrono::steady_clock::now();
    const double cpu_us = std::chrono::duration<double, std::micro>(cpu_end - cpu_start).count()
        / static_cast<double>(repeats);

    const double mpx_us = 1000.0 * run_mpx_like_scan<NX>(input, mpx_output, repeats);
    const double pcr_us = 1000.0 * run_pcr_like_scan<NX>(input, pcr_output, repeats);
    const double mpx_err = max_abs_error<NX>(cpu_output, mpx_output);
    const double pcr_err = max_abs_error<NX>(cpu_output, pcr_output);

    std::cout << std::setw(3) << NX << " " << std::setw(6) << n << " " << std::setw(4) << repeats
              << " " << std::setw(12) << std::fixed << std::setprecision(2) << cpu_us << " "
              << std::setw(12) << mpx_us << " " << std::setw(12) << pcr_us << " " << std::setw(8)
              << std::setprecision(2) << (cpu_us / mpx_us) << " " << std::setw(8)
              << (cpu_us / pcr_us) << " " << std::scientific << std::setprecision(2)
              << std::setw(11) << mpx_err << " " << std::setw(11) << pcr_err << "\n";
}

template <int NX> void run_dimension_suite()
{
    for (const int n : { 64, 256, 1024, 4096, 16384 }) {
        const int repeats = (n <= 4096) ? 30 : 10;
        run_case<NX>(n, repeats);
    }
}

} // namespace

int main()
{
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp props {};
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));

    std::cout << "MiniSolver CUDA block-LFT scan microbenchmark\n";
    std::cout << "Device: " << props.name << "\n";
    std::cout << "Metric: device-resident scan time; host<->device transfers excluded\n";
    std::cout << "MPX-like = thrust inclusive_scan, PCR-like = custom Hillis-Steele scan\n";
    std::cout << " NX      N    R      CPU_us      MPX_us      PCR_us  MPX_spd  PCR_spd     MPX_err"
                 "     PCR_err\n";

    run_dimension_suite<2>();
    run_dimension_suite<4>();
    run_dimension_suite<6>();

    return 0;
}
