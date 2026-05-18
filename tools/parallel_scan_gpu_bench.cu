// Standalone exploratory benchmark for MPX/PCR-style parallel prefix scans.
//
// This is intentionally not wired into Backend::GPU_* yet. MiniSolver's GPU
// backend remains unsupported until a correctness and performance story exists
// beyond this microbenchmark.

#include <cuda_runtime.h>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
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

template <int NX> struct AffineOp {
    double A[NX * NX];
    double b[NX];
};

template <int NX>
__host__ __device__ AffineOp<NX> compose_after(
    const AffineOp<NX>& after, const AffineOp<NX>& before)
{
    AffineOp<NX> result;

    for (int r = 0; r < NX; ++r) {
        for (int c = 0; c < NX; ++c) {
            double acc = 0.0;
            for (int k = 0; k < NX; ++k) {
                acc += after.A[r * NX + k] * before.A[k * NX + c];
            }
            result.A[r * NX + c] = acc;
        }
    }

    for (int r = 0; r < NX; ++r) {
        double acc = after.b[r];
        for (int k = 0; k < NX; ++k) {
            acc += after.A[r * NX + k] * before.b[k];
        }
        result.b[r] = acc;
    }

    return result;
}

template <int NX> struct PrefixCompose {
    __host__ __device__ AffineOp<NX> operator()(
        const AffineOp<NX>& left, const AffineOp<NX>& right) const
    {
        // Binary op for inclusive_scan: left_prefix (+) right_current.
        // The affine prefix at i is op_i o ... o op_0, so (+) means
        // "apply right after left".
        return compose_after<NX>(right, left);
    }
};

template <int NX>
__global__ void pcr_scan_step_kernel(
    const AffineOp<NX>* in, AffineOp<NX>* out, int n, int total, int stride)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const int local = idx % n;
    if (local >= stride) {
        out[idx] = compose_after<NX>(in[idx], in[idx - stride]);
    } else {
        out[idx] = in[idx];
    }
}

template <int NX> std::vector<AffineOp<NX>> make_ops(int n, int batch)
{
    std::vector<AffineOp<NX>> ops(static_cast<std::size_t>(n) * static_cast<std::size_t>(batch));
    std::mt19937 rng(1234 + NX * 17 + n * 3 + batch * 11);
    std::uniform_real_distribution<double> noise(-0.01, 0.01);
    std::uniform_real_distribution<double> bias(-0.02, 0.02);

    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < n; ++i) {
            auto& op = ops[static_cast<std::size_t>(b) * static_cast<std::size_t>(n)
                + static_cast<std::size_t>(i)];
            for (int r = 0; r < NX; ++r) {
                for (int c = 0; c < NX; ++c) {
                    const double diag = (r == c) ? 0.98 : 0.0;
                    op.A[r * NX + c] = diag + noise(rng) / static_cast<double>(NX);
                }
                op.b[r] = bias(rng);
            }
        }
    }
    return ops;
}

template <int NX>
void cpu_scan(const std::vector<AffineOp<NX>>& input, std::vector<AffineOp<NX>>& output, int n)
{
    const std::size_t horizon = static_cast<std::size_t>(n);
    const std::size_t batch = input.size() / horizon;
    for (std::size_t b = 0; b < batch; ++b) {
        const std::size_t base = b * horizon;
        output[base] = input[base];
        for (std::size_t i = 1; i < horizon; ++i) {
            output[base + i] = compose_after<NX>(input[base + i], output[base + i - 1]);
        }
    }
}

template <int NX>
double max_abs_error(
    const std::vector<AffineOp<NX>>& expected, const std::vector<AffineOp<NX>>& actual)
{
    double err = 0.0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        for (int j = 0; j < NX * NX; ++j) {
            err = std::max(err, std::abs(expected[i].A[j] - actual[i].A[j]));
        }
        for (int j = 0; j < NX; ++j) {
            err = std::max(err, std::abs(expected[i].b[j] - actual[i].b[j]));
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
float run_mpx_like_single_scan(
    const std::vector<AffineOp<NX>>& input, std::vector<AffineOp<NX>>& gpu_output, int repeats)
{
    thrust::device_vector<AffineOp<NX>> d_input(input.begin(), input.end());
    thrust::device_vector<AffineOp<NX>> d_output(input.size());

    thrust::inclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), PrefixCompose<NX> {});
    CUDA_CHECK(cudaDeviceSynchronize());

    const float total_ms = time_cuda_ms([&]() {
        for (int r = 0; r < repeats; ++r) {
            thrust::inclusive_scan(
                d_input.begin(), d_input.end(), d_output.begin(), PrefixCompose<NX> {});
        }
    });

    thrust::copy(d_output.begin(), d_output.end(), gpu_output.begin());
    CUDA_CHECK(cudaDeviceSynchronize());
    return total_ms / static_cast<float>(repeats);
}

template <int NX>
float run_mpx_like_segmented_scan(const std::vector<AffineOp<NX>>& input,
    std::vector<AffineOp<NX>>& gpu_output, int n, int repeats)
{
    std::vector<int> keys(input.size());
    for (std::size_t i = 0; i < keys.size(); ++i) {
        keys[i] = static_cast<int>(i / static_cast<std::size_t>(n));
    }
    thrust::device_vector<int> d_keys(keys.begin(), keys.end());
    thrust::device_vector<AffineOp<NX>> d_input(input.begin(), input.end());
    thrust::device_vector<AffineOp<NX>> d_output(input.size());

    thrust::inclusive_scan_by_key(d_keys.begin(), d_keys.end(), d_input.begin(), d_output.begin(),
        thrust::equal_to<int> {}, PrefixCompose<NX> {});
    CUDA_CHECK(cudaDeviceSynchronize());

    const float total_ms = time_cuda_ms([&]() {
        for (int r = 0; r < repeats; ++r) {
            thrust::inclusive_scan_by_key(d_keys.begin(), d_keys.end(), d_input.begin(),
                d_output.begin(), thrust::equal_to<int> {}, PrefixCompose<NX> {});
        }
    });

    thrust::copy(d_output.begin(), d_output.end(), gpu_output.begin());
    CUDA_CHECK(cudaDeviceSynchronize());
    return total_ms / static_cast<float>(repeats);
}

template <int NX>
float run_pcr_like_scan(const std::vector<AffineOp<NX>>& input,
    std::vector<AffineOp<NX>>& gpu_output, int n, int repeats)
{
    thrust::device_vector<AffineOp<NX>> d_input(input.begin(), input.end());
    thrust::device_vector<AffineOp<NX>> d_work(input.size());
    thrust::device_vector<AffineOp<NX>> d_tmp(input.size());
    const int total = static_cast<int>(input.size());
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    auto run_once = [&]() {
        const AffineOp<NX>* const input_ptr = thrust::raw_pointer_cast(d_input.data());
        AffineOp<NX>* const work_ptr = thrust::raw_pointer_cast(d_work.data());
        AffineOp<NX>* const tmp_ptr = thrust::raw_pointer_cast(d_tmp.data());
        const AffineOp<NX>* src = input_ptr;

        for (int stride = 1; stride < n; stride <<= 1) {
            AffineOp<NX>* const dst = (src == work_ptr) ? tmp_ptr : work_ptr;
            pcr_scan_step_kernel<NX><<<blocks, threads>>>(src, dst, n, total, stride);
            CUDA_CHECK(cudaGetLastError());
            src = dst;
        }
        return src;
    };

    const AffineOp<NX>* final_src = run_once();
    CUDA_CHECK(cudaDeviceSynchronize());

    const float total_ms = time_cuda_ms([&]() {
        for (int r = 0; r < repeats; ++r) {
            final_src = run_once();
        }
    });

    CUDA_CHECK(cudaMemcpy(
        gpu_output.data(), final_src, input.size() * sizeof(AffineOp<NX>), cudaMemcpyDeviceToHost));
    return total_ms / static_cast<float>(repeats);
}

template <int NX> void run_case(int n, int batch, int repeats)
{
    const auto input = make_ops<NX>(n, batch);
    std::vector<AffineOp<NX>> cpu_output(input.size());
    std::vector<AffineOp<NX>> mpx_output(input.size());
    std::vector<AffineOp<NX>> pcr_output(input.size());

    cpu_scan<NX>(input, cpu_output, n);
    const auto cpu_start = std::chrono::steady_clock::now();
    for (int r = 0; r < repeats; ++r) {
        cpu_scan<NX>(input, cpu_output, n);
        consume_for_benchmark(cpu_output.back().b[0]);
    }
    const auto cpu_end = std::chrono::steady_clock::now();
    const double cpu_us = std::chrono::duration<double, std::micro>(cpu_end - cpu_start).count()
        / static_cast<double>(repeats);

    const float mpx_ms = run_mpx_like_segmented_scan<NX>(input, mpx_output, n, repeats);
    const float pcr_ms = run_pcr_like_scan<NX>(input, pcr_output, n, repeats);
    const double mpx_us = 1000.0 * static_cast<double>(mpx_ms);
    const double pcr_us = 1000.0 * static_cast<double>(pcr_ms);
    const double mpx_err = max_abs_error<NX>(cpu_output, mpx_output);
    const double pcr_err = max_abs_error<NX>(cpu_output, pcr_output);

    std::cout << std::setw(3) << NX << " " << std::setw(6) << n << " " << std::setw(6) << batch
              << " " << std::setw(4) << repeats << " " << std::setw(11) << std::fixed
              << std::setprecision(2) << cpu_us << " " << std::setw(11) << mpx_us << " "
              << std::setw(11) << pcr_us << " " << std::setw(9) << std::setprecision(2)
              << (cpu_us / mpx_us) << " " << std::setw(9) << (cpu_us / pcr_us) << " "
              << std::scientific << std::setprecision(2) << std::setw(11) << mpx_err << " "
              << std::setw(11) << pcr_err << "\n";
}

template <int NX> void run_dimension_suite()
{
    for (const int n : { 32, 128 }) {
        for (const int batch : { 1, 256, 4096 }) {
            const int repeats = (batch <= 256) ? 20 : 5;
            run_case<NX>(n, batch, repeats);
        }
    }
}

template <int NX> void run_single_horizon_stress_suite()
{
    for (const int n : { 4096, 16384, 65536 }) {
        const auto input = make_ops<NX>(n, 1);
        std::vector<AffineOp<NX>> cpu_output(input.size());
        std::vector<AffineOp<NX>> mpx_output(input.size());
        std::vector<AffineOp<NX>> pcr_output(input.size());
        const int repeats = (n <= 4096) ? 20 : 5;

        cpu_scan<NX>(input, cpu_output, n);
        const auto cpu_start = std::chrono::steady_clock::now();
        for (int r = 0; r < repeats; ++r) {
            cpu_scan<NX>(input, cpu_output, n);
            consume_for_benchmark(cpu_output.back().b[0]);
        }
        const auto cpu_end = std::chrono::steady_clock::now();
        const double cpu_us = std::chrono::duration<double, std::micro>(cpu_end - cpu_start).count()
            / static_cast<double>(repeats);

        const double mpx_us = 1000.0 * run_mpx_like_single_scan<NX>(input, mpx_output, repeats);
        const double pcr_us = 1000.0 * run_pcr_like_scan<NX>(input, pcr_output, n, repeats);
        const double mpx_err = max_abs_error<NX>(cpu_output, mpx_output);
        const double pcr_err = max_abs_error<NX>(cpu_output, pcr_output);

        std::cout << std::setw(3) << NX << " " << std::setw(6) << n << " " << std::setw(6) << 1
                  << " " << std::setw(4) << repeats << " " << std::setw(11) << std::fixed
                  << std::setprecision(2) << cpu_us << " " << std::setw(11) << mpx_us << " "
                  << std::setw(11) << pcr_us << " " << std::setw(9) << std::setprecision(2)
                  << (cpu_us / mpx_us) << " " << std::setw(9) << (cpu_us / pcr_us) << " "
                  << std::scientific << std::setprecision(2) << std::setw(11) << mpx_err << " "
                  << std::setw(11) << pcr_err << "\n";
    }
}

} // namespace

int main()
{
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp props {};
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));

    std::cout << "MiniSolver CUDA prefix-scan microbenchmark\n";
    std::cout << "Device: " << props.name << "\n";
    std::cout << "Metric: device-resident scan time; host<->device transfers excluded\n";
    std::cout << "MPX-like = thrust inclusive_scan, PCR-like = custom Hillis-Steele scan\n";
    std::cout << " NX      N  batch    R      CPU_us      MPX_us      PCR_us  MPX_spd  PCR_spd"
                 "     MPX_err     PCR_err\n";

    std::cout << "# aligned batched route grid\n";
    run_dimension_suite<4>();
    std::cout << "# extended single-horizon scan stress, not used for cross-route gate\n";
    run_single_horizon_stress_suite<4>();
    run_single_horizon_stress_suite<8>();
    run_single_horizon_stress_suite<12>();

    return 0;
}
