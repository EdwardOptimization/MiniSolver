// Standalone exploratory benchmark for scalar Riccati MPX/PCR-style scans.
//
// The scalar discrete Riccati backward recursion can be written as a sequence
// of fractional-linear transforms. This benchmark scans those transforms on CPU
// and GPU to test the MPX/PCR idea closer to Riccati than a generic affine scan.

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

struct RiccatiStage {
    double a;
    double b;
    double q;
    double r;
};

struct MobiusOp {
    double a;
    double b;
    double c;
    double d;
};

inline void consume_for_benchmark(double value)
{
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "g"(value) : "memory");
#else
    volatile double sink = value;
    (void)sink;
#endif
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

__host__ __device__ MobiusOp normalize(MobiusOp op)
{
    double scale = fabs(op.a);
    scale = fmax(scale, fabs(op.b));
    scale = fmax(scale, fabs(op.c));
    scale = fmax(scale, fabs(op.d));
    if (scale > 0.0) {
        op.a /= scale;
        op.b /= scale;
        op.c /= scale;
        op.d /= scale;
    }
    return op;
}

__host__ __device__ MobiusOp compose_after(const MobiusOp& after, const MobiusOp& before)
{
    MobiusOp result {
        after.a * before.a + after.b * before.c,
        after.a * before.b + after.b * before.d,
        after.c * before.a + after.d * before.c,
        after.c * before.b + after.d * before.d,
    };
    return normalize(result);
}

__host__ __device__ double apply(const MobiusOp& op, double x)
{
    return (op.a * x + op.b) / (op.c * x + op.d);
}

MobiusOp stage_to_op(const RiccatiStage& s)
{
    // P_k = q + a^2 r P_{k+1} / (r + b^2 P_{k+1})
    //     = ((q b^2 + a^2 r) P + q r) / (b^2 P + r)
    return normalize(MobiusOp {
        s.q * s.b * s.b + s.a * s.a * s.r,
        s.q * s.r,
        s.b * s.b,
        s.r,
    });
}

struct PrefixCompose {
    __host__ __device__ MobiusOp operator()(const MobiusOp& left, const MobiusOp& right) const
    {
        return compose_after(right, left);
    }
};

__global__ void pcr_scan_step_kernel(const MobiusOp* in, MobiusOp* out, int n, int stride)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    out[idx] = (idx >= stride) ? compose_after(in[idx], in[idx - stride]) : in[idx];
}

std::vector<RiccatiStage> make_stages(int n)
{
    std::vector<RiccatiStage> stages(static_cast<std::size_t>(n));
    std::mt19937 rng(777 + n);
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
    return stages;
}

std::vector<MobiusOp> make_reversed_ops(const std::vector<RiccatiStage>& stages)
{
    std::vector<MobiusOp> ops(stages.size());
    for (std::size_t i = 0; i < stages.size(); ++i) {
        ops[i] = stage_to_op(stages[stages.size() - 1 - i]);
    }
    return ops;
}

std::vector<double> cpu_sequential_riccati(const std::vector<RiccatiStage>& stages, double terminal)
{
    std::vector<double> P(stages.size() + 1);
    P.back() = terminal;
    for (int k = static_cast<int>(stages.size()) - 1; k >= 0; --k) {
        const RiccatiStage& s = stages[static_cast<std::size_t>(k)];
        const double next = P[static_cast<std::size_t>(k + 1)];
        P[static_cast<std::size_t>(k)] = s.q + (s.a * s.a * s.r * next) / (s.r + s.b * s.b * next);
    }
    return P;
}

std::vector<MobiusOp> cpu_scan_ops(const std::vector<MobiusOp>& input)
{
    std::vector<MobiusOp> output(input.size());
    output[0] = input[0];
    for (std::size_t i = 1; i < input.size(); ++i) {
        output[i] = compose_after(input[i], output[i - 1]);
    }
    return output;
}

std::vector<double> prefix_to_riccati(
    const std::vector<MobiusOp>& reversed_prefix, int n, double terminal)
{
    std::vector<double> P(static_cast<std::size_t>(n + 1));
    P[static_cast<std::size_t>(n)] = terminal;
    for (int i = 0; i < n; ++i) {
        P[static_cast<std::size_t>(n - 1 - i)]
            = apply(reversed_prefix[static_cast<std::size_t>(i)], terminal);
    }
    return P;
}

double max_abs_error(const std::vector<double>& expected, const std::vector<double>& actual)
{
    double err = 0.0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        err = std::max(err, std::abs(expected[i] - actual[i]));
    }
    return err;
}

float run_mpx_like_scan(
    const std::vector<MobiusOp>& input, std::vector<MobiusOp>& output, int repeats)
{
    thrust::device_vector<MobiusOp> d_input(input.begin(), input.end());
    thrust::device_vector<MobiusOp> d_output(input.size());

    thrust::inclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), PrefixCompose {});
    CUDA_CHECK(cudaDeviceSynchronize());

    const float total_ms = time_cuda_ms([&]() {
        for (int r = 0; r < repeats; ++r) {
            thrust::inclusive_scan(
                d_input.begin(), d_input.end(), d_output.begin(), PrefixCompose {});
        }
    });

    thrust::copy(d_output.begin(), d_output.end(), output.begin());
    CUDA_CHECK(cudaDeviceSynchronize());
    return total_ms / static_cast<float>(repeats);
}

float run_pcr_like_scan(
    const std::vector<MobiusOp>& input, std::vector<MobiusOp>& output, int repeats)
{
    thrust::device_vector<MobiusOp> d_input(input.begin(), input.end());
    thrust::device_vector<MobiusOp> d_work(input.size());
    thrust::device_vector<MobiusOp> d_tmp(input.size());
    const int n = static_cast<int>(input.size());
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    auto run_once = [&]() {
        const MobiusOp* const input_ptr = thrust::raw_pointer_cast(d_input.data());
        MobiusOp* const work_ptr = thrust::raw_pointer_cast(d_work.data());
        MobiusOp* const tmp_ptr = thrust::raw_pointer_cast(d_tmp.data());
        const MobiusOp* src = input_ptr;

        for (int stride = 1; stride < n; stride <<= 1) {
            MobiusOp* const dst = (src == work_ptr) ? tmp_ptr : work_ptr;
            pcr_scan_step_kernel<<<blocks, threads>>>(src, dst, n, stride);
            CUDA_CHECK(cudaGetLastError());
            src = dst;
        }
        return src;
    };

    const MobiusOp* final_src = run_once();
    CUDA_CHECK(cudaDeviceSynchronize());

    const float total_ms = time_cuda_ms([&]() {
        for (int r = 0; r < repeats; ++r) {
            final_src = run_once();
        }
    });

    CUDA_CHECK(cudaMemcpy(
        output.data(), final_src, input.size() * sizeof(MobiusOp), cudaMemcpyDeviceToHost));
    return total_ms / static_cast<float>(repeats);
}

void run_case(int n, int repeats)
{
    constexpr double terminal_cost = 1.0;
    const auto stages = make_stages(n);
    const auto reversed_ops = make_reversed_ops(stages);
    const auto sequential = cpu_sequential_riccati(stages, terminal_cost);
    std::vector<MobiusOp> cpu_prefix(static_cast<std::size_t>(n));
    std::vector<MobiusOp> mpx_prefix(static_cast<std::size_t>(n));
    std::vector<MobiusOp> pcr_prefix(static_cast<std::size_t>(n));

    cpu_prefix = cpu_scan_ops(reversed_ops);
    const auto cpu_start = std::chrono::steady_clock::now();
    for (int r = 0; r < repeats; ++r) {
        cpu_prefix = cpu_scan_ops(reversed_ops);
        consume_for_benchmark(cpu_prefix.back().a);
    }
    const auto cpu_end = std::chrono::steady_clock::now();
    const double cpu_us = std::chrono::duration<double, std::micro>(cpu_end - cpu_start).count()
        / static_cast<double>(repeats);

    const float mpx_ms = run_mpx_like_scan(reversed_ops, mpx_prefix, repeats);
    const float pcr_ms = run_pcr_like_scan(reversed_ops, pcr_prefix, repeats);
    const auto cpu_scan_solution = prefix_to_riccati(cpu_prefix, n, terminal_cost);
    const auto mpx_solution = prefix_to_riccati(mpx_prefix, n, terminal_cost);
    const auto pcr_solution = prefix_to_riccati(pcr_prefix, n, terminal_cost);

    const double cpu_err = max_abs_error(sequential, cpu_scan_solution);
    const double mpx_err = max_abs_error(sequential, mpx_solution);
    const double pcr_err = max_abs_error(sequential, pcr_solution);
    const double mpx_us = 1000.0 * static_cast<double>(mpx_ms);
    const double pcr_us = 1000.0 * static_cast<double>(pcr_ms);

    std::cout << std::setw(7) << n << " " << std::setw(4) << repeats << " " << std::setw(11)
              << std::fixed << std::setprecision(2) << cpu_us << " " << std::setw(11) << mpx_us
              << " " << std::setw(11) << pcr_us << " " << std::setw(9) << std::setprecision(2)
              << (cpu_us / mpx_us) << " " << std::setw(9) << (cpu_us / pcr_us) << " "
              << std::scientific << std::setprecision(2) << std::setw(11) << cpu_err << " "
              << std::setw(11) << mpx_err << " " << std::setw(11) << pcr_err << "\n";
}

} // namespace

int main()
{
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp props {};
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));

    std::cout << "MiniSolver CUDA scalar Riccati scan microbenchmark\n";
    std::cout << "Device: " << props.name << "\n";
    std::cout << "Metric: device-resident scan time; host<->device transfers excluded\n";
    std::cout << "MPX-like = thrust inclusive_scan, PCR-like = custom Hillis-Steele scan\n";
    std::cout << "      N    R      CPU_us      MPX_us      PCR_us  MPX_spd  PCR_spd     CPU_err"
                 "     MPX_err     PCR_err\n";

    for (const int n : { 64, 256, 1024, 4096, 16384, 65536 }) {
        const int repeats = (n <= 4096) ? 50 : 10;
        run_case(n, repeats);
    }

    return 0;
}
