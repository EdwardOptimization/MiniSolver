// Standalone exploratory benchmark for batched affine block LQR Riccati recursions.
//
// This benchmark is closer to a real Riccati workload than prefix-scan
// microbenchmarks, but it is still not wired into Backend::GPU_*.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
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

template <int NX, int NU> struct LqrProblem {
    static constexpr int NC = NX + NU;

    double A[NX * NX];
    double B[NX * NU];
    double Q[NX * NX];
    double R[NU * NU];
    double H[NU * NX];
    double Qf[NX * NX];
    double q[NX];
    double r[NU];
    double qf[NX];
    double C[NC * NX];
    double D[NC * NU];
    double sigma[NC];
    double grad[NC];
};

template <int NX> constexpr int riccati_output_size()
{
    return NX * NX + NX;
}

template <int NX, int NU> LqrProblem<NX, NU> make_problem()
{
    LqrProblem<NX, NU> p {};
    for (int r = 0; r < NX; ++r) {
        for (int c = 0; c < NX; ++c) {
            const double diag = (r == c) ? 0.92 : 0.0;
            const double offdiag = (r != c && ((r + 2 * c) % 5 == 0)) ? 0.01 : 0.0;
            p.A[r * NX + c] = diag + offdiag;
            p.Q[r * NX + c] = (r == c) ? (1.0 + 0.05 * static_cast<double>(r)) : 0.0;
            p.Qf[r * NX + c] = (r == c) ? (2.0 + 0.05 * static_cast<double>(r)) : 0.0;
        }
        for (int c = 0; c < NU; ++c) {
            p.B[r * NU + c] = ((r + c) % 3 == 0) ? (0.08 / static_cast<double>(1 + c)) : 0.015;
        }
        p.q[r] = 0.01 * static_cast<double>(r + 1);
        p.qf[r] = 0.02 * static_cast<double>(r + 1);
    }
    for (int r = 0; r < NU; ++r) {
        for (int c = 0; c < NU; ++c) {
            p.R[r * NU + c] = (r == c) ? (0.5 + 0.1 * static_cast<double>(r)) : 0.0;
        }
        for (int c = 0; c < NX; ++c) {
            p.H[r * NX + c] = ((2 * r + c) % 5 == 0) ? 0.01 : 0.0;
        }
        p.r[r] = -0.005 * static_cast<double>(r + 1);
    }

    for (int row = 0; row < LqrProblem<NX, NU>::NC; ++row) {
        p.sigma[row] = 0.02 + 0.003 * static_cast<double>((row % 5) + 1);
        p.grad[row] = 0.001 * static_cast<double>((row % 3) - 1);
        for (int col = 0; col < NX; ++col) {
            p.C[row * NX + col] = ((row + 2 * col) % 4 == 0) ? (0.10 / (1.0 + col)) : 0.0;
        }
        for (int col = 0; col < NU; ++col) {
            p.D[row * NU + col] = ((2 * row + col) % 5 == 0) ? (0.04 / (1.0 + col)) : 0.0;
        }
    }
    return p;
}

template <int NX, int NU>
__host__ __device__ void build_barrier_packet(const LqrProblem<NX, NU>& p, double* Qbar,
    double* Rbar, double* Hbar, double* qbar, double* rbar)
{
    constexpr int NC = LqrProblem<NX, NU>::NC;

    for (int r = 0; r < NX; ++r) {
        for (int c = 0; c < NX; ++c) {
            Qbar[r * NX + c] = p.Q[r * NX + c];
        }
        qbar[r] = p.q[r];
    }
    for (int r = 0; r < NU; ++r) {
        for (int c = 0; c < NU; ++c) {
            Rbar[r * NU + c] = p.R[r * NU + c];
        }
        for (int c = 0; c < NX; ++c) {
            Hbar[r * NX + c] = p.H[r * NX + c];
        }
        rbar[r] = p.r[r];
    }

    for (int row = 0; row < NC; ++row) {
        const double weight = p.sigma[row];
        const double grad = p.grad[row];
        for (int r = 0; r < NX; ++r) {
            const double Cr = p.C[row * NX + r];
            qbar[r] += Cr * grad;
            for (int c = 0; c < NX; ++c) {
                Qbar[r * NX + c] += Cr * weight * p.C[row * NX + c];
            }
        }
        for (int r = 0; r < NU; ++r) {
            const double Dr = p.D[row * NU + r];
            rbar[r] += Dr * grad;
            for (int c = 0; c < NU; ++c) {
                Rbar[r * NU + c] += Dr * weight * p.D[row * NU + c];
            }
            for (int c = 0; c < NX; ++c) {
                Hbar[r * NX + c] += Dr * weight * p.C[row * NX + c];
            }
        }
    }
}

template <int N> __host__ __device__ bool cholesky_lower(const double* A, double* L)
{
    for (int i = 0; i < N * N; ++i) {
        L[i] = 0.0;
    }
    for (int j = 0; j < N; ++j) {
        double diag = A[j * N + j];
        for (int k = 0; k < j; ++k) {
            const double v = L[j * N + k];
            diag -= v * v;
        }
        if (!(diag > 0.0)) {
            return false;
        }
        L[j * N + j] = sqrt(diag);

        for (int i = j + 1; i < N; ++i) {
            double acc = A[i * N + j];
            for (int k = 0; k < j; ++k) {
                acc -= L[i * N + k] * L[j * N + k];
            }
            L[i * N + j] = acc / L[j * N + j];
        }
    }
    return true;
}

template <int NX, int NU>
__host__ __device__ bool solve_spd_multi_rhs(const double* S, const double* G, double* K)
{
    double L[NU * NU];
    if (!cholesky_lower<NU>(S, L)) {
        return false;
    }

    for (int col = 0; col < NX; ++col) {
        double y[NU];
        for (int i = 0; i < NU; ++i) {
            double acc = G[i * NX + col];
            for (int j = 0; j < i; ++j) {
                acc -= L[i * NU + j] * y[j];
            }
            y[i] = acc / L[i * NU + i];
        }
        for (int i = NU - 1; i >= 0; --i) {
            double acc = y[i];
            for (int j = i + 1; j < NU; ++j) {
                acc -= L[j * NU + i] * K[j * NX + col];
            }
            K[i * NX + col] = acc / L[i * NU + i];
        }
    }
    return true;
}

template <int NU>
__host__ __device__ bool solve_spd_vector(const double* S, const double* rhs, double* x)
{
    double L[NU * NU];
    if (!cholesky_lower<NU>(S, L)) {
        return false;
    }

    double y[NU];
    for (int i = 0; i < NU; ++i) {
        double acc = rhs[i];
        for (int j = 0; j < i; ++j) {
            acc -= L[i * NU + j] * y[j];
        }
        y[i] = acc / L[i * NU + i];
    }
    for (int i = NU - 1; i >= 0; --i) {
        double acc = y[i];
        for (int j = i + 1; j < NU; ++j) {
            acc -= L[j * NU + i] * x[j];
        }
        x[i] = acc / L[i * NU + i];
    }
    return true;
}

template <int NX, int NU>
__host__ __device__ bool riccati_one(const LqrProblem<NX, NU>& p, int horizon, double* output)
{
    double P[NX * NX];
    double pvec[NX];
    double PB[NX * NU];
    double PA[NX * NX];
    double Qbar[NX * NX];
    double Rbar[NU * NU];
    double Hbar[NU * NX];
    double qbar[NX];
    double rbar[NU];
    double S[NU * NU];
    double G[NU * NX];
    double g[NU];
    double K[NU * NX];
    double kff[NU];
    double AtPA[NX * NX];
    double Atp[NX];
    double nextP[NX * NX];
    double nextp[NX];

    for (int i = 0; i < NX * NX; ++i) {
        P[i] = p.Qf[i];
    }
    for (int i = 0; i < NX; ++i) {
        pvec[i] = p.qf[i];
    }

    for (int step = 0; step < horizon; ++step) {
        build_barrier_packet<NX, NU>(p, Qbar, Rbar, Hbar, qbar, rbar);

        for (int r = 0; r < NX; ++r) {
            for (int c = 0; c < NU; ++c) {
                double acc = 0.0;
                for (int k = 0; k < NX; ++k) {
                    acc += P[r * NX + k] * p.B[k * NU + c];
                }
                PB[r * NU + c] = acc;
            }
            for (int c = 0; c < NX; ++c) {
                double acc = 0.0;
                for (int k = 0; k < NX; ++k) {
                    acc += P[r * NX + k] * p.A[k * NX + c];
                }
                PA[r * NX + c] = acc;
            }
        }

        for (int r = 0; r < NU; ++r) {
            for (int c = 0; c < NU; ++c) {
                double acc = Rbar[r * NU + c];
                for (int k = 0; k < NX; ++k) {
                    acc += p.B[k * NU + r] * PB[k * NU + c];
                }
                S[r * NU + c] = acc;
            }
            for (int c = 0; c < NX; ++c) {
                double acc = Hbar[r * NX + c];
                for (int k = 0; k < NX; ++k) {
                    acc += p.B[k * NU + r] * PA[k * NX + c];
                }
                G[r * NX + c] = acc;
            }
            double g_acc = rbar[r];
            for (int k = 0; k < NX; ++k) {
                g_acc += p.B[k * NU + r] * pvec[k];
            }
            g[r] = g_acc;
        }

        for (int r = 0; r < NX; ++r) {
            for (int c = 0; c < NX; ++c) {
                double acc = 0.0;
                for (int k = 0; k < NX; ++k) {
                    acc += p.A[k * NX + r] * PA[k * NX + c];
                }
                AtPA[r * NX + c] = acc;
            }
            double p_acc = 0.0;
            for (int k = 0; k < NX; ++k) {
                p_acc += p.A[k * NX + r] * pvec[k];
            }
            Atp[r] = p_acc;
        }

        if (!solve_spd_multi_rhs<NX, NU>(S, G, K)) {
            return false;
        }
        if (!solve_spd_vector<NU>(S, g, kff)) {
            return false;
        }

        for (int r = 0; r < NX; ++r) {
            for (int c = 0; c < NX; ++c) {
                double feedback = 0.0;
                for (int u = 0; u < NU; ++u) {
                    feedback += G[u * NX + r] * K[u * NX + c];
                }
                nextP[r * NX + c] = Qbar[r * NX + c] + AtPA[r * NX + c] - feedback;
            }
            double affine_feedback = 0.0;
            for (int u = 0; u < NU; ++u) {
                affine_feedback += G[u * NX + r] * kff[u];
            }
            nextp[r] = qbar[r] + Atp[r] - affine_feedback;
        }

        for (int r = 0; r < NX; ++r) {
            for (int c = 0; c < NX; ++c) {
                const double sym = 0.5 * (nextP[r * NX + c] + nextP[c * NX + r]);
                P[r * NX + c] = sym;
            }
            pvec[r] = nextp[r];
        }
    }

    for (int i = 0; i < NX * NX; ++i) {
        output[i] = P[i];
    }
    for (int i = 0; i < NX; ++i) {
        output[NX * NX + i] = pvec[i];
    }
    return true;
}

template <int NX, int NU>
__global__ void riccati_batch_kernel(
    LqrProblem<NX, NU> problem, int horizon, double* output, int* info, int count)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    constexpr int output_size = NX * NX + NX;
    double* out = output + static_cast<std::size_t>(idx * output_size);
    info[idx] = riccati_one<NX, NU>(problem, horizon, out) ? 0 : 1;
}

template <int NX, int NU>
void cpu_riccati_range(
    const LqrProblem<NX, NU>& problem, int horizon, std::vector<double>& output, int begin, int end)
{
    for (int idx = begin; idx < end; ++idx) {
        double* out = output.data() + static_cast<std::size_t>(idx * riccati_output_size<NX>());
        if (!riccati_one<NX, NU>(problem, horizon, out)) {
            std::cerr << "CPU Riccati failed at problem " << idx << "\n";
            std::exit(1);
        }
    }
}

inline int benchmark_thread_count(int count)
{
    const unsigned int hardware_threads = std::thread::hardware_concurrency();
    const int max_threads = hardware_threads == 0u ? 1 : static_cast<int>(hardware_threads);
    constexpr int min_problems_per_thread = 128;
    return std::max(1, std::min(max_threads, count / min_problems_per_thread));
}

template <int NX, int NU>
void cpu_riccati_batch(
    const LqrProblem<NX, NU>& problem, int horizon, std::vector<double>& output, int count)
{
    output.resize(static_cast<std::size_t>(count * riccati_output_size<NX>()));
    cpu_riccati_range<NX, NU>(problem, horizon, output, 0, count);
}

template <int NX, int NU>
double time_cpu_threaded_us(const LqrProblem<NX, NU>& problem, int horizon,
    std::vector<double>& output, int count, int num_threads, int repeats)
{
    output.resize(static_cast<std::size_t>(count * riccati_output_size<NX>()));
    const auto start = std::chrono::steady_clock::now();
    if (num_threads <= 1 || count <= 1) {
        for (int r = 0; r < repeats; ++r) {
            cpu_riccati_range<NX, NU>(problem, horizon, output, 0, count);
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
            workers.emplace_back([&problem, horizon, &output, begin, end, repeats]() {
                for (int r = 0; r < repeats; ++r) {
                    cpu_riccati_range<NX, NU>(problem, horizon, output, begin, end);
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

template <int NX, int NU>
float gpu_riccati_batch(const LqrProblem<NX, NU>& problem, int horizon, std::vector<double>& output,
    int count, int repeats)
{
    double* d_output = nullptr;
    int* d_info = nullptr;
    const std::size_t output_bytes
        = static_cast<std::size_t>(count * riccati_output_size<NX>()) * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_output, output_bytes));
    CUDA_CHECK(cudaMalloc(&d_info, static_cast<std::size_t>(count) * sizeof(int)));

    const int threads = 128;
    const int blocks = (count + threads - 1) / threads;
    riccati_batch_kernel<NX, NU><<<blocks, threads>>>(problem, horizon, d_output, d_info, count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const float total_ms = time_cuda_ms([&]() {
        for (int r = 0; r < repeats; ++r) {
            riccati_batch_kernel<NX, NU>
                <<<blocks, threads>>>(problem, horizon, d_output, d_info, count);
            CUDA_CHECK(cudaGetLastError());
        }
    });

    output.resize(static_cast<std::size_t>(count * riccati_output_size<NX>()));
    std::vector<int> info(static_cast<std::size_t>(count));
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, output_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(info.data(), d_info, static_cast<std::size_t>(count) * sizeof(int),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_info));

    for (int i = 0; i < count; ++i) {
        if (info[static_cast<std::size_t>(i)] != 0) {
            std::cerr << "GPU Riccati failed at problem " << i << "\n";
            std::exit(1);
        }
    }
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

template <int NX, int NU> void run_case(int horizon, int count, int repeats)
{
    const LqrProblem<NX, NU> problem = make_problem<NX, NU>();
    std::vector<double> cpu_output;
    std::vector<double> threaded_output;
    std::vector<double> gpu_output;

    cpu_riccati_batch<NX, NU>(problem, horizon, cpu_output, count);
    const auto cpu_start = std::chrono::steady_clock::now();
    for (int r = 0; r < repeats; ++r) {
        cpu_riccati_batch<NX, NU>(problem, horizon, cpu_output, count);
        consume_for_benchmark(cpu_output.back());
    }
    const auto cpu_end = std::chrono::steady_clock::now();
    const double cpu_us = std::chrono::duration<double, std::micro>(cpu_end - cpu_start).count()
        / static_cast<double>(repeats);

    const int num_threads = benchmark_thread_count(count);
    const double threaded_us = time_cpu_threaded_us<NX, NU>(
        problem, horizon, threaded_output, count, num_threads, repeats);
    const float gpu_ms = gpu_riccati_batch<NX, NU>(problem, horizon, gpu_output, count, repeats);
    const double gpu_us = 1000.0 * static_cast<double>(gpu_ms);
    const double gpu_err = max_abs_error(cpu_output, gpu_output);
    const double threaded_err = max_abs_error(cpu_output, threaded_output);

    std::cout << std::setw(3) << NX << " " << std::setw(3) << NU << " " << std::setw(4) << horizon
              << " " << std::setw(7) << count << " " << std::setw(4) << repeats << " "
              << std::setw(3) << num_threads << " " << std::setw(12) << std::fixed
              << std::setprecision(2) << cpu_us << " " << std::setw(12) << threaded_us << " "
              << std::setw(12) << gpu_us << " " << std::setw(9) << std::setprecision(2)
              << (threaded_us / gpu_us) << " " << std::scientific << std::setprecision(2)
              << std::setw(11) << gpu_err << " " << std::setw(11) << threaded_err << "\n";
}

template <int NX, int NU> void run_suite()
{
    for (const int horizon : { 32, 128 }) {
        for (const int count : { 1, 256, 4096, 65536 }) {
            const int repeats = (count <= 256) ? 20 : 5;
            run_case<NX, NU>(horizon, count, repeats);
        }
    }
}

} // namespace

int main()
{
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp props {};
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));

    std::cout << "MiniSolver CUDA batched barrier-affine block Riccati microbenchmark\n";
    std::cout << "Device: " << props.name << "\n";
    std::cout << "Metric: device-resident Riccati direction recursion time; host<->device "
                 "transfers excluded\n";
    std::cout << "Packet: synthetic C/D/sigma/grad barrier contribution folded into Q/R/H/q/r\n";
    std::cout << "GPU kernel: one CUDA thread solves one independent LQR horizon\n";
    std::cout << " NX  NU    N   batch    R   T       CPU_us    CPUthr_us       GPU_us  GPU_spdT"
                 "     GPU_err     Thr_err\n";

    run_suite<4, 2>();
    run_suite<8, 4>();

    return 0;
}
