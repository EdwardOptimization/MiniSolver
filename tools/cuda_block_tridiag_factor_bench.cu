// Standalone exploratory benchmark for explicit block-tridiagonal factorization.
//
// This route probes a sparse/block alternative to assembling the full KKT as a
// dense matrix. It factors synthetic block-tridiagonal SPD systems that can be
// interpreted as a regularized normal-equation/Schur-complement view of an OCP
// Newton step. It is intentionally not a Riccati backend.

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

struct CaseConfig {
    int block_dim;
    int horizon;
    int batch;
    int repeats;
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

template <int BS>
void build_block_tridiag(std::vector<double>& diag, std::vector<double>& lower,
    std::vector<double>& rhs, int horizon, int batch)
{
    diag.assign(static_cast<std::size_t>(batch * horizon * BS * BS), 0.0);
    lower.assign(static_cast<std::size_t>(batch * (horizon - 1) * BS * BS), 0.0);
    rhs.assign(static_cast<std::size_t>(batch * horizon * BS), 0.0);

    for (int sample = 0; sample < batch; ++sample) {
        for (int k = 0; k < horizon; ++k) {
            double* D = diag.data() + static_cast<std::size_t>((sample * horizon + k) * BS * BS);
            for (int r = 0; r < BS; ++r) {
                for (int c = 0; c < BS; ++c) {
                    D[r * BS + c] = (r == c)
                        ? 3.0 + 0.02 * static_cast<double>((r + k + sample) % 7)
                        : 0.01 / static_cast<double>(1 + std::abs(r - c));
                }
                rhs[static_cast<std::size_t>((sample * horizon + k) * BS + r)]
                    = 0.05 + 0.001 * static_cast<double>((r + 3 * k + sample) % 19);
            }
        }
        for (int k = 1; k < horizon; ++k) {
            double* L = lower.data()
                + static_cast<std::size_t>((sample * (horizon - 1) + (k - 1)) * BS * BS);
            for (int r = 0; r < BS; ++r) {
                for (int c = 0; c < BS; ++c) {
                    const double diag_like = (r == c) ? -0.12 : 0.0;
                    const double sparse_like = ((r + 2 * c + k) % 5 == 0) ? 0.01 : 0.0;
                    L[r * BS + c] = diag_like + sparse_like;
                }
            }
        }
    }
}

template <int BS> __host__ __device__ bool chol_lower_inplace(double* A)
{
    for (int j = 0; j < BS; ++j) {
        double diag = A[j * BS + j];
        for (int p = 0; p < j; ++p) {
            const double v = A[j * BS + p];
            diag -= v * v;
        }
        if (!(diag > 0.0)) {
            return false;
        }
        A[j * BS + j] = sqrt(diag);
        for (int i = j + 1; i < BS; ++i) {
            double acc = A[i * BS + j];
            for (int p = 0; p < j; ++p) {
                acc -= A[i * BS + p] * A[j * BS + p];
            }
            A[i * BS + j] = acc / A[j * BS + j];
        }
        for (int c = j + 1; c < BS; ++c) {
            A[j * BS + c] = 0.0;
        }
    }
    return true;
}

template <int BS> __host__ __device__ void solve_lower(const double* L, double* x)
{
    for (int i = 0; i < BS; ++i) {
        double acc = x[i];
        for (int j = 0; j < i; ++j) {
            acc -= L[i * BS + j] * x[j];
        }
        x[i] = acc / L[i * BS + i];
    }
}

template <int BS> __host__ __device__ void solve_lower_transpose(const double* L, double* x)
{
    for (int i = BS - 1; i >= 0; --i) {
        double acc = x[i];
        for (int j = i + 1; j < BS; ++j) {
            acc -= L[j * BS + i] * x[j];
        }
        x[i] = acc / L[i * BS + i];
    }
}

template <int BS>
__host__ __device__ void right_solve_lower_transpose(
    const double* B, const double* Lprev, double* X)
{
    for (int r = 0; r < BS; ++r) {
        for (int c = 0; c < BS; ++c) {
            double acc = B[r * BS + c];
            for (int p = 0; p < c; ++p) {
                acc -= X[r * BS + p] * Lprev[c * BS + p];
            }
            X[r * BS + c] = acc / Lprev[c * BS + c];
        }
    }
}

template <int BS>
__host__ __device__ bool block_tridiag_solve_one(const double* diag, const double* lower,
    const double* rhs, double* solution, double* fact_diag, double* fact_lower, int horizon)
{
    for (int i = 0; i < horizon * BS * BS; ++i) {
        fact_diag[i] = diag[i];
    }

    if (!chol_lower_inplace<BS>(fact_diag)) {
        return false;
    }

    for (int k = 1; k < horizon; ++k) {
        const double* Lprev = fact_diag + static_cast<std::size_t>((k - 1) * BS * BS);
        const double* B = lower + static_cast<std::size_t>((k - 1) * BS * BS);
        double* X = fact_lower + static_cast<std::size_t>((k - 1) * BS * BS);
        double* D = fact_diag + static_cast<std::size_t>(k * BS * BS);
        right_solve_lower_transpose<BS>(B, Lprev, X);

        for (int r = 0; r < BS; ++r) {
            for (int c = 0; c <= r; ++c) {
                double update = 0.0;
                for (int p = 0; p < BS; ++p) {
                    update += X[r * BS + p] * X[c * BS + p];
                }
                D[r * BS + c] -= update;
                D[c * BS + r] = D[r * BS + c];
            }
        }
        if (!chol_lower_inplace<BS>(D)) {
            return false;
        }
    }

    for (int i = 0; i < horizon * BS; ++i) {
        solution[i] = rhs[i];
    }
    solve_lower<BS>(fact_diag, solution);
    for (int k = 1; k < horizon; ++k) {
        double* y = solution + static_cast<std::size_t>(k * BS);
        const double* yprev = solution + static_cast<std::size_t>((k - 1) * BS);
        const double* X = fact_lower + static_cast<std::size_t>((k - 1) * BS * BS);
        for (int r = 0; r < BS; ++r) {
            double acc = y[r];
            for (int c = 0; c < BS; ++c) {
                acc -= X[r * BS + c] * yprev[c];
            }
            y[r] = acc;
        }
        solve_lower<BS>(fact_diag + static_cast<std::size_t>(k * BS * BS), y);
    }

    solve_lower_transpose<BS>(fact_diag + static_cast<std::size_t>((horizon - 1) * BS * BS),
        solution + static_cast<std::size_t>((horizon - 1) * BS));
    for (int k = horizon - 2; k >= 0; --k) {
        double* x = solution + static_cast<std::size_t>(k * BS);
        const double* xnext = solution + static_cast<std::size_t>((k + 1) * BS);
        const double* Xnext = fact_lower + static_cast<std::size_t>(k * BS * BS);
        for (int r = 0; r < BS; ++r) {
            double acc = x[r];
            for (int c = 0; c < BS; ++c) {
                acc -= Xnext[c * BS + r] * xnext[c];
            }
            x[r] = acc;
        }
        solve_lower_transpose<BS>(fact_diag + static_cast<std::size_t>(k * BS * BS), x);
    }

    return true;
}

template <int BS>
void cpu_solve_range(const std::vector<double>& diag, const std::vector<double>& lower,
    const std::vector<double>& rhs, std::vector<double>& solution, int horizon, int begin, int end)
{
    std::vector<double> fact_diag(static_cast<std::size_t>(horizon * BS * BS));
    std::vector<double> fact_lower(static_cast<std::size_t>((horizon - 1) * BS * BS));
    for (int sample = begin; sample < end; ++sample) {
        const double* D = diag.data() + static_cast<std::size_t>(sample * horizon * BS * BS);
        const double* L = lower.data() + static_cast<std::size_t>(sample * (horizon - 1) * BS * BS);
        const double* b = rhs.data() + static_cast<std::size_t>(sample * horizon * BS);
        double* x = solution.data() + static_cast<std::size_t>(sample * horizon * BS);
        if (!block_tridiag_solve_one<BS>(
                D, L, b, x, fact_diag.data(), fact_lower.data(), horizon)) {
            std::cerr << "CPU block tridiagonal solve failed at sample " << sample << "\n";
            std::exit(1);
        }
    }
}

template <int BS>
double time_cpu_solve_us(const std::vector<double>& diag, const std::vector<double>& lower,
    const std::vector<double>& rhs, std::vector<double>& solution, int horizon, int batch,
    int repeats, int threads)
{
    solution.resize(static_cast<std::size_t>(batch * horizon * BS));
    const auto start = std::chrono::steady_clock::now();
    for (int rep = 0; rep < repeats; ++rep) {
        if (threads <= 1 || batch <= 1) {
            cpu_solve_range<BS>(diag, lower, rhs, solution, horizon, 0, batch);
        } else {
            std::vector<std::thread> workers;
            const int chunk = (batch + threads - 1) / threads;
            for (int t = 0; t < threads; ++t) {
                const int begin = t * chunk;
                const int end = std::min(batch, begin + chunk);
                if (begin >= end) {
                    break;
                }
                workers.emplace_back([&diag, &lower, &rhs, &solution, horizon, begin, end]() {
                    cpu_solve_range<BS>(diag, lower, rhs, solution, horizon, begin, end);
                });
            }
            for (auto& worker : workers) {
                worker.join();
            }
        }
    }
    consume_for_benchmark(solution.back());
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count()
        / static_cast<double>(repeats);
}

template <int BS>
__global__ void block_tridiag_solve_kernel(const double* diag, const double* lower,
    const double* rhs, double* solution, double* fact_diag, double* fact_lower, int* info,
    int horizon, int batch)
{
    const int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= batch) {
        return;
    }
    const double* D = diag + static_cast<std::size_t>(sample * horizon * BS * BS);
    const double* L = lower + static_cast<std::size_t>(sample * (horizon - 1) * BS * BS);
    const double* b = rhs + static_cast<std::size_t>(sample * horizon * BS);
    double* x = solution + static_cast<std::size_t>(sample * horizon * BS);
    double* fD = fact_diag + static_cast<std::size_t>(sample * horizon * BS * BS);
    double* fL = fact_lower + static_cast<std::size_t>(sample * (horizon - 1) * BS * BS);
    info[sample] = block_tridiag_solve_one<BS>(D, L, b, x, fD, fL, horizon) ? 0 : 1;
}

template <int BS>
float time_gpu_solve_ms(const std::vector<double>& diag, const std::vector<double>& lower,
    const std::vector<double>& rhs, std::vector<double>& solution, int horizon, int batch,
    int repeats)
{
    double* d_diag = nullptr;
    double* d_lower = nullptr;
    double* d_rhs = nullptr;
    double* d_solution = nullptr;
    double* d_fact_diag = nullptr;
    double* d_fact_lower = nullptr;
    int* d_info = nullptr;
    const std::size_t diag_bytes
        = static_cast<std::size_t>(batch * horizon * BS * BS) * sizeof(double);
    const std::size_t lower_bytes
        = static_cast<std::size_t>(batch * (horizon - 1) * BS * BS) * sizeof(double);
    const std::size_t vec_bytes = static_cast<std::size_t>(batch * horizon * BS) * sizeof(double);

    CUDA_CHECK(cudaMalloc(&d_diag, diag_bytes));
    CUDA_CHECK(cudaMalloc(&d_lower, lower_bytes));
    CUDA_CHECK(cudaMalloc(&d_rhs, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_solution, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_fact_diag, diag_bytes));
    CUDA_CHECK(cudaMalloc(&d_fact_lower, lower_bytes));
    CUDA_CHECK(cudaMalloc(&d_info, static_cast<std::size_t>(batch) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_diag, diag.data(), diag_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lower, lower.data(), lower_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rhs, rhs.data(), vec_bytes, cudaMemcpyHostToDevice));

    const int threads = 128;
    const int blocks = (batch + threads - 1) / threads;
    block_tridiag_solve_kernel<BS><<<blocks, threads>>>(
        d_diag, d_lower, d_rhs, d_solution, d_fact_diag, d_fact_lower, d_info, horizon, batch);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const float total_ms = time_cuda_ms([&]() {
        for (int rep = 0; rep < repeats; ++rep) {
            block_tridiag_solve_kernel<BS><<<blocks, threads>>>(d_diag, d_lower, d_rhs, d_solution,
                d_fact_diag, d_fact_lower, d_info, horizon, batch);
            CUDA_CHECK(cudaGetLastError());
        }
    });

    solution.resize(static_cast<std::size_t>(batch * horizon * BS));
    std::vector<int> info(static_cast<std::size_t>(batch));
    CUDA_CHECK(cudaMemcpy(solution.data(), d_solution, vec_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(info.data(), d_info, static_cast<std::size_t>(batch) * sizeof(int),
        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_diag));
    CUDA_CHECK(cudaFree(d_lower));
    CUDA_CHECK(cudaFree(d_rhs));
    CUDA_CHECK(cudaFree(d_solution));
    CUDA_CHECK(cudaFree(d_fact_diag));
    CUDA_CHECK(cudaFree(d_fact_lower));
    CUDA_CHECK(cudaFree(d_info));

    for (int i = 0; i < batch; ++i) {
        if (info[static_cast<std::size_t>(i)] != 0) {
            std::cerr << "GPU block tridiagonal solve failed at sample " << i << "\n";
            std::exit(1);
        }
    }
    return total_ms / static_cast<float>(repeats);
}

template <int BS>
double max_solution_error(const std::vector<double>& expected, const std::vector<double>& actual)
{
    double err = 0.0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        err = std::max(err, std::abs(expected[i] - actual[i]));
    }
    return err;
}

int benchmark_thread_count(int batch)
{
    const unsigned int hardware_threads = std::thread::hardware_concurrency();
    const int max_threads = hardware_threads == 0u ? 1 : static_cast<int>(hardware_threads);
    return std::max(1, std::min(max_threads, batch));
}

template <int BS> void run_case(const CaseConfig& cfg)
{
    std::vector<double> diag;
    std::vector<double> lower;
    std::vector<double> rhs;
    build_block_tridiag<BS>(diag, lower, rhs, cfg.horizon, cfg.batch);

    std::vector<double> cpu_solution;
    std::vector<double> cpu_threaded_solution;
    std::vector<double> gpu_solution;
    const double cpu_us = time_cpu_solve_us<BS>(
        diag, lower, rhs, cpu_solution, cfg.horizon, cfg.batch, cfg.repeats, 1);
    const int threads = benchmark_thread_count(cfg.batch);
    const double cpu_threaded_us = time_cpu_solve_us<BS>(
        diag, lower, rhs, cpu_threaded_solution, cfg.horizon, cfg.batch, cfg.repeats, threads);
    const float gpu_ms = time_gpu_solve_ms<BS>(
        diag, lower, rhs, gpu_solution, cfg.horizon, cfg.batch, cfg.repeats);
    const double gpu_us = 1000.0 * static_cast<double>(gpu_ms);
    const double best_cpu = std::min(cpu_us, cpu_threaded_us);
    const double err = max_solution_error<BS>(cpu_solution, gpu_solution);
    const double mib
        = static_cast<double>((diag.size() + lower.size()) * sizeof(double)) / (1024.0 * 1024.0);

    std::cout << std::setw(3) << BS << " " << std::setw(5) << cfg.horizon << " " << std::setw(6)
              << cfg.batch << " " << std::setw(4) << cfg.repeats << " " << std::setw(9)
              << std::fixed << std::setprecision(2) << mib << " " << std::setw(12) << cpu_us << " "
              << std::setw(12) << cpu_threaded_us << " " << std::setw(12) << gpu_us << " "
              << std::setw(8) << best_cpu / gpu_us << " " << std::scientific << std::setprecision(2)
              << std::setw(11) << err << "\n";
}

void run_dispatch(const CaseConfig& cfg)
{
    if (cfg.block_dim == 4) {
        run_case<4>(cfg);
    } else if (cfg.block_dim == 8) {
        run_case<8>(cfg);
    } else if (cfg.block_dim == 12) {
        run_case<12>(cfg);
    } else {
        std::cerr << "Unsupported block_dim " << cfg.block_dim << "\n";
        std::exit(1);
    }
}

} // namespace

int main()
{
    cudaDeviceProp prop {};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "MiniSolver CUDA block-tridiagonal factorization microbenchmark\n";
    std::cout << "Device: " << prop.name << "\n";
    std::cout
        << "Metric: explicit block-tridiagonal Cholesky/solve; host/device transfers excluded\n";
    std::cout << "This is a block-sparse route probe, not a Riccati backend.\n";
    std::cout << " BS     N  batch    R       MiB       CPU_us  CPU_thr_us       GPU_us"
                 "    speedup     sol_err\n";

    for (const CaseConfig cfg : {
             CaseConfig { 4, 16, 1, 100 },
             CaseConfig { 4, 64, 256, 20 },
             CaseConfig { 4, 256, 1024, 5 },
             CaseConfig { 8, 64, 256, 10 },
             CaseConfig { 12, 64, 64, 10 },
             CaseConfig { 12, 128, 64, 5 },
         }) {
        run_dispatch(cfg);
    }
    return 0;
}
