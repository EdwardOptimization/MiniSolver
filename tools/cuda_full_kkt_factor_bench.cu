// Standalone exploratory benchmark for explicit full-KKT factorization.
//
// This route assembles a synthetic quasi-definite OCP KKT matrix and solves it
// with dense no-pivot LDL on CPU and CUDA. It is intentionally a route probe,
// not a solver backend: the matrix is block-sparse by construction but stored
// densely to expose the cost of treating the whole Newton system as one matrix.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
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
    int nx;
    int nu;
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

int kkt_dim(int nx, int nu, int horizon)
{
    const int stage_dim = nx + nu;
    const int primal_dim = horizon * stage_dim;
    const int constraint_dim = (horizon - 1) * nx;
    return primal_dim + constraint_dim;
}

int primal_index(int stage, int local, int nx, int nu)
{
    return stage * (nx + nu) + local;
}

int lambda_index(int edge, int row, int nx, int nu, int horizon)
{
    return horizon * (nx + nu) + edge * nx + row;
}

void add_symmetric(double* K, int dim, int row, int col, double value)
{
    K[row * dim + col] += value;
    if (row != col) {
        K[col * dim + row] += value;
    }
}

void build_kkt_system(
    std::vector<double>& matrices, std::vector<double>& rhs, const CaseConfig& cfg)
{
    const int dim = kkt_dim(cfg.nx, cfg.nu, cfg.horizon);
    const int matrix_entries = dim * dim;
    matrices.assign(static_cast<std::size_t>(cfg.batch * matrix_entries), 0.0);
    rhs.assign(static_cast<std::size_t>(cfg.batch * dim), 0.0);

    for (int sample = 0; sample < cfg.batch; ++sample) {
        double* K = matrices.data() + static_cast<std::size_t>(sample * matrix_entries);
        double* b = rhs.data() + static_cast<std::size_t>(sample * dim);

        for (int k = 0; k < cfg.horizon; ++k) {
            for (int i = 0; i < cfg.nx + cfg.nu; ++i) {
                const int row = primal_index(k, i, cfg.nx, cfg.nu);
                const double diag = 2.0 + 0.02 * static_cast<double>((i + k + sample) % 7);
                K[row * dim + row] = diag;
                b[row] = 0.1 + 0.001 * static_cast<double>((row + sample) % 17);
            }
            for (int i = 0; i + 1 < cfg.nx + cfg.nu; ++i) {
                const int row = primal_index(k, i, cfg.nx, cfg.nu);
                const int col = primal_index(k, i + 1, cfg.nx, cfg.nu);
                add_symmetric(K, dim, row, col, 0.01);
            }
        }

        constexpr double dual_regularization = 0.25;
        for (int edge = 0; edge + 1 < cfg.horizon; ++edge) {
            for (int i = 0; i < cfg.nx; ++i) {
                const int lam = lambda_index(edge, i, cfg.nx, cfg.nu, cfg.horizon);
                K[lam * dim + lam] = -dual_regularization;
                b[lam] = 0.02 * static_cast<double>((edge + i + sample) % 5);

                const int x_next = primal_index(edge + 1, i, cfg.nx, cfg.nu);
                const int x_cur = primal_index(edge, i, cfg.nx, cfg.nu);
                const int u_cur = primal_index(edge, cfg.nx + (i % cfg.nu), cfg.nx, cfg.nu);
                add_symmetric(K, dim, lam, x_next, 1.0);
                add_symmetric(K, dim, lam, x_cur, -0.92);
                add_symmetric(K, dim, lam, u_cur, -0.03);
            }
        }
    }
}

__host__ __device__ bool ldl_solve_one(
    const double* A, const double* b, double* x, double* L, double* D, int dim)
{
    for (int i = 0; i < dim * dim; ++i) {
        L[i] = 0.0;
    }

    for (int k = 0; k < dim; ++k) {
        double diag = A[k * dim + k];
        for (int p = 0; p < k; ++p) {
            const double lkp = L[k * dim + p];
            diag -= lkp * D[p] * lkp;
        }
        if (!(fabs(diag) > 1.0e-12)) {
            return false;
        }
        D[k] = diag;
        L[k * dim + k] = 1.0;

        for (int i = k + 1; i < dim; ++i) {
            double acc = A[i * dim + k];
            for (int p = 0; p < k; ++p) {
                acc -= L[i * dim + p] * D[p] * L[k * dim + p];
            }
            L[i * dim + k] = acc / D[k];
        }
    }

    for (int i = 0; i < dim; ++i) {
        double acc = b[i];
        for (int j = 0; j < i; ++j) {
            acc -= L[i * dim + j] * x[j];
        }
        x[i] = acc;
    }
    for (int i = 0; i < dim; ++i) {
        x[i] /= D[i];
    }
    for (int i = dim - 1; i >= 0; --i) {
        double acc = x[i];
        for (int j = i + 1; j < dim; ++j) {
            acc -= L[j * dim + i] * x[j];
        }
        x[i] = acc;
    }

    return true;
}

void cpu_solve_range(const std::vector<double>& matrices, const std::vector<double>& rhs,
    std::vector<double>& solution, int dim, int begin, int end)
{
    std::vector<double> L(static_cast<std::size_t>(dim * dim));
    std::vector<double> D(static_cast<std::size_t>(dim));
    for (int sample = begin; sample < end; ++sample) {
        const double* A = matrices.data() + static_cast<std::size_t>(sample * dim * dim);
        const double* b = rhs.data() + static_cast<std::size_t>(sample * dim);
        double* x = solution.data() + static_cast<std::size_t>(sample * dim);
        if (!ldl_solve_one(A, b, x, L.data(), D.data(), dim)) {
            std::cerr << "CPU LDL failed at sample " << sample << "\n";
            std::exit(1);
        }
    }
}

double time_cpu_solve_us(const std::vector<double>& matrices, const std::vector<double>& rhs,
    std::vector<double>& solution, int dim, int batch, int repeats, int threads)
{
    solution.resize(static_cast<std::size_t>(batch * dim));
    const auto start = std::chrono::steady_clock::now();
    for (int rep = 0; rep < repeats; ++rep) {
        if (threads <= 1 || batch <= 1) {
            cpu_solve_range(matrices, rhs, solution, dim, 0, batch);
        } else {
            std::vector<std::thread> workers;
            const int chunk = (batch + threads - 1) / threads;
            for (int t = 0; t < threads; ++t) {
                const int begin = t * chunk;
                const int end = std::min(batch, begin + chunk);
                if (begin >= end) {
                    break;
                }
                workers.emplace_back([&matrices, &rhs, &solution, dim, begin, end]() {
                    cpu_solve_range(matrices, rhs, solution, dim, begin, end);
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

__global__ void ldl_solve_batch_kernel(const double* matrices, const double* rhs, double* solution,
    double* factors, double* diag, int* info, int dim, int batch)
{
    const int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= batch) {
        return;
    }

    const double* A = matrices + static_cast<std::size_t>(sample * dim * dim);
    const double* b = rhs + static_cast<std::size_t>(sample * dim);
    double* x = solution + static_cast<std::size_t>(sample * dim);
    double* L = factors + static_cast<std::size_t>(sample * dim * dim);
    double* D = diag + static_cast<std::size_t>(sample * dim);
    info[sample] = ldl_solve_one(A, b, x, L, D, dim) ? 0 : 1;
}

float time_gpu_solve_ms(const std::vector<double>& matrices, const std::vector<double>& rhs,
    std::vector<double>& solution, int dim, int batch, int repeats)
{
    double* d_matrices = nullptr;
    double* d_rhs = nullptr;
    double* d_solution = nullptr;
    double* d_factors = nullptr;
    double* d_diag = nullptr;
    int* d_info = nullptr;

    const std::size_t matrix_bytes = static_cast<std::size_t>(batch * dim * dim) * sizeof(double);
    const std::size_t vector_bytes = static_cast<std::size_t>(batch * dim) * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_matrices, matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_rhs, vector_bytes));
    CUDA_CHECK(cudaMalloc(&d_solution, vector_bytes));
    CUDA_CHECK(cudaMalloc(&d_factors, matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_diag, vector_bytes));
    CUDA_CHECK(cudaMalloc(&d_info, static_cast<std::size_t>(batch) * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_matrices, matrices.data(), matrix_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rhs, rhs.data(), vector_bytes, cudaMemcpyHostToDevice));

    const int threads = 128;
    const int blocks = (batch + threads - 1) / threads;
    ldl_solve_batch_kernel<<<blocks, threads>>>(
        d_matrices, d_rhs, d_solution, d_factors, d_diag, d_info, dim, batch);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const float total_ms = time_cuda_ms([&]() {
        for (int rep = 0; rep < repeats; ++rep) {
            ldl_solve_batch_kernel<<<blocks, threads>>>(
                d_matrices, d_rhs, d_solution, d_factors, d_diag, d_info, dim, batch);
            CUDA_CHECK(cudaGetLastError());
        }
    });

    solution.resize(static_cast<std::size_t>(batch * dim));
    std::vector<int> info(static_cast<std::size_t>(batch));
    CUDA_CHECK(cudaMemcpy(solution.data(), d_solution, vector_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(info.data(), d_info, static_cast<std::size_t>(batch) * sizeof(int),
        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_matrices));
    CUDA_CHECK(cudaFree(d_rhs));
    CUDA_CHECK(cudaFree(d_solution));
    CUDA_CHECK(cudaFree(d_factors));
    CUDA_CHECK(cudaFree(d_diag));
    CUDA_CHECK(cudaFree(d_info));

    for (int i = 0; i < batch; ++i) {
        if (info[static_cast<std::size_t>(i)] != 0) {
            std::cerr << "GPU LDL failed at sample " << i << "\n";
            std::exit(1);
        }
    }
    return total_ms / static_cast<float>(repeats);
}

double max_solution_error(const std::vector<double>& expected, const std::vector<double>& actual)
{
    double err = 0.0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        err = std::max(err, std::abs(expected[i] - actual[i]));
    }
    return err;
}

double max_residual(const std::vector<double>& matrices, const std::vector<double>& rhs,
    const std::vector<double>& solution, int dim, int batch)
{
    double max_err = 0.0;
    for (int sample = 0; sample < batch; ++sample) {
        const double* A = matrices.data() + static_cast<std::size_t>(sample * dim * dim);
        const double* b = rhs.data() + static_cast<std::size_t>(sample * dim);
        const double* x = solution.data() + static_cast<std::size_t>(sample * dim);
        for (int r = 0; r < dim; ++r) {
            double acc = 0.0;
            for (int c = 0; c < dim; ++c) {
                acc += A[r * dim + c] * x[c];
            }
            max_err = std::max(max_err, std::abs(acc - b[r]));
        }
    }
    return max_err;
}

int benchmark_thread_count(int batch)
{
    const unsigned int hardware_threads = std::thread::hardware_concurrency();
    const int max_threads = hardware_threads == 0u ? 1 : static_cast<int>(hardware_threads);
    return std::max(1, std::min(max_threads, batch));
}

void run_case(const CaseConfig& cfg)
{
    const int dim = kkt_dim(cfg.nx, cfg.nu, cfg.horizon);
    std::vector<double> matrices;
    std::vector<double> rhs;
    build_kkt_system(matrices, rhs, cfg);

    std::vector<double> cpu_solution;
    std::vector<double> cpu_threaded_solution;
    std::vector<double> gpu_solution;

    const double cpu_us
        = time_cpu_solve_us(matrices, rhs, cpu_solution, dim, cfg.batch, cfg.repeats, 1);
    const int threads = benchmark_thread_count(cfg.batch);
    const double cpu_threaded_us = time_cpu_solve_us(
        matrices, rhs, cpu_threaded_solution, dim, cfg.batch, cfg.repeats, threads);
    const double best_cpu_us = std::min(cpu_us, cpu_threaded_us);

    const float gpu_ms
        = time_gpu_solve_ms(matrices, rhs, gpu_solution, dim, cfg.batch, cfg.repeats);
    const double gpu_us = 1000.0 * static_cast<double>(gpu_ms);
    const double sol_err = max_solution_error(cpu_solution, gpu_solution);
    const double residual = max_residual(matrices, rhs, gpu_solution, dim, cfg.batch);
    const double mib = static_cast<double>(matrices.size() * sizeof(double)) / (1024.0 * 1024.0);

    std::cout << std::setw(3) << cfg.nx << " " << std::setw(3) << cfg.nu << " " << std::setw(4)
              << cfg.horizon << " " << std::setw(5) << dim << " " << std::setw(6) << cfg.batch
              << " " << std::setw(4) << cfg.repeats << " " << std::setw(9) << std::fixed
              << std::setprecision(2) << mib << " " << std::setw(12) << cpu_us << " "
              << std::setw(12) << cpu_threaded_us << " " << std::setw(12) << gpu_us << " "
              << std::setw(8) << best_cpu_us / gpu_us << " " << std::scientific
              << std::setprecision(2) << std::setw(11) << sol_err << " " << std::setw(11)
              << residual << "\n";
}

} // namespace

int main()
{
    cudaDeviceProp prop {};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "MiniSolver CUDA explicit full-KKT factorization microbenchmark\n";
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Metric: dense no-pivot LDL solve of explicit quasi-definite OCP KKT; "
                 "host/device transfers excluded\n";
    std::cout << "This is a full-matrix route probe, not a Riccati backend.\n";
    std::cout << " NX  NU    N   dim  batch    R       MiB       CPU_us  CPU_thr_us       GPU_us"
                 "    speedup     sol_err    residual\n";

    for (const CaseConfig cfg : {
             CaseConfig { 2, 1, 8, 1, 100 },
             CaseConfig { 2, 1, 16, 64, 20 },
             CaseConfig { 4, 2, 8, 64, 20 },
             CaseConfig { 4, 2, 16, 16, 10 },
             CaseConfig { 8, 4, 8, 16, 10 },
             CaseConfig { 4, 2, 24, 8, 5 },
         }) {
        run_case(cfg);
    }

    return 0;
}
