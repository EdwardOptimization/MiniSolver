// Standalone benchmark for uploading generated-model Riccati packets to CUDA.
//
// This isolates the host-generated packet transfer cost that a future GPU
// backend would have to hide or fuse away. It is not a solver backend.

#include "../examples/01_car_tutorial/generated/car_model.h"
#include "../examples/02_advanced_bicycle/generated/bicycleextmodel.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
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

template <typename F> double time_host_us(F&& fn)
{
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count();
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

template <typename Model> constexpr int packet_entries()
{
    constexpr int NX = Model::NX;
    constexpr int NU = Model::NU;
    constexpr int NC = Model::NC;
    return NX * NX + NX * NU + NC * NX + NC * NU + NX * NX + NU * NU + NU * NX + NX + NU + NC + NC
        + NC + NC + NX;
}

template <typename Mat>
void append_matrix(
    std::vector<double>& out, std::size_t& offset, const Mat& mat, int rows, int cols)
{
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            out[offset++] = mat(r, c);
        }
    }
}

template <typename Vec>
void append_vector(std::vector<double>& out, std::size_t& offset, const Vec& vec, int n)
{
    for (int i = 0; i < n; ++i) {
        out[offset++] = vec(i);
    }
}

template <typename Model>
void seed_knot(minisolver::KnotPoint<double, Model::NX, Model::NU, Model::NC, Model::NP>& kp,
    int stage, int sample)
{
    constexpr int NX = Model::NX;
    constexpr int NU = Model::NU;
    constexpr int NP = Model::NP;

    kp.set_zero();
    for (int i = 0; i < NX; ++i) {
        kp.x(i) = 0.05 * static_cast<double>(stage) + 0.001 * static_cast<double>(sample + i);
    }
    for (int i = 0; i < NU; ++i) {
        kp.u(i) = 0.01 * static_cast<double>((stage + 2 * i) % 5);
    }
    for (int i = 0; i < NP; ++i) {
        kp.p(i) = 1.0 + 0.01 * static_cast<double>((stage + sample + i) % 11);
    }

    if constexpr (NP >= 15) {
        // Advanced bicycle parameter order from the generated model.
        kp.p(0) = 10.0; // v_ref
        kp.p(1) = 0.5 * static_cast<double>(stage); // x_ref
        kp.p(2) = 0.2 * std::sin(0.05 * static_cast<double>(stage)); // y_ref
        kp.p(3) = 12.0; // obs_x
        kp.p(4) = 0.0; // obs_y
        kp.p(5) = 1.0; // obs_rad
        kp.p(6) = 2.7; // L
        kp.p(7) = 1.0; // car_rad
        for (int i = 8; i < NP; ++i) {
            kp.p(i) = 1.0;
        }
    } else if constexpr (NP >= 13) {
        // Car tutorial parameter order from the generated model.
        kp.p(0) = 5.0;
        kp.p(1) = 0.25 * static_cast<double>(stage);
        kp.p(2) = 0.1 * std::sin(0.05 * static_cast<double>(stage));
        kp.p(3) = 8.0;
        kp.p(4) = 0.0;
        kp.p(5) = 1.0;
        kp.p(6) = 2.7;
        kp.p(7) = 0.8;
        for (int i = 8; i < NP; ++i) {
            kp.p(i) = 1.0;
        }
    }

    for (int i = 0; i < Model::NC; ++i) {
        kp.s(i) = 0.2 + 0.01 * static_cast<double>((stage + i) % 7);
        kp.lam(i) = 0.05 + 0.002 * static_cast<double>((sample + i) % 9);
        kp.soft_s(i) = 0.1 + 0.01 * static_cast<double>(i % 5);
    }
}

template <typename Model>
void pack_knot(const minisolver::KnotPoint<double, Model::NX, Model::NU, Model::NC, Model::NP>& kp,
    std::vector<double>& out, std::size_t packet_index)
{
    constexpr int NX = Model::NX;
    constexpr int NU = Model::NU;
    constexpr int NC = Model::NC;
    std::size_t offset = packet_index * static_cast<std::size_t>(packet_entries<Model>());

    append_matrix(out, offset, kp.A, NX, NX);
    append_matrix(out, offset, kp.B, NX, NU);
    append_matrix(out, offset, kp.C, NC, NX);
    append_matrix(out, offset, kp.D, NC, NU);
    append_matrix(out, offset, kp.Q, NX, NX);
    append_matrix(out, offset, kp.R, NU, NU);
    append_matrix(out, offset, kp.H, NU, NX);
    append_vector(out, offset, kp.q, NX);
    append_vector(out, offset, kp.r, NU);
    append_vector(out, offset, kp.g_val, NC);
    append_vector(out, offset, kp.s, NC);
    append_vector(out, offset, kp.lam, NC);
    append_vector(out, offset, kp.soft_s, NC);
    append_vector(out, offset, kp.f_resid, NX);
}

template <typename Model>
double fill_generated_packets(std::vector<double>& packets, int horizon, int batch)
{
    using Knot = minisolver::KnotPoint<double, Model::NX, Model::NU, Model::NC, Model::NP>;
    const std::size_t total_packets
        = static_cast<std::size_t>(horizon) * static_cast<std::size_t>(batch);
    packets.resize(total_packets * static_cast<std::size_t>(packet_entries<Model>()));

    return time_host_us([&]() {
        Knot kp;
        std::size_t packet = 0;
        for (int sample = 0; sample < batch; ++sample) {
            for (int stage = 0; stage < horizon; ++stage) {
                seed_knot<Model>(kp, stage, sample);
                Model::template compute_exact<double>(
                    kp, minisolver::IntegratorType::RK4_EXPLICIT, 0.05);
                pack_knot<Model>(kp, packets, packet++);
            }
        }
    });
}

double bandwidth_gbps(std::size_t bytes, double us)
{
    if (us <= 0.0) {
        return 0.0;
    }
    return static_cast<double>(bytes) / us / 1000.0;
}

template <typename Model> void run_case(const char* name, int horizon, int batch, int repeats)
{
    std::vector<double> packets;
    const double eval_pack_us = fill_generated_packets<Model>(packets, horizon, batch);
    const std::size_t bytes = packets.size() * sizeof(double);

    double* d_packets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_packets, bytes));

    const float pageable_ms = time_cuda_ms([&]() {
        for (int r = 0; r < repeats; ++r) {
            CUDA_CHECK(cudaMemcpy(d_packets, packets.data(), bytes, cudaMemcpyHostToDevice));
        }
    });

    double* pinned_packets = nullptr;
    CUDA_CHECK(cudaMallocHost(&pinned_packets, bytes));
    std::memcpy(pinned_packets, packets.data(), bytes);
    const float pinned_ms = time_cuda_ms([&]() {
        for (int r = 0; r < repeats; ++r) {
            CUDA_CHECK(cudaMemcpy(d_packets, pinned_packets, bytes, cudaMemcpyHostToDevice));
        }
    });

    CUDA_CHECK(cudaFreeHost(pinned_packets));
    CUDA_CHECK(cudaFree(d_packets));

    const double pageable_us
        = 1000.0 * static_cast<double>(pageable_ms) / static_cast<double>(repeats);
    const double pinned_us = 1000.0 * static_cast<double>(pinned_ms) / static_cast<double>(repeats);
    const double mib = static_cast<double>(bytes) / (1024.0 * 1024.0);

    std::cout << std::setw(18) << name << " " << std::setw(3) << Model::NX << " " << std::setw(3)
              << Model::NU << " " << std::setw(3) << Model::NC << " " << std::setw(4) << horizon
              << " " << std::setw(6) << batch << " " << std::setw(9) << packet_entries<Model>()
              << " " << std::setw(9) << std::fixed << std::setprecision(2) << mib << " "
              << std::setw(12) << eval_pack_us << " " << std::setw(12) << pageable_us << " "
              << std::setw(9) << std::setprecision(2) << bandwidth_gbps(bytes, pageable_us) << " "
              << std::setw(12) << pinned_us << " " << std::setw(9)
              << bandwidth_gbps(bytes, pinned_us) << "\n";
}

template <typename Model> void run_model(const char* name)
{
    run_case<Model>(name, 50, 1, 50);
    run_case<Model>(name, 50, 256, 20);
    run_case<Model>(name, 50, 4096, 5);
    run_case<Model>(name, 100, 1024, 10);
}

} // namespace

int main()
{
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp props {};
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));

    std::cout << "MiniSolver CUDA generated packet upload microbenchmark\n";
    std::cout << "Device: " << props.name << "\n";
    std::cout
        << "Metric: host generated-model eval+pack time and host-to-device packet copy time\n";
    std::cout << "Packet fields: A/B/C/D/Q/R/H/q/r/g/s/lam/soft_s/f_resid\n";
    std::cout << "             model  NX  NU  NC    N  batch   entries       MiB eval_pack_us  "
                 "page_H2D_us "
                 "page_GBps  pin_H2D_us  pin_GBps\n";

    run_model<minisolver::CarModel>("CarModel");
    run_model<minisolver::BicycleExtModel>("BicycleExtModel");

    return 0;
}
