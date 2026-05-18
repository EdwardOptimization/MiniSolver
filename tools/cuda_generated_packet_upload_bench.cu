// Standalone benchmark for uploading generated-model Riccati packets to CUDA.
//
// This isolates the host-generated packet transfer cost that a future GPU
// backend would have to hide or fuse away. It is not a solver backend.

#include "../examples/01_car_tutorial/generated/car_model.h"
#include "../examples/02_advanced_bicycle/generated/bicycleextmodel.h"

#include <cuda_runtime.h>

#include <algorithm>
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

__global__ void fill_packet_kernel(double* packets, std::size_t total_entries)
{
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_entries) {
        return;
    }
    const double stage_term = static_cast<double>(idx % 97u) * 0.001;
    const double packet_term = static_cast<double>((idx / 97u) % 31u) * 0.0001;
    packets[idx] = stage_term + packet_term;
}

__global__ void fill_car_exact_packet_kernel(double* packets, int horizon, int batch)
{
    constexpr int NX = minisolver::CarModel::NX;
    constexpr int NU = minisolver::CarModel::NU;
    constexpr int NC = minisolver::CarModel::NC;
    constexpr int entries = 112;

    const int packet_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_packets = horizon * batch;
    if (packet_idx >= total_packets) {
        return;
    }

    const int sample = packet_idx / horizon;
    const int stage = packet_idx - sample * horizon;
    double* out = packets + static_cast<std::size_t>(packet_idx) * entries;
    for (int i = 0; i < entries; ++i) {
        out[i] = 0.0;
    }

    const double x = 0.05 * static_cast<double>(stage) + 0.001 * static_cast<double>(sample);
    const double y = 0.05 * static_cast<double>(stage) + 0.001 * static_cast<double>(sample + 1);
    const double theta
        = 0.05 * static_cast<double>(stage) + 0.001 * static_cast<double>(sample + 2);
    const double v = 0.05 * static_cast<double>(stage) + 0.001 * static_cast<double>(sample + 3);
    const double acc = 0.01 * static_cast<double>(stage % 5);
    const double steer = 0.01 * static_cast<double>((stage + 2) % 5);
    const double v_ref = 5.0;
    const double x_ref = 0.25 * static_cast<double>(stage);
    const double y_ref = 0.1 * sin(0.05 * static_cast<double>(stage));
    const double obs_x = 8.0;
    const double obs_y = 0.0;
    const double obs_rad = 1.0;
    const double L = 2.7;
    const double car_rad = 0.8;
    const double w_pos = 1.0;
    const double w_vel = 1.0;
    const double w_theta = 1.0;
    const double w_acc = 1.0;
    const double w_steer = 1.0;
    const double lam4 = 0.05 + 0.002 * static_cast<double>((sample + 4) % 9);
    constexpr double dt = 0.05;

    const int off_A = 0;
    const int off_B = off_A + NX * NX;
    const int off_C = off_B + NX * NU;
    const int off_D = off_C + NC * NX;
    const int off_Q = off_D + NC * NU;
    const int off_R = off_Q + NX * NX;
    const int off_H = off_R + NU * NU;
    const int off_q = off_H + NU * NX;
    const int off_r = off_q + NX;
    const int off_g = off_r + NU;
    const int off_s = off_g + NC;
    const int off_lam = off_s + NC;
    const int off_soft_s = off_lam + NC;
    const int off_f = off_soft_s + NC;

    auto mat = [out](int offset, int cols, int row, int col) -> double& {
        return out[offset + row * cols + col];
    };

    const double tmp_d0 = cos(theta);
    const double tmp_d1 = acc * dt;
    const double tmp_d2 = tmp_d1 + v;
    const double tmp_d3 = 1.5 * tmp_d1 + v;
    const double tmp_d4 = 1.0 / L;
    const double tmp_d5 = tan(steer);
    const double tmp_d6 = tmp_d4 * tmp_d5;
    const double tmp_d7 = dt * tmp_d6;
    const double tmp_d8 = theta + tmp_d3 * tmp_d7;
    const double tmp_d9 = cos(tmp_d8);
    const double tmp_d10 = tmp_d2 * tmp_d9;
    const double tmp_d11 = 0.5 * tmp_d1 + v;
    const double tmp_d12 = 0.5 * tmp_d7;
    const double tmp_d13 = theta + tmp_d11 * tmp_d12;
    const double tmp_d14 = cos(tmp_d13);
    const double tmp_d15 = 2.0 * tmp_d14;
    const double tmp_d16 = tmp_d1 + v;
    const double tmp_d17 = theta + tmp_d12 * tmp_d16;
    const double tmp_d18 = cos(tmp_d17);
    const double tmp_d19 = 2.0 * tmp_d18;
    const double tmp_d20 = dt / 6.0;
    const double tmp_d21 = tmp_d20 * (tmp_d0 * v + tmp_d10 + tmp_d11 * tmp_d15 + tmp_d11 * tmp_d19);
    const double tmp_d22 = sin(theta);
    const double tmp_d23 = sin(tmp_d8);
    const double tmp_d24 = tmp_d2 * tmp_d23;
    const double tmp_d25 = sin(tmp_d13);
    const double tmp_d26 = 2.0 * tmp_d25;
    const double tmp_d27 = sin(tmp_d17);
    const double tmp_d28 = 2.0 * tmp_d27;
    const double tmp_d29 = tmp_d11 * tmp_d26 + tmp_d11 * tmp_d28 + tmp_d22 * v + tmp_d24;
    const double tmp_d30 = 4.0 * tmp_d11;
    const double tmp_d31 = dt;
    const double tmp_d32 = tmp_d25 * tmp_d31;
    const double tmp_d33 = tmp_d11 * tmp_d6;
    const double tmp_d34 = tmp_d27 * tmp_d31;
    const double tmp_d35 = tmp_d14 * tmp_d31;
    const double tmp_d36 = tmp_d18 * tmp_d31;
    const double tmp_d37 = dt * dt * tmp_d6;
    const double tmp_d38 = 1.5 * tmp_d37;
    const double tmp_d39 = 0.5 * tmp_d37;
    const double tmp_d40 = tmp_d11 * tmp_d39;
    const double tmp_d41 = tmp_d11 * tmp_d37;
    const double tmp_d42 = tmp_d4 * (tmp_d5 * tmp_d5 + 1.0);
    const double tmp_d43 = tmp_d11 * tmp_d11 * tmp_d42;
    const double tmp_d44 = dt * tmp_d3 * tmp_d42;
    const double tmp_d45 = tmp_d11 * tmp_d16 * tmp_d42;

    out[off_f + 0] = tmp_d21 + x;
    out[off_f + 1] = tmp_d20 * tmp_d29 + y;
    out[off_f + 2] = theta + tmp_d20 * (tmp_d2 * tmp_d6 + tmp_d30 * tmp_d6 + tmp_d6 * v);
    out[off_f + 3] = tmp_d16;

    mat(off_A, NX, 0, 0) = 1.0;
    mat(off_A, NX, 0, 2) = -tmp_d20 * tmp_d29;
    mat(off_A, NX, 0, 3) = tmp_d20
        * (tmp_d0 + tmp_d15 + tmp_d19 - tmp_d24 * tmp_d7 - tmp_d32 * tmp_d33 - tmp_d33 * tmp_d34
            + tmp_d9);
    mat(off_A, NX, 1, 1) = 1.0;
    mat(off_A, NX, 1, 2) = tmp_d21;
    mat(off_A, NX, 1, 3) = tmp_d20
        * (tmp_d10 * tmp_d7 + tmp_d22 + tmp_d23 + tmp_d26 + tmp_d28 + tmp_d33 * tmp_d35
            + tmp_d33 * tmp_d36);
    mat(off_A, NX, 2, 2) = 1.0;
    mat(off_A, NX, 2, 3) = tmp_d31 * tmp_d6;
    mat(off_A, NX, 3, 3) = 1.0;

    mat(off_B, NU, 0, 0) = tmp_d20
        * (dt * tmp_d9 - tmp_d24 * tmp_d38 - tmp_d25 * tmp_d40 - tmp_d27 * tmp_d41 + tmp_d35
            + tmp_d36);
    mat(off_B, NU, 0, 1) = tmp_d20 * (-tmp_d24 * tmp_d44 - tmp_d32 * tmp_d43 - tmp_d34 * tmp_d45);
    mat(off_B, NU, 1, 0) = tmp_d20
        * (dt * tmp_d23 + tmp_d10 * tmp_d38 + tmp_d14 * tmp_d40 + tmp_d18 * tmp_d41 + tmp_d32
            + tmp_d34);
    mat(off_B, NU, 1, 1) = tmp_d20 * (tmp_d10 * tmp_d44 + tmp_d35 * tmp_d43 + tmp_d36 * tmp_d45);
    mat(off_B, NU, 2, 0) = tmp_d39;
    mat(off_B, NU, 2, 1) = tmp_d20 * (tmp_d2 * tmp_d42 + tmp_d30 * tmp_d42 + tmp_d42 * v);
    mat(off_B, NU, 3, 0) = tmp_d31;

    const double dx_obs = x - obs_x;
    const double dy_obs = y - obs_y;
    const double obs_dist = sqrt(dx_obs * dx_obs + dy_obs * dy_obs + 1.0e-6);
    const double inv_obs_dist = 1.0 / obs_dist;

    out[off_g + 0] = acc - 3.0;
    out[off_g + 1] = -acc - 3.0;
    out[off_g + 2] = steer - 0.5;
    out[off_g + 3] = -steer - 0.5;
    out[off_g + 4] = -obs_dist + sqrt((car_rad + obs_rad) * (car_rad + obs_rad));
    mat(off_C, NX, 4, 0) = -dx_obs * inv_obs_dist;
    mat(off_C, NX, 4, 1) = -dy_obs * inv_obs_dist;
    mat(off_D, NU, 0, 0) = 1.0;
    mat(off_D, NU, 1, 0) = -1.0;
    mat(off_D, NU, 2, 1) = 1.0;
    mat(off_D, NU, 3, 1) = -1.0;

    out[off_q + 0] = w_pos * (2.0 * x - 2.0 * x_ref);
    out[off_q + 1] = w_pos * (2.0 * y - 2.0 * y_ref);
    out[off_q + 2] = theta * (2.0 * w_theta);
    out[off_q + 3] = w_vel * (2.0 * v - 2.0 * v_ref);
    out[off_r + 0] = acc * (2.0 * w_acc);
    out[off_r + 1] = steer * (2.0 * w_steer);

    const double obs_y_minus_y = obs_y - y;
    const double obs_x_minus_x = obs_x - x;
    const double tmp_j5 = obs_y_minus_y * obs_y_minus_y + 1.0e-6;
    const double tmp_j7 = obs_x_minus_x * obs_x_minus_x;
    const double tmp_j8 = lam4 / pow(tmp_j5 + tmp_j7, 1.5);
    const double tmp_j9 = obs_y_minus_y * obs_x_minus_x * tmp_j8;
    mat(off_Q, NX, 0, 0) = 2.0 * w_pos - tmp_j5 * tmp_j8;
    mat(off_Q, NX, 0, 1) = tmp_j9;
    mat(off_Q, NX, 1, 0) = tmp_j9;
    mat(off_Q, NX, 1, 1) = 2.0 * w_pos - tmp_j8 * (tmp_j7 + 1.0e-6);
    mat(off_Q, NX, 2, 2) = 2.0 * w_theta;
    mat(off_Q, NX, 3, 3) = 2.0 * w_vel;
    mat(off_R, NU, 0, 0) = 2.0 * w_acc;
    mat(off_R, NU, 1, 1) = 2.0 * w_steer;

    for (int i = 0; i < NC; ++i) {
        out[off_s + i] = 0.2 + 0.01 * static_cast<double>((stage + i) % 7);
        out[off_lam + i] = 0.05 + 0.002 * static_cast<double>((sample + i) % 9);
        out[off_soft_s + i] = 0.1 + 0.01 * static_cast<double>(i % 5);
    }
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
void append_matrix(double* out, std::size_t& offset, const Mat& mat, int rows, int cols)
{
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            out[offset++] = mat(r, c);
        }
    }
}

template <typename Vec> void append_vector(double* out, std::size_t& offset, const Vec& vec, int n)
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
    double* out, std::size_t packet_index)
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

template <typename Model> double fill_generated_packets_raw(double* packets, int horizon, int batch)
{
    using Knot = minisolver::KnotPoint<double, Model::NX, Model::NU, Model::NC, Model::NP>;
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

template <typename Model>
double fill_generated_packets(std::vector<double>& packets, int horizon, int batch)
{
    const std::size_t total_packets
        = static_cast<std::size_t>(horizon) * static_cast<std::size_t>(batch);
    packets.resize(total_packets * static_cast<std::size_t>(packet_entries<Model>()));
    return fill_generated_packets_raw<Model>(packets.data(), horizon, batch);
}

double bandwidth_gbps(std::size_t bytes, double us)
{
    if (us <= 0.0) {
        return 0.0;
    }
    return static_cast<double>(bytes) / us / 1000.0;
}

double benchmark_device_packet_fill(double* d_packets, std::size_t total_entries, int repeats)
{
    constexpr int threads = 256;
    const int blocks = static_cast<int>((total_entries + threads - 1u) / threads);
    fill_packet_kernel<<<blocks, threads>>>(d_packets, total_entries);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const float fill_ms = time_cuda_ms([&]() {
        for (int r = 0; r < repeats; ++r) {
            fill_packet_kernel<<<blocks, threads>>>(d_packets, total_entries);
            CUDA_CHECK(cudaGetLastError());
        }
    });
    return 1000.0 * static_cast<double>(fill_ms) / static_cast<double>(repeats);
}

double max_abs_error(const std::vector<double>& expected, const std::vector<double>& actual)
{
    double err = 0.0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        err = std::max(err, std::abs(expected[i] - actual[i]));
    }
    return err;
}

float benchmark_car_exact_device_packets(
    std::vector<double>& output, int horizon, int batch, int repeats)
{
    static_assert(packet_entries<minisolver::CarModel>() == 112);
    const std::size_t total_entries = static_cast<std::size_t>(horizon) * batch
        * static_cast<std::size_t>(packet_entries<minisolver::CarModel>());
    const std::size_t bytes = total_entries * sizeof(double);

    double* d_packets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_packets, bytes));
    constexpr int threads = 128;
    const int blocks = (horizon * batch + threads - 1) / threads;
    fill_car_exact_packet_kernel<<<blocks, threads>>>(d_packets, horizon, batch);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const float total_ms = time_cuda_ms([&]() {
        for (int r = 0; r < repeats; ++r) {
            fill_car_exact_packet_kernel<<<blocks, threads>>>(d_packets, horizon, batch);
            CUDA_CHECK(cudaGetLastError());
        }
    });

    output.resize(total_entries);
    CUDA_CHECK(cudaMemcpy(output.data(), d_packets, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_packets));
    return total_ms / static_cast<float>(repeats);
}

template <typename Model> void run_case(const char* name, int horizon, int batch, int repeats)
{
    std::vector<double> packets;
    const double eval_pack_us = fill_generated_packets<Model>(packets, horizon, batch);
    const std::size_t bytes = packets.size() * sizeof(double);

    double* d_packets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_packets, bytes));
    const double device_fill_us = benchmark_device_packet_fill(d_packets, packets.size(), repeats);

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

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    const double persistent_eval_upload_us = time_host_us([&]() {
        for (int r = 0; r < repeats; ++r) {
            (void)fill_generated_packets_raw<Model>(pinned_packets, horizon, batch);
            CUDA_CHECK(
                cudaMemcpyAsync(d_packets, pinned_packets, bytes, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }) / static_cast<double>(repeats);
    CUDA_CHECK(cudaStreamDestroy(stream));

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
              << bandwidth_gbps(bytes, pinned_us) << " " << std::setw(15)
              << persistent_eval_upload_us << " " << std::setw(12) << device_fill_us << " "
              << std::setw(12) << bandwidth_gbps(bytes, device_fill_us) << "\n";
}

template <typename Model> void run_model(const char* name)
{
    run_case<Model>(name, 32, 1, 50);
    run_case<Model>(name, 32, 256, 20);
    run_case<Model>(name, 32, 4096, 5);
    run_case<Model>(name, 128, 1, 20);
    run_case<Model>(name, 128, 256, 5);
    run_case<Model>(name, 128, 4096, 2);
}

void run_car_exact_case(int horizon, int batch, int repeats)
{
    std::vector<double> cpu_packets;
    const double cpu_eval_pack_us
        = fill_generated_packets<minisolver::CarModel>(cpu_packets, horizon, batch);
    std::vector<double> gpu_packets;
    const float gpu_exact_ms
        = benchmark_car_exact_device_packets(gpu_packets, horizon, batch, repeats);
    const double gpu_exact_us = 1000.0 * static_cast<double>(gpu_exact_ms);
    const std::size_t bytes = cpu_packets.size() * sizeof(double);
    const double mib = static_cast<double>(bytes) / (1024.0 * 1024.0);
    const double err = max_abs_error(cpu_packets, gpu_packets);

    std::cout << std::setw(4) << horizon << " " << std::setw(6) << batch << " " << std::setw(9)
              << packet_entries<minisolver::CarModel>() << " " << std::setw(9) << std::fixed
              << std::setprecision(2) << mib << " " << std::setw(12) << cpu_eval_pack_us << " "
              << std::setw(12) << gpu_exact_us << " " << std::setw(12)
              << bandwidth_gbps(bytes, gpu_exact_us) << " " << std::scientific
              << std::setprecision(2) << std::setw(12) << err << "\n";
}

void run_car_exact_device_suite()
{
    std::cout << "\nCarModel CUDA exact packet assembly lower-bound\n";
    std::cout << "   N  batch   entries       MiB eval_pack_us gpu_exact_us gpu_exact_GBps"
                 "      max_err\n";
    run_car_exact_case(32, 1, 50);
    run_car_exact_case(32, 256, 20);
    run_car_exact_case(32, 4096, 5);
    run_car_exact_case(128, 1, 20);
    run_car_exact_case(128, 256, 5);
    run_car_exact_case(128, 4096, 2);
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
                 "page_GBps  pin_H2D_us  pin_GBps pin_eval_H2D_us  gpu_fill_us gpu_fill_GBps\n";

    run_model<minisolver::CarModel>("CarModel");
    run_model<minisolver::BicycleExtModel>("BicycleExtModel");
    run_car_exact_device_suite();

    return 0;
}
