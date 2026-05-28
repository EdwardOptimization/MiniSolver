// Benchmark: ORDINARY_SCHUR vs SQRT_CHOLESKY Riccati factorization modes.
// Run: cmake --build build --target riccati_mode_bench && ./build/riccati_mode_bench

#include "../examples/01_car_tutorial/generated/car_model.h"
#include "minisolver/solver/solver.h"
#include <chrono>
#include <cstdio>

using namespace minisolver;

static constexpr int kWarmupRuns = 3;
static constexpr int kBenchRuns = 20;

struct BenchResult {
    double avg_ms = 0.0;
    double min_ms = 1e9;
    double max_ms = 0.0;
    int iters = 0;
    SolverStatus status = SolverStatus::UNSOLVED;
};

void debug_sqrt_qr()
{
    // Small problem to debug SQRT_QR
    MiniSolver<CarModel, 50> solver(3, Backend::CPU_SERIAL);
    SolverConfig config;
    config.riccati_factorization = RiccatiFactorizationMode::SQRT_QR;
    config.print_level = PrintLevel::DEBUG;
    config.mu_init = 0.1;
    config.mu_final = 1e-4;
    config.max_iters = 5;
    solver.set_config(config);
    solver.set_dt(0.1);
    solver.set_initial_state({ 0.0, 0.0, 0.0, 0.0 });
    for (int k = 0; k <= 3; ++k) {
        solver.set_state_guess(k, 0, k * 0.1);
        solver.set_state_guess(k, 1, 0.0);
        solver.set_state_guess(k, 2, 0.0);
        solver.set_state_guess(k, 3, 0.0);
        solver.set_parameter(k, 0, 5.0);
        solver.set_parameter(k, 1, k * 0.1);
        solver.set_parameter(k, 2, 0.0);
        solver.set_parameter(k, 3, 100.0);
        solver.set_parameter(k, 4, 100.0);
        solver.set_parameter(k, 5, 0.1);
        solver.set_parameter(k, 6, 2.5);
        solver.set_parameter(k, 7, 1.0);
        solver.set_parameter(k, 8, 1.0);
        solver.set_parameter(k, 9, 1.0);
        solver.set_parameter(k, 10, 0.1);
        solver.set_parameter(k, 11, 0.1);
        solver.set_parameter(k, 12, 1.0);
    }
    for (int k = 0; k < 3; ++k) {
        solver.set_control_guess(k, 0, 0.0);
        solver.set_control_guess(k, 1, 0.0);
    }
    auto status = solver.solve();
    printf("SQRT_QR debug: status=%d\n", (int)status);
}

BenchResult run_bench(RiccatiFactorizationMode mode, int horizon)
{
    BenchResult result;
    const std::vector<double> x0 = { 0.0, 0.0, 0.0, 0.0 };

    for (int r = 0; r < kWarmupRuns + kBenchRuns; ++r) {
        MiniSolver<CarModel, 50> solver(horizon, Backend::CPU_SERIAL);
        SolverConfig config;
        config.riccati_factorization = mode;
        config.print_level = PrintLevel::NONE;
        config.mu_init = 0.1;
        config.mu_final = 1e-4;
        config.max_iters = 50;
        solver.set_config(config);
        solver.set_dt(0.1);
        solver.set_initial_state(x0);

        for (int k = 0; k <= horizon; ++k) {
            const double x_ref = k * 0.1;
            solver.set_state_guess(k, 0, x_ref);
            solver.set_state_guess(k, 1, 0.0);
            solver.set_state_guess(k, 2, 0.0);
            solver.set_state_guess(k, 3, 0.0);
            // Parameters: v_ref, x_ref, y_ref, obs_x, obs_y, obs_rad, L, car_rad, w_pos, w_vel,
            // w_theta, w_acc, w_steer
            solver.set_parameter(k, 0, 5.0); // v_ref
            solver.set_parameter(k, 1, x_ref); // x_ref
            solver.set_parameter(k, 2, 0.0); // y_ref
            solver.set_parameter(k, 3, 100.0); // obs_x (far away)
            solver.set_parameter(k, 4, 100.0); // obs_y (far away)
            solver.set_parameter(k, 5, 0.1); // obs_rad
            solver.set_parameter(k, 6, 2.5); // L
            solver.set_parameter(k, 7, 1.0); // car_rad
            solver.set_parameter(k, 8, 1.0); // w_pos
            solver.set_parameter(k, 9, 1.0); // w_vel
            solver.set_parameter(k, 10, 0.1); // w_theta
            solver.set_parameter(k, 11, 0.1); // w_acc
            solver.set_parameter(k, 12, 1.0); // w_steer
        }
        for (int k = 0; k < horizon; ++k) {
            solver.set_control_guess(k, 0, 0.0);
            solver.set_control_guess(k, 1, 0.0);
        }

        const auto t0 = std::chrono::steady_clock::now();
        const SolverStatus status = solver.solve();
        const auto t1 = std::chrono::steady_clock::now();

        if (r >= kWarmupRuns) {
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            result.avg_ms += ms;
            if (ms < result.min_ms) {
                result.min_ms = ms;
            }
            if (ms > result.max_ms) {
                result.max_ms = ms;
            }
            result.status = status;
            result.iters = solver.get_iteration_count();
        }
    }
    result.avg_ms /= kBenchRuns;
    return result;
}

int main()
{
    printf("=== Riccati Factorization Mode Benchmark ===\n");
    printf("Warmup: %d runs, Benchmark: %d runs\n\n", kWarmupRuns, kBenchRuns);

    const int horizons[] = { 10, 20, 30, 50 };

    printf("%-6s  %-14s  %8s  %8s  %8s  %5s  %s\n", "N", "Mode", "Avg(ms)", "Min(ms)", "Max(ms)",
        "Iters", "Status");
    printf("------  --------------  --------  --------  --------  -----  ------\n");

    for (const int N : horizons) {
        for (const auto mode :
            { RiccatiFactorizationMode::ORDINARY_SCHUR, RiccatiFactorizationMode::SQRT_CHOLESKY,
                RiccatiFactorizationMode::SQRT_QR, RiccatiFactorizationMode::DUAL_SCHUR_CHOLESKY,
                RiccatiFactorizationMode::DUAL_SCHUR_LDLT,
                RiccatiFactorizationMode::CONTROL_CONDENSED_KKT_LDLT }) {
            const char* mode_str = "UNKNOWN";
            switch (mode) {
            case RiccatiFactorizationMode::ORDINARY_SCHUR:
                mode_str = "ORDINARY";
                break;
            case RiccatiFactorizationMode::SQRT_CHOLESKY:
                mode_str = "SQRT_CHOL";
                break;
            case RiccatiFactorizationMode::SQRT_QR:
                mode_str = "SQRT_QR";
                break;
            case RiccatiFactorizationMode::DUAL_SCHUR_CHOLESKY:
                mode_str = "DUAL_CHOL";
                break;
            case RiccatiFactorizationMode::DUAL_SCHUR_LDLT:
                mode_str = "DUAL_LDLT";
                break;
            case RiccatiFactorizationMode::CONTROL_CONDENSED_KKT_LDLT:
                mode_str = "CTRL_COND";
                break;
            }

            const auto r = run_bench(mode, N);
            printf("%-6d  %-14s  %8.3f  %8.3f  %8.3f  %5d  %s\n", N, mode_str, r.avg_ms, r.min_ms,
                r.max_ms, r.iters, status_to_string(r.status));
        }
        printf("\n");
    }

    return 0;
}
