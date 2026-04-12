#include "../examples/01_car_tutorial/generated/car_model.h"
#include "minisolver/core/serializer.h"
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio> // for remove()
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <string>
#include <vector>

using namespace minisolver;

namespace {
std::string MakeUniqueTestFilename(const char* stem, const char* ext)
{
    static std::atomic<uint64_t> seq { 0 };
    const uint64_t n = ++seq;
    const uint64_t t = static_cast<uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());

    std::string name = stem;
    name += "_";
    name += std::to_string(t);
    name += "_";
    name += std::to_string(n);
    name += ext;
    return name;
}

SolverConfig MakeNonDefaultConfig()
{
    SolverConfig config;
    config.backend = Backend::GPU_PCR;
    config.initialization = InitializationMode::REUSE_PRIMAL_DUAL;
    config.integrator = IntegratorType::EULER_IMPLICIT;
    config.default_dt = 0.123;
    config.barrier_strategy = BarrierStrategy::MEHROTRA;

    config.mu_init = 0.7;
    config.mu_final = 1e-7;
    config.mu_linear_decrease_factor = 0.33;
    config.barrier_tolerance_factor = 9.0;
    config.mu_safety_margin = 0.25;

    config.inertia_strategy = InertiaStrategy::IGNORE_SINGULAR;
    config.reg_init = 1e-3;
    config.reg_min = 1e-10;
    config.reg_max = 1e8;
    config.reg_scale_up = 11.0;
    config.reg_scale_down = 1.7;
    config.regularization_step = 1e-5;
    config.singular_threshold = 1e-6;
    config.huge_penalty = 9e8;
    config.inertia_max_retries = 3;

    config.tol_grad = 1e-3;
    config.tol_con = 2e-3;
    config.tol_dual = 3e-3;
    config.tol_mu = 4e-6;
    config.tol_cost = 5e-7;
    config.feasible_tol_scale = 12.0;

    config.line_search_type = LineSearchType::MERIT;
    config.line_search_max_iters = 7;
    config.line_search_tau = 0.9;
    config.line_search_backtrack_factor = 0.3;
    config.filter_gamma_theta = 2e-5;
    config.filter_gamma_phi = 3e-5;

    config.min_barrier_slack = 1e-13;
    config.barrier_inf_cost = 1e7;
    config.slack_reset_trigger = 2e-3;
    config.warm_start_slack_init = 2e-6;
    config.soc_trigger_alpha = 0.45;
    config.merit_nu_init = 321.0;
    config.eta_suff_descent = 2e-4;

    config.max_restoration_iters = 2;
    config.restoration_mu = 2e-2;
    config.restoration_reg = 3e-2;
    config.restoration_alpha = 0.85;

    config.max_iters = 17;
    config.print_level = PrintLevel::DEBUG;
    config.enable_profiling = false;

    config.hessian_approximation = HessianApproximation::EXACT;
    config.enable_iterative_refinement = true;
    config.max_refinement_steps = 2;
    config.enable_rti = true;
    config.enable_line_search_rollout = true;

    config.enable_defect_correction = false;
    config.enable_corrector = false;
    config.enable_aggressive_barrier = false;
    config.enable_slack_reset = false;
    config.enable_feasibility_restoration = false;
    config.enable_soc = false;

    return config;
}

void ExpectConfigEq(const SolverConfig& a, const SolverConfig& b)
{
    EXPECT_EQ(a.backend, b.backend);
    EXPECT_EQ(a.initialization, b.initialization);
    EXPECT_EQ(a.integrator, b.integrator);
    EXPECT_DOUBLE_EQ(a.default_dt, b.default_dt);
    EXPECT_EQ(a.barrier_strategy, b.barrier_strategy);

    EXPECT_DOUBLE_EQ(a.mu_init, b.mu_init);
    EXPECT_DOUBLE_EQ(a.mu_final, b.mu_final);
    EXPECT_DOUBLE_EQ(a.mu_linear_decrease_factor, b.mu_linear_decrease_factor);
    EXPECT_DOUBLE_EQ(a.barrier_tolerance_factor, b.barrier_tolerance_factor);
    EXPECT_DOUBLE_EQ(a.mu_safety_margin, b.mu_safety_margin);

    EXPECT_EQ(a.inertia_strategy, b.inertia_strategy);
    EXPECT_DOUBLE_EQ(a.reg_init, b.reg_init);
    EXPECT_DOUBLE_EQ(a.reg_min, b.reg_min);
    EXPECT_DOUBLE_EQ(a.reg_max, b.reg_max);
    EXPECT_DOUBLE_EQ(a.reg_scale_up, b.reg_scale_up);
    EXPECT_DOUBLE_EQ(a.reg_scale_down, b.reg_scale_down);
    EXPECT_DOUBLE_EQ(a.regularization_step, b.regularization_step);
    EXPECT_DOUBLE_EQ(a.singular_threshold, b.singular_threshold);
    EXPECT_DOUBLE_EQ(a.huge_penalty, b.huge_penalty);
    EXPECT_EQ(a.inertia_max_retries, b.inertia_max_retries);

    EXPECT_DOUBLE_EQ(a.tol_grad, b.tol_grad);
    EXPECT_DOUBLE_EQ(a.tol_con, b.tol_con);
    EXPECT_DOUBLE_EQ(a.tol_dual, b.tol_dual);
    EXPECT_DOUBLE_EQ(a.tol_mu, b.tol_mu);
    EXPECT_DOUBLE_EQ(a.tol_cost, b.tol_cost);
    EXPECT_DOUBLE_EQ(a.feasible_tol_scale, b.feasible_tol_scale);

    EXPECT_EQ(a.line_search_type, b.line_search_type);
    EXPECT_EQ(a.line_search_max_iters, b.line_search_max_iters);
    EXPECT_DOUBLE_EQ(a.line_search_tau, b.line_search_tau);
    EXPECT_DOUBLE_EQ(a.line_search_backtrack_factor, b.line_search_backtrack_factor);
    EXPECT_DOUBLE_EQ(a.filter_gamma_theta, b.filter_gamma_theta);
    EXPECT_DOUBLE_EQ(a.filter_gamma_phi, b.filter_gamma_phi);

    EXPECT_DOUBLE_EQ(a.min_barrier_slack, b.min_barrier_slack);
    EXPECT_DOUBLE_EQ(a.barrier_inf_cost, b.barrier_inf_cost);
    EXPECT_DOUBLE_EQ(a.slack_reset_trigger, b.slack_reset_trigger);
    EXPECT_DOUBLE_EQ(a.warm_start_slack_init, b.warm_start_slack_init);
    EXPECT_DOUBLE_EQ(a.soc_trigger_alpha, b.soc_trigger_alpha);
    EXPECT_DOUBLE_EQ(a.merit_nu_init, b.merit_nu_init);
    EXPECT_DOUBLE_EQ(a.eta_suff_descent, b.eta_suff_descent);

    EXPECT_EQ(a.max_restoration_iters, b.max_restoration_iters);
    EXPECT_DOUBLE_EQ(a.restoration_mu, b.restoration_mu);
    EXPECT_DOUBLE_EQ(a.restoration_reg, b.restoration_reg);
    EXPECT_DOUBLE_EQ(a.restoration_alpha, b.restoration_alpha);

    EXPECT_EQ(a.max_iters, b.max_iters);
    EXPECT_EQ(a.print_level, b.print_level);
    EXPECT_EQ(a.enable_profiling, b.enable_profiling);

    EXPECT_EQ(a.hessian_approximation, b.hessian_approximation);
    EXPECT_EQ(a.enable_iterative_refinement, b.enable_iterative_refinement);
    EXPECT_EQ(a.max_refinement_steps, b.max_refinement_steps);
    EXPECT_EQ(a.enable_rti, b.enable_rti);
    EXPECT_EQ(a.enable_line_search_rollout, b.enable_line_search_rollout);

    EXPECT_EQ(a.enable_defect_correction, b.enable_defect_correction);
    EXPECT_EQ(a.enable_corrector, b.enable_corrector);
    EXPECT_EQ(a.enable_aggressive_barrier, b.enable_aggressive_barrier);
    EXPECT_EQ(a.enable_slack_reset, b.enable_slack_reset);
    EXPECT_EQ(a.enable_feasibility_restoration, b.enable_feasibility_restoration);
    EXPECT_EQ(a.enable_soc, b.enable_soc);
}

// A minimal L1 soft-constraint model to verify that snapshots serialize soft_s correctly.
struct L1SoftModel {
    static constexpr int NX = 1;
    static constexpr int NU = 1;
    static constexpr int NC = 1;
    static constexpr int NP = 0;

    inline static std::array<double, NC> constraint_weights = { 0.0 };
    inline static std::array<int, NC> constraint_types = { 0 };

    static constexpr std::array<const char*, NX> state_names = { "x" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = {};

    template <typename T>
    static MSVec<T, NX> integrate(const MSVec<T, NX>& x, const MSVec<T, NU>& u,
        const MSVec<T, NP>& /*p*/, double dt, IntegratorType /*type*/)
    {
        MSVec<T, NX> x_next;
        x_next(0) = x(0) + u(0) * dt;
        return x_next;
    }

    template <typename T>
    static void compute_dynamics(
        KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType /*type*/, double dt)
    {
        const T x = kp.x(0);
        const T u = kp.u(0);
        kp.f_resid(0) = x + u * dt;
        kp.A(0, 0) = 1.0;
        kp.B(0, 0) = dt;
    }

    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        // x <= 5  ->  g = x - 5 <= 0
        kp.g_val(0) = kp.x(0) - 5.0;
        kp.C(0, 0) = 1.0;
        kp.D(0, 0) = 0.0;
    }

    template <typename T> static void compute_cost_impl(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        const T diff = kp.x(0) - 10.0;
        kp.cost = diff * diff + 1e-4 * kp.u(0) * kp.u(0);

        kp.q(0) = 2 * diff;
        kp.r(0) = 2e-4 * kp.u(0);
        kp.Q(0, 0) = 2.0;
        kp.R(0, 0) = 2e-4;
        kp.H.setZero();
    }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_impl(kp);
    }
    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_impl(kp);
    }
    template <typename T> static void compute_cost(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_impl(kp);
    }
    template <typename T>
    static void compute(KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType type, double dt)
    {
        compute_dynamics(kp, type, dt);
        compute_constraints(kp);
        compute_cost(kp);
    }
};
} // namespace

TEST(SerializerTest, CaptureAndSaveAndLoad)
{
    int N = 20;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    // Set some non-default config values to verify they are saved correctly
    config.mu_init = 0.5;
    config.tol_con = 1e-4;
    config.barrier_strategy = BarrierStrategy::MEHROTRA;

    using MySolver = MiniSolver<CarModel, 50>;
    using Serializer = SolverSerializer<CarModel, 50>;

    MySolver solver(N, Backend::CPU_SERIAL, config);
    solver.set_dt(0.1);

    // Set parameters and initial state
    for (int k = 0; k <= N; ++k) {
        solver.set_parameter(k, "v_ref", 5.0);
        solver.set_parameter(k, "w_vel", 1.0);
        solver.set_parameter(k, "obs_x", 100.0);
        solver.set_parameter(k, "obs_rad", 1.0);
        solver.set_parameter(k, "L", 2.5);
        solver.set_parameter(k, "car_rad", 1.0);

        // Set some dummy state values manually to verify trajectory save/load
        // (Instead of solving, we just fill data to check exact IO reproduction)
        for (int i = 0; i < CarModel::NX; ++i)
            solver.set_state_guess(k, i, k * 1.0);
        if (k < N) {
            for (int i = 0; i < CarModel::NU; ++i)
                solver.set_control_guess(k, i, k * 0.5);
        }
        for (int i = 0; i < CarModel::NC; ++i) {
            solver.set_slack_guess(k, i, 0.1);
            solver.set_dual_guess(k, i, 0.2);
        }
    }

    // 1. Capture State (Test New Interface)
    auto snapshot = Serializer::capture_state(solver, SolverStatus::FEASIBLE);
    snapshot.iterations = 7;
    snapshot.mu = 0.3;
    snapshot.reg = 1.2;

    EXPECT_EQ(snapshot.N, N);
    EXPECT_EQ(snapshot.config.mu_init, 0.5);
    EXPECT_EQ(snapshot.status, SolverStatus::FEASIBLE);
    EXPECT_EQ(snapshot.iterations, 7);
    EXPECT_DOUBLE_EQ(snapshot.mu, 0.3);
    EXPECT_DOUBLE_EQ(snapshot.reg, 1.2);
    EXPECT_DOUBLE_EQ(snapshot.total_cost, 0.0);

    // 2. Save to Disk
    std::string filename = MakeUniqueTestFilename("test_snapshot", ".dat");
    bool save_ok = Serializer::save_state(filename, snapshot);
    EXPECT_TRUE(save_ok);

    // 3. Load into new solver
    MySolver solver2(10, Backend::CPU_SERIAL); // Initialize with different N
    bool load_ok = Serializer::load_case(filename, solver2);
    EXPECT_TRUE(load_ok);

    // 4. Verify Loaded Data
    EXPECT_EQ(solver2.get_horizon(), N);
    EXPECT_EQ(solver2.get_config().mu_init, 0.5);
    EXPECT_EQ(solver2.get_config().barrier_strategy, BarrierStrategy::MEHROTRA);
    auto loaded_snapshot = Serializer::capture_state(solver2);
    EXPECT_EQ(loaded_snapshot.iterations, 7);
    EXPECT_DOUBLE_EQ(loaded_snapshot.mu, 0.3);
    EXPECT_DOUBLE_EQ(loaded_snapshot.reg, 1.2);

    // Verify Trajectory
    for (int k = 0; k <= N; ++k) {
        // X
        for (int i = 0; i < CarModel::NX; ++i) {
            EXPECT_DOUBLE_EQ(solver2.get_state(k, i), k * 1.0);
        }
        // U
        if (k < N) {
            for (int i = 0; i < CarModel::NU; ++i) {
                EXPECT_DOUBLE_EQ(solver2.get_control(k, i), k * 0.5);
            }
        }
        // P
        EXPECT_EQ(solver2.get_parameter(k, solver2.get_param_idx("v_ref")), 5.0);
        EXPECT_EQ(solver2.get_parameter(k, solver2.get_param_idx("L")), 2.5);

        // Slacks/Duals
        for (int i = 0; i < CarModel::NC; ++i) {
            EXPECT_DOUBLE_EQ(solver2.get_slack(k, i), 0.1);
            EXPECT_DOUBLE_EQ(solver2.get_dual(k, i), 0.2);
        }
    }

    // Cleanup
    std::remove(filename.c_str());
}

TEST(SerializerTest, SnapshotSerializesSoftS_L1)
{
    L1SoftModel::constraint_types[0] = 1; // L1
    L1SoftModel::constraint_weights[0] = 1.0; // w=1

    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.max_iters = 0; // Presolve initializes soft_s; no need to iterate.

    using Solver = MiniSolver<L1SoftModel, 5>;
    using Serializer = SolverSerializer<L1SoftModel, 5>;

    Solver solver(1, Backend::CPU_SERIAL, config);
    solver.set_dt(1.0);
    solver.set_initial_state("x", 0.0);

    solver.solve();

    auto snapA = Serializer::capture_state(solver);
    ASSERT_EQ(snapA.N, 1);
    ASSERT_EQ(snapA.trajectory.size(), 2u);
    const double soft_s_A = snapA.trajectory[0].soft_s[0];
    ASSERT_TRUE(std::isfinite(soft_s_A));
    ASSERT_GT(soft_s_A, 0.0);

    const std::string filename = MakeUniqueTestFilename("test_softs_snapshot", ".bin");
    ASSERT_TRUE(Serializer::save_state(filename, snapA));

    Solver solver2(1, Backend::CPU_SERIAL, config);
    ASSERT_TRUE(Serializer::load_case(filename, solver2));

    auto snapB = Serializer::capture_state(solver2);
    ASSERT_EQ(snapB.N, 1);
    const double soft_s_B = snapB.trajectory[0].soft_s[0];
    EXPECT_DOUBLE_EQ(soft_s_B, soft_s_A);

    std::remove(filename.c_str());
}

TEST(SerializerTest, SnapshotSerializesAllConfigFields)
{
    SolverConfig config = MakeNonDefaultConfig();

    // Create Solver A (config.backend is overwritten by ctor backend argument).
    MiniSolver<CarModel, 10> solverA(2, config.backend, config);
    ASSERT_EQ(solverA.get_config().backend, Backend::GPU_PCR);

    std::string filename = MakeUniqueTestFilename("test_config_snapshot", ".bin");
    ASSERT_TRUE((minisolver::SolverSerializer<CarModel, 10>::save_case(filename, solverA)));

    MiniSolver<CarModel, 10> solverB(2, Backend::CPU_SERIAL);
    ASSERT_TRUE((minisolver::SolverSerializer<CarModel, 10>::load_case(filename, solverB)));

    const SolverConfig& cfgA = solverA.get_config();
    const SolverConfig& cfgB = solverB.get_config();
    ExpectConfigEq(cfgA, cfgB);

    std::remove(filename.c_str());
}

TEST(SerializerTest, RejectsOldFormatMagic)
{
    const std::string filename = MakeUniqueTestFilename("test_old_magic", ".bin");
    {
        std::ofstream out(filename, std::ios::binary);
        ASSERT_TRUE(out.good());
        out.write("MINISOLV_2", 10); // old magic
    }

    MiniSolver<CarModel, 10> solver(2, Backend::CPU_SERIAL);
    EXPECT_FALSE((minisolver::SolverSerializer<CarModel, 10>::load_case(filename, solver)));
    std::remove(filename.c_str());
}

TEST(SerializerTest, FullRoundTrip)
{
    int N = 5;
    SolverConfig config;

    // 1. Create and Run Solver A
    MiniSolver<CarModel, 10> solverA(N, Backend::CPU_SERIAL, config);
    solverA.set_dt(0.1);
    solverA.set_initial_state("x", 1.0);
    solverA.set_parameter(0, "v_ref", 5.0);

    // Run 1 step to populate internal state (s, lam, etc.)
    SolverConfig cfgA = solverA.get_config();
    cfgA.max_iters = 1;
    solverA.set_config(cfgA);
    solverA.solve();

    // 2. Serialize to File
    std::string filename = MakeUniqueTestFilename("test_roundtrip", ".bin");
    minisolver::SolverSerializer<CarModel, 10>::save_case(filename, solverA);

    // 3. Deserialize to Solver B
    MiniSolver<CarModel, 10> solverB(N, Backend::CPU_SERIAL, config);
    bool load_ok = minisolver::SolverSerializer<CarModel, 10>::load_case(filename, solverB);
    EXPECT_TRUE(load_ok);

    // 4. Cleanup
    std::remove(filename.c_str());

    // 5. Compare State (Bit-Exact)
    EXPECT_EQ(solverA.get_horizon(), solverB.get_horizon());
    EXPECT_EQ(solverA.get_iteration_count(), solverB.get_iteration_count());

    auto stateA = minisolver::SolverSerializer<CarModel, 10>::capture_state(solverA);
    auto stateB = minisolver::SolverSerializer<CarModel, 10>::capture_state(solverB);
    EXPECT_DOUBLE_EQ(stateA.mu, stateB.mu);
    EXPECT_DOUBLE_EQ(stateA.reg, stateB.reg);
    EXPECT_DOUBLE_EQ(stateA.total_cost, stateB.total_cost);

    for (int k = 0; k <= N; ++k) {
        for (int i = 0; i < CarModel::NX; ++i)
            EXPECT_EQ(solverA.get_state(k, i), solverB.get_state(k, i));
        if (k < N) {
            for (int i = 0; i < CarModel::NU; ++i)
                EXPECT_EQ(solverA.get_control(k, i), solverB.get_control(k, i));
        }
        for (int i = 0; i < CarModel::NC; ++i) {
            EXPECT_EQ(solverA.get_slack(k, i), solverB.get_slack(k, i));
            EXPECT_EQ(solverA.get_dual(k, i), solverB.get_dual(k, i));
        }
        for (int i = 0; i < CarModel::NP; ++i)
            EXPECT_EQ(solverA.get_parameter(k, i), solverB.get_parameter(k, i));
    }
}

TEST(SerializerTest, TruncatedFileRejected)
{
    int N = 3;
    MiniSolver<CarModel, 10> solver(N, Backend::CPU_SERIAL);
    solver.set_dt(0.1);
    solver.set_initial_state("x", 1.0);
    solver.set_parameter(0, "v_ref", 5.0);
    solver.solve();

    std::string filename = MakeUniqueTestFilename("test_truncated", ".bin");
    ASSERT_TRUE((minisolver::SolverSerializer<CarModel, 10>::save_case(filename, solver)));

    std::ifstream in(filename, std::ios::binary);
    ASSERT_TRUE(in.good());
    std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();
    ASSERT_GT(bytes.size(), 8u);
    bytes.resize(bytes.size() - 8);

    std::ofstream out(filename, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(out.good());
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    out.close();

    SolverConfig cfg2;
    cfg2.mu_init = 0.123;
    cfg2.barrier_strategy = BarrierStrategy::MONOTONE;
    MiniSolver<CarModel, 10> solver2(1, Backend::CPU_SERIAL, cfg2);
    EXPECT_EQ(solver2.get_horizon(), 1);
    EXPECT_DOUBLE_EQ(solver2.get_config().mu_init, 0.123);

    EXPECT_FALSE((minisolver::SolverSerializer<CarModel, 10>::load_case(filename, solver2)));

    // Load should be atomic: on failure, the solver state should remain unchanged.
    EXPECT_EQ(solver2.get_horizon(), 1);
    EXPECT_DOUBLE_EQ(solver2.get_config().mu_init, 0.123);

    std::remove(filename.c_str());
}

TEST(SerializerTest, OversizeHorizonRejected)
{
    // Save with MAX_N=50 and N=20, then attempt to load with MAX_N=10.
    // The loader should refuse to truncate.
    int N = 20;
    MiniSolver<CarModel, 50> solver_big(N, Backend::CPU_SERIAL);
    solver_big.set_dt(0.1);

    std::string filename = MakeUniqueTestFilename("test_oversize", ".bin");
    ASSERT_TRUE((minisolver::SolverSerializer<CarModel, 50>::save_case(filename, solver_big)));

    MiniSolver<CarModel, 10> solver_small(5, Backend::CPU_SERIAL);
    EXPECT_FALSE((minisolver::SolverSerializer<CarModel, 10>::load_case(filename, solver_small)));

    std::remove(filename.c_str());
}
