#include "../examples/01_car_tutorial/generated/car_model.h"
#include "minisolver/debug/solver_snapshot.h"
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

bool FileExists(const std::string& filename)
{
    std::ifstream in(filename, std::ios::binary);
    return in.good();
}

SolverConfig MakeNonDefaultConfig()
{
    SolverConfig config;
    config.backend = Backend::GPU_PCR;
    config.initialization = InitializationMode::REUSE_PRIMAL_DUAL;
    config.warm_start_barrier = WarmStartBarrierMode::FROM_COMPLEMENTARITY_GAP;
    config.warm_start_regularization = WarmStartRegularizationMode::DECAY_PREVIOUS_REG;
    config.termination_profile = TerminationProfile::ACCEPTABLE_NMPC;
    config.constraint_scaling = ConstraintScalingMethod::ROW_INF_NORM;
    config.objective_scaling = ObjectiveScalingMethod::HESSIAN_GERSHGORIN;
    config.problem_scaling = ProblemScalingMethod::RUIZ_EQUILIBRATION;
    config.constraint_row_scale_min = 2e-5;
    config.constraint_row_scale_max = 3e3;
    config.objective_scale_min = 4e-5;
    config.objective_scale_max = 0.25;

    config.integrator = IntegratorType::EULER_IMPLICIT;
    config.default_dt = 0.123;
    config.newton_config.max_iters = 9;
    config.newton_config.tol = 3e-9;
    config.newton_config.regularization = 4e-11;

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
    config.filter_theta_max_factor = 77.0;
    config.armijo_c1 = 7e-5;

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
    config.restoration_sufficient_decrease_factor = 0.75;

    config.max_iters = 17;
    config.print_level = PrintLevel::DEBUG;
    config.enable_profiling = false;

    config.hessian_approximation = HessianApproximation::EXACT;
    config.direction_refinement = DirectionRefinementMode::DYNAMICS_DEFECT_ROLLOUT;
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

template <typename SnapshotIO> void ExpectConfigEq(const SolverConfig& a, const SolverConfig& b)
{
    EXPECT_TRUE(SnapshotIO::config_equal(a, b));
}

// A minimal L1 soft-constraint model to verify that snapshots preserve soft_s correctly.
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

struct L1SoftModelAltNames : L1SoftModel {
    static constexpr std::array<const char*, NX> state_names = { "x_alt" };
    static constexpr std::array<const char*, NU> control_names = { "u_alt" };
    static constexpr std::array<const char*, NP> param_names = {};
};
} // namespace

TEST(SolverSnapshotTest, CaptureAndSaveAndLoad)
{
    int N = 20;
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    // Set some non-default config values to verify they are saved correctly
    config.mu_init = 0.5;
    config.tol_con = 1e-4;
    config.barrier_strategy = BarrierStrategy::MEHROTRA;

    using MySolver = MiniSolver<CarModel, 50>;
    using SnapshotIO = SolverSnapshotIO<CarModel, 50>;

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
        for (int i = 0; i < CarModel::NX; ++i) {
            solver.set_state_guess(k, i, k * 1.0);
        }
        if (k < N) {
            for (int i = 0; i < CarModel::NU; ++i) {
                solver.set_control_guess(k, i, k * 0.5);
            }
        }
        for (int i = 0; i < CarModel::NC; ++i) {
            solver.set_slack_guess(k, i, 0.1);
            solver.set_dual_guess(k, i, 0.2);
        }
    }

    // 1. Capture State (Test New Interface)
    auto snapshot = SnapshotIO::capture_snapshot(solver, SolverStatus::FEASIBLE);
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
    SnapshotResult save_result = SnapshotIO::save_snapshot(filename, snapshot);
    EXPECT_EQ(save_result.status, SnapshotStatus::OK);

    // 3. Load into new solver
    MySolver solver2(10, Backend::CPU_SERIAL); // Initialize with different N
    SnapshotResult load_result = SnapshotIO::load_case(filename, solver2);
    EXPECT_EQ(load_result.status, SnapshotStatus::OK);

    // 4. Verify Loaded Data
    EXPECT_EQ(solver2.get_horizon(), N);
    EXPECT_EQ(solver2.get_config().mu_init, 0.5);
    EXPECT_EQ(solver2.get_config().barrier_strategy, BarrierStrategy::MEHROTRA);
    auto loaded_snapshot = SnapshotIO::capture_snapshot(solver2);
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

TEST(SolverSnapshotTest, SnapshotPreservesSoftS_L1)
{
    L1SoftModel::constraint_types[0] = 1; // L1
    L1SoftModel::constraint_weights[0] = 1.0; // w=1

    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.max_iters = 0; // Presolve initializes soft_s; no need to iterate.

    using Solver = MiniSolver<L1SoftModel, 5>;
    using SnapshotIO = SolverSnapshotIO<L1SoftModel, 5>;

    Solver solver(1, Backend::CPU_SERIAL, config);
    solver.set_dt(1.0);
    solver.set_initial_state("x", 0.0);

    solver.solve();

    auto snapA = SnapshotIO::capture_snapshot(solver);
    ASSERT_EQ(snapA.N, 1);
    ASSERT_EQ(snapA.trajectory.size(), 2u);
    const double soft_s_A = snapA.trajectory[0].soft_s[0];
    ASSERT_TRUE(std::isfinite(soft_s_A));
    ASSERT_GT(soft_s_A, 0.0);

    const std::string filename = MakeUniqueTestFilename("test_softs_snapshot", ".bin");
    ASSERT_EQ(SnapshotIO::save_snapshot(filename, snapA).status, SnapshotStatus::OK);

    Solver solver2(1, Backend::CPU_SERIAL, config);
    ASSERT_EQ(SnapshotIO::load_case(filename, solver2).status, SnapshotStatus::OK);

    auto snapB = SnapshotIO::capture_snapshot(solver2);
    ASSERT_EQ(snapB.N, 1);
    const double soft_s_B = snapB.trajectory[0].soft_s[0];
    EXPECT_DOUBLE_EQ(soft_s_B, soft_s_A);

    std::remove(filename.c_str());
}

TEST(SolverSnapshotTest, SnapshotPreservesAllConfigFields)
{
    SolverConfig config = MakeNonDefaultConfig();

    // Create Solver A (config.backend is overwritten by ctor backend argument).
    MiniSolver<CarModel, 10> solverA(2, config.backend, config);
    ASSERT_EQ(solverA.get_config().backend, Backend::GPU_PCR);

    std::string filename = MakeUniqueTestFilename("test_config_snapshot", ".bin");
    using SnapshotIO = minisolver::SolverSnapshotIO<CarModel, 10>;
    ASSERT_EQ(SnapshotIO::save_case(filename, solverA).status, SnapshotStatus::OK);

    MiniSolver<CarModel, 10> solverB(2, Backend::CPU_SERIAL);
    SnapshotLoadOptions options;
    options.backend_policy = SnapshotBackendPolicy::UseSnapshotBackend;
    ASSERT_EQ(SnapshotIO::load_case(filename, solverB, options).status, SnapshotStatus::OK);

    const SolverConfig& cfgA = solverA.get_config();
    const SolverConfig& cfgB = solverB.get_config();
    ExpectConfigEq<SnapshotIO>(cfgA, cfgB);

    std::remove(filename.c_str());
}

TEST(SolverSnapshotTest, LoadRejectsInvalidSnapshotConfigAtomically)
{
    MiniSolver<CarModel, 10> solverA(2, Backend::CPU_SERIAL);
    using SnapshotIO = minisolver::SolverSnapshotIO<CarModel, 10>;
    auto snapshot = SnapshotIO::capture_snapshot(solverA);
    snapshot.config.reg_scale_up = 1.0;

    std::string filename = MakeUniqueTestFilename("test_invalid_config_snapshot", ".bin");
    ASSERT_EQ(SnapshotIO::save_snapshot(filename, snapshot).status, SnapshotStatus::OK);

    SolverConfig preserved;
    preserved.mu_init = 0.321;
    MiniSolver<CarModel, 10> solverB(1, Backend::CPU_SERIAL, preserved);
    SnapshotResult load_result = SnapshotIO::load_case(filename, solverB);
    EXPECT_EQ(load_result.status, SnapshotStatus::InvalidConfig);
    EXPECT_EQ(solverB.get_horizon(), 1);
    EXPECT_DOUBLE_EQ(solverB.get_config().mu_init, 0.321);
    EXPECT_GT(solverB.get_config().reg_scale_up, 1.0);

    std::remove(filename.c_str());
}

TEST(SolverSnapshotTest, LoadKeepsConstructedBackendByDefault)
{
    SolverConfig config = MakeNonDefaultConfig();
    MiniSolver<CarModel, 10> solverA(2, Backend::GPU_PCR, config);
    ASSERT_EQ(solverA.get_config().backend, Backend::GPU_PCR);

    std::string filename = MakeUniqueTestFilename("test_backend_policy_snapshot", ".bin");
    using SnapshotIO = minisolver::SolverSnapshotIO<CarModel, 10>;
    ASSERT_EQ(SnapshotIO::save_case(filename, solverA).status, SnapshotStatus::OK);

    MiniSolver<CarModel, 10> solverB(2, Backend::CPU_SERIAL);
    ASSERT_EQ(SnapshotIO::load_case(filename, solverB).status, SnapshotStatus::OK);
    EXPECT_EQ(solverB.get_config().backend, Backend::CPU_SERIAL);

    std::remove(filename.c_str());
}

TEST(SolverSnapshotTest, LoadCanOverrideBackendExplicitly)
{
    MiniSolver<CarModel, 10> solverA(2, Backend::CPU_SERIAL);

    std::string filename = MakeUniqueTestFilename("test_backend_override_snapshot", ".bin");
    using SnapshotIO = minisolver::SolverSnapshotIO<CarModel, 10>;
    ASSERT_EQ(SnapshotIO::save_case(filename, solverA).status, SnapshotStatus::OK);

    MiniSolver<CarModel, 10> solverB(2, Backend::CPU_SERIAL);
    SnapshotLoadOptions options;
    options.backend_policy = SnapshotBackendPolicy::OverrideWith;
    options.override_backend = Backend::GPU_MPX;
    ASSERT_EQ(SnapshotIO::load_case(filename, solverB, options).status, SnapshotStatus::OK);
    EXPECT_EQ(solverB.get_config().backend, Backend::GPU_MPX);

    std::remove(filename.c_str());
}

TEST(SolverSnapshotTest, SaveFailureSnapshotIsNoOpForSuccessfulStatuses)
{
    MiniSolver<CarModel, 10> solver(2, Backend::CPU_SERIAL);
    using SnapshotIO = minisolver::SolverSnapshotIO<CarModel, 10>;
    const auto pre_solve = SnapshotIO::capture_snapshot(solver);

    const std::string optimal_file = MakeUniqueTestFilename("test_snapshot_optimal_noop", ".bin");
    std::remove(optimal_file.c_str());
    EXPECT_EQ(
        SnapshotIO::save_failure_snapshot(optimal_file, pre_solve, SolverStatus::OPTIMAL).status,
        SnapshotStatus::OK);
    EXPECT_FALSE(FileExists(optimal_file));

    const std::string feasible_file = MakeUniqueTestFilename("test_snapshot_feasible_noop", ".bin");
    std::remove(feasible_file.c_str());
    EXPECT_EQ(
        SnapshotIO::save_failure_snapshot(feasible_file, pre_solve, SolverStatus::FEASIBLE).status,
        SnapshotStatus::OK);
    EXPECT_FALSE(FileExists(feasible_file));
}

TEST(SolverSnapshotTest, SaveFailureSnapshotWritesPreSolveStateOnFailure)
{
    MiniSolver<CarModel, 10> solver(2, Backend::CPU_SERIAL);
    solver.set_initial_state("x", 1.25);
    using SnapshotIO = minisolver::SolverSnapshotIO<CarModel, 10>;
    const auto pre_solve = SnapshotIO::capture_snapshot(solver);

    solver.set_initial_state("x", 9.5);

    const std::string filename = MakeUniqueTestFilename("test_snapshot_failure_presolve", ".bin");
    std::remove(filename.c_str());
    ASSERT_EQ(SnapshotIO::save_failure_snapshot(filename, pre_solve, SolverStatus::MAX_ITER).status,
        SnapshotStatus::OK);
    ASSERT_TRUE(FileExists(filename));

    MiniSolver<CarModel, 10> loaded(2, Backend::CPU_SERIAL);
    ASSERT_EQ(SnapshotIO::load_case(filename, loaded).status, SnapshotStatus::OK);
    EXPECT_DOUBLE_EQ(loaded.get_state(0, 0), 1.25);

    std::remove(filename.c_str());
}

TEST(SolverSnapshotTest, LoadRejectsSameDimensionDifferentModelFingerprint)
{
    L1SoftModel::constraint_types[0] = 0;
    L1SoftModel::constraint_weights[0] = 0.0;

    MiniSolver<L1SoftModel, 5> solverA(1, Backend::CPU_SERIAL);
    const std::string filename = MakeUniqueTestFilename("test_model_fingerprint", ".bin");
    ASSERT_EQ((SolverSnapshotIO<L1SoftModel, 5>::save_case(filename, solverA).status),
        SnapshotStatus::OK);

    MiniSolver<L1SoftModelAltNames, 5> solverB(1, Backend::CPU_SERIAL);
    EXPECT_EQ((SolverSnapshotIO<L1SoftModelAltNames, 5>::load_case(filename, solverB).status),
        SnapshotStatus::ModelMismatch);

    std::remove(filename.c_str());
}

TEST(SolverSnapshotTest, RejectsOldFormatMagic)
{
    const std::string filename = MakeUniqueTestFilename("test_old_magic", ".bin");
    {
        std::ofstream out(filename, std::ios::binary);
        ASSERT_TRUE(out.good());
        out.write("MINISOLV_3", 10); // previous config layout had the dead tol_grad field.
    }

    MiniSolver<CarModel, 10> solver(2, Backend::CPU_SERIAL);
    EXPECT_EQ((minisolver::SolverSnapshotIO<CarModel, 10>::load_case(filename, solver).status),
        SnapshotStatus::UnsupportedVersion);
    std::remove(filename.c_str());
}

TEST(SolverSnapshotTest, FullRoundTrip)
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

    // 2. Save snapshot to file
    std::string filename = MakeUniqueTestFilename("test_roundtrip", ".bin");
    minisolver::SolverSnapshotIO<CarModel, 10>::save_case(filename, solverA);

    // 3. Load snapshot into Solver B
    MiniSolver<CarModel, 10> solverB(N, Backend::CPU_SERIAL, config);
    SnapshotResult load_ok
        = minisolver::SolverSnapshotIO<CarModel, 10>::load_case(filename, solverB);
    EXPECT_EQ(load_ok.status, SnapshotStatus::OK);

    // 4. Cleanup
    std::remove(filename.c_str());

    // 5. Compare State (Bit-Exact)
    EXPECT_EQ(solverA.get_horizon(), solverB.get_horizon());
    EXPECT_EQ(solverA.get_iteration_count(), solverB.get_iteration_count());

    auto stateA = minisolver::SolverSnapshotIO<CarModel, 10>::capture_snapshot(solverA);
    auto stateB = minisolver::SolverSnapshotIO<CarModel, 10>::capture_snapshot(solverB);
    EXPECT_DOUBLE_EQ(stateA.mu, stateB.mu);
    EXPECT_DOUBLE_EQ(stateA.reg, stateB.reg);
    EXPECT_DOUBLE_EQ(stateA.total_cost, stateB.total_cost);

    for (int k = 0; k <= N; ++k) {
        for (int i = 0; i < CarModel::NX; ++i) {
            EXPECT_EQ(solverA.get_state(k, i), solverB.get_state(k, i));
        }
        if (k < N) {
            for (int i = 0; i < CarModel::NU; ++i) {
                EXPECT_EQ(solverA.get_control(k, i), solverB.get_control(k, i));
            }
        }
        for (int i = 0; i < CarModel::NC; ++i) {
            EXPECT_EQ(solverA.get_slack(k, i), solverB.get_slack(k, i));
            EXPECT_EQ(solverA.get_dual(k, i), solverB.get_dual(k, i));
        }
        for (int i = 0; i < CarModel::NP; ++i) {
            EXPECT_EQ(solverA.get_parameter(k, i), solverB.get_parameter(k, i));
        }
    }
}

TEST(SolverSnapshotTest, TruncatedFileRejected)
{
    int N = 3;
    MiniSolver<CarModel, 10> solver(N, Backend::CPU_SERIAL);
    solver.set_dt(0.1);
    solver.set_initial_state("x", 1.0);
    solver.set_parameter(0, "v_ref", 5.0);
    solver.solve();

    std::string filename = MakeUniqueTestFilename("test_truncated", ".bin");
    ASSERT_EQ((minisolver::SolverSnapshotIO<CarModel, 10>::save_case(filename, solver).status),
        SnapshotStatus::OK);

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

    EXPECT_EQ((minisolver::SolverSnapshotIO<CarModel, 10>::load_case(filename, solver2).status),
        SnapshotStatus::TruncatedFile);

    // Load should be atomic: on failure, the solver state should remain unchanged.
    EXPECT_EQ(solver2.get_horizon(), 1);
    EXPECT_DOUBLE_EQ(solver2.get_config().mu_init, 0.123);

    std::remove(filename.c_str());
}

TEST(SolverSnapshotTest, TrailingBytesRejectedByDefault)
{
    MiniSolver<CarModel, 10> solver(3, Backend::CPU_SERIAL);
    solver.set_dt(0.1);

    std::string filename = MakeUniqueTestFilename("test_trailing_bytes", ".bin");
    using SnapshotIO = minisolver::SolverSnapshotIO<CarModel, 10>;
    ASSERT_EQ(SnapshotIO::save_case(filename, solver).status, SnapshotStatus::OK);

    {
        std::ofstream out(filename, std::ios::binary | std::ios::app);
        ASSERT_TRUE(out.good());
        const std::array<char, 4> extra = { 'J', 'U', 'N', 'K' };
        out.write(extra.data(), static_cast<std::streamsize>(extra.size()));
    }

    MiniSolver<CarModel, 10> strict_loader(1, Backend::CPU_SERIAL);
    EXPECT_EQ(SnapshotIO::load_case(filename, strict_loader).status, SnapshotStatus::TrailingBytes);

    MiniSolver<CarModel, 10> permissive_loader(1, Backend::CPU_SERIAL);
    SnapshotLoadOptions options;
    options.reject_trailing_bytes = false;
    EXPECT_EQ(
        SnapshotIO::load_case(filename, permissive_loader, options).status, SnapshotStatus::OK);
    EXPECT_EQ(permissive_loader.get_horizon(), 3);

    std::remove(filename.c_str());
}

TEST(SolverSnapshotTest, OversizeHorizonRejected)
{
    // Save with MAX_N=50 and N=20, then attempt to load with MAX_N=10.
    // The loader should refuse to truncate.
    int N = 20;
    MiniSolver<CarModel, 50> solver_big(N, Backend::CPU_SERIAL);
    solver_big.set_dt(0.1);

    std::string filename = MakeUniqueTestFilename("test_oversize", ".bin");
    ASSERT_EQ((minisolver::SolverSnapshotIO<CarModel, 50>::save_case(filename, solver_big).status),
        SnapshotStatus::OK);

    MiniSolver<CarModel, 10> solver_small(5, Backend::CPU_SERIAL);
    EXPECT_EQ(
        (minisolver::SolverSnapshotIO<CarModel, 10>::load_case(filename, solver_small).status),
        SnapshotStatus::HorizonTooLarge);

    std::remove(filename.c_str());
}
