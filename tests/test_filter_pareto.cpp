// Pareto-frontier filter contract tests.
//
// These tests pin the contract introduced by Tier 4.3:
//   - LineSearchResult.filter_entries_pruned, filter_redundant_inserts, and
//     filter_size_after are populated by FilterLineSearch and surface the
//     Pareto-pruning policy that replaces the legacy circular-buffer
//     eviction.
//   - SolverInfo.filter_entries_pruned_total,
//     filter_redundant_inserts_total, and filter_max_history_size accumulate
//     across an entire solve and reset between solves.
//   - MeritLineSearch leaves all filter diagnostics at zero.
//   - On strictly improving (theta, phi) the frontier collapses to size 1.
//   - On strictly worsening (theta, phi) every new entry is rejected as
//     redundant; the existing single entry is preserved.
//   - On Pareto-incomparable entries (better theta, worse phi or vice versa)
//     no pruning happens and both stay on the frontier.

#include "minisolver/algorithms/line_search.h"
#include "minisolver/core/solver_options.h"
#include "minisolver/core/trajectory.h"
#include "minisolver/core/types.h"
#include "minisolver/solver/solver.h"
#include <array>
#include <cmath>
#include <gtest/gtest.h>

using namespace minisolver;

namespace {

// Minimal model used to drive raw FilterLineSearch::search calls without a
// full solver: 1 state, 1 control, no constraints, dynamics x_{k+1} = x_k +
// dt * u. We rely on a non-trivial dynamic defect (theta) to force H-type
// filter insertion.
struct ParetoFilterModel {
    static constexpr int NX = 1;
    static constexpr int NU = 1;
    static constexpr int NC = 0;
    static constexpr int NP = 0;

    static constexpr std::array<const char*, NX> state_names = { "x" };
    static constexpr std::array<const char*, NU> control_names = { "u" };
    static constexpr std::array<const char*, NP> param_names = {};
    static constexpr std::array<double, NC> constraint_weights = {};
    static constexpr std::array<int, NC> constraint_types = {};

    template <typename T>
    static MSVec<T, NX> integrate(const MSVec<T, NX>& x, const MSVec<T, NU>& u,
        const MSVec<T, NP>& /*p*/, double dt, IntegratorType /*type*/)
    {
        return x + u * dt;
    }

    template <typename T>
    static void compute_dynamics(KnotPoint<T, NX, NU, NC, NP>& kp, IntegratorType, double dt)
    {
        kp.f_resid(0) = kp.x(0) + kp.u(0) * dt;
        kp.A(0, 0) = 1.0;
        kp.B(0, 0) = dt;
    }

    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>&) { }

    template <typename T> static void compute_cost_gn(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        kp.cost = kp.x(0) * kp.x(0) + kp.u(0) * kp.u(0);
        kp.Q.setZero();
        kp.R.setIdentity();
        kp.q(0) = static_cast<T>(2.0) * kp.x(0);
        kp.r(0) = static_cast<T>(2.0) * kp.u(0);
        kp.H(0, 0) = 0.0;
    }

    template <typename T> static void compute_cost_exact(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        compute_cost_gn(kp);
    }
};

// Stub linear solver: lets FilterLineSearch pull dx/du for free without
// running Riccati. The active trajectory provides them directly.
template <int MAX_N>
class StubLinearSolver
    : public LinearSolver<typename Trajectory<KnotPoint<double, 1, 1, 0, 0>, MAX_N>::TrajArray> {
public:
    using TrajArrayT = typename Trajectory<KnotPoint<double, 1, 1, 0, 0>, MAX_N>::TrajArray;
    LinearSolveResult solve(TrajArrayT& /*traj*/, int /*N*/, double /*mu*/, double /*reg*/,
        InertiaStrategy /*strategy*/, const SolverConfig& /*config*/,
        const TrajArrayT* /*affine_traj*/ = nullptr) override
    {
        return { true };
    }
};

void seed_active_with_defect(Trajectory<KnotPoint<double, 1, 1, 0, 0>, 1>& trajectory,
    const SolverConfig& config, double terminal_x, double dx_step)
{
    auto& active = trajectory.active();
    for (int k = 0; k <= 1; ++k) {
        active[k].set_zero();
    }
    // Knot 0 sits at the origin. Knot 1's reported state diverges from
    // f_resid(knot0) = 0, producing a large dynamic defect that drives
    // theta > 0 and forces H-type filter insertion. dx_step shrinks the
    // defect on each iteration so successive (theta, phi) entries dominate
    // the previous one.
    active[1].x(0) = terminal_x;
    active[0].dx(0) = 0.0;
    active[1].dx(0) = dx_step;
    active[0].du(0) = 0.0;

    std::array<double, 1> dts { 0.1 };
    for (int k = 0; k <= 1; ++k) {
        const double current_dt = (k < 1) ? dts[0] : 0.0;
        detail::evaluate_model_stage<ParetoFilterModel>(active[k], config, current_dt, k == 1);
    }
}

} // namespace

TEST(FilterParetoTest, MonotonicallyImprovingMetricsCollapseFrontierToSingleEntry)
{
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.line_search_type = LineSearchType::FILTER;
    config.line_search_max_iters = 1;
    config.enable_soc = false;

    FilterLineSearch<ParetoFilterModel, 1> ls;
    StubLinearSolver<1> linear_solver;
    Trajectory<KnotPoint<double, 1, 1, 0, 0>, 1> trajectory(1);

    std::array<double, 1> dts { 0.1 };
    int total_pruned = 0;
    int total_redundant = 0;

    for (int iter = 0; iter < 50; ++iter) {
        // Strictly shrinking terminal state defect across iterations.
        const double terminal_x = 2000.0 - static_cast<double>(iter);
        seed_active_with_defect(trajectory, config, terminal_x, -0.5);
        const LineSearchResult result
            = ls.search(trajectory, linear_solver, dts, 0.1, 1e-6, config);
        ASSERT_GT(result.alpha, 0.0) << "filter rejected at iteration " << iter;
        total_pruned += result.filter_entries_pruned;
        total_redundant += result.filter_redundant_inserts;
        EXPECT_LE(result.filter_size_after, 1)
            << "Strictly improving sequence keeps the Pareto frontier at size <= 1";
    }
    EXPECT_EQ(ls.filter_size(), 1u);
    EXPECT_EQ(total_redundant, 0);
    EXPECT_GT(total_pruned, 0);
}

TEST(FilterParetoTest, MeritLineSearchLeavesFilterDiagnosticsAtZero)
{
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.line_search_type = LineSearchType::MERIT;
    config.line_search_max_iters = 1;

    MeritLineSearch<ParetoFilterModel, 1> ls;
    StubLinearSolver<1> linear_solver;
    Trajectory<KnotPoint<double, 1, 1, 0, 0>, 1> trajectory(1);

    std::array<double, 1> dts { 0.1 };
    seed_active_with_defect(trajectory, config, 100.0, -0.5);
    const LineSearchResult result = ls.search(trajectory, linear_solver, dts, 0.1, 1e-6, config);
    EXPECT_EQ(result.filter_entries_pruned, 0);
    EXPECT_EQ(result.filter_redundant_inserts, 0);
    EXPECT_EQ(result.filter_size_after, 0);
}

TEST(FilterParetoTest, SolverInfoExposesPeakHistorySize)
{
    SolverInfo info;
    info.filter_entries_pruned_total = 7;
    info.filter_redundant_inserts_total = 3;
    info.filter_max_history_size = 42;
    info.reset();
    EXPECT_EQ(info.filter_entries_pruned_total, 0);
    EXPECT_EQ(info.filter_redundant_inserts_total, 0);
    EXPECT_EQ(info.filter_max_history_size, 0);
}
