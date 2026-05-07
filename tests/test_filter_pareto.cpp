// Pareto-frontier filter contract tests.
//
// These tests pin the contract introduced by Tier 4.3 and the gamma_phi
// correctness fix that followed it:
//
//   - LineSearchResult.filter_entries_pruned, filter_redundant_inserts, and
//     filter_size_after are populated by FilterLineSearch and surface the
//     Pareto-pruning policy that replaces the legacy circular-buffer
//     eviction. (MonotonicallyImprovingMetricsCollapseFrontierToSingleEntry
//     and the gamma_phi tests below).
//   - SolverInfo.filter_entries_pruned_total,
//     filter_redundant_inserts_total, and filter_max_history_size are
//     cleared by SolverInfo::reset() so they reflect the current solve
//     only. (SolverInfoExposesPeakHistorySize.)
//   - MeritLineSearch leaves all filter diagnostics at zero.
//     (MeritLineSearchLeavesFilterDiagnosticsAtZero.)
//   - On strictly improving (theta, phi) the frontier collapses to size 1
//     and the redundant-insert counter stays zero.
//     (MonotonicallyImprovingMetricsCollapseFrontierToSingleEntry.)
//   - On strictly worsening (theta, phi) every new entry is rejected as
//     redundant; the existing single entry is preserved.
//     (StrictlyWorseningSequenceMarksEveryNewEntryRedundant.)
//   - gamma_phi-aware dominance: the IPOPT-style forbidden region of an
//     entry (theta_e, phi_e) is theta_c >= (1-gamma_theta) theta_e AND
//     phi_c >= phi_e - gamma_phi * theta_e. The Pareto check therefore has
//     to compare psi = phi - gamma_phi * theta on the ordinate, not phi.
//     Plain (theta, phi) Pareto silently over-prunes when gamma_phi > 0;
//     the corrected (theta, psi) Pareto:
//       * preserves entries that are incomparable in (theta, psi)
//         (NonZeroGammaPhiPreservesParetoIncomparableEntry);
//       * still prunes entries dominated in (theta, psi)
//         (NonZeroGammaPhiPrunesEntriesDominatedInPsiSpace);
//       * reduces to plain Pareto when gamma_phi = 0
//         (ZeroGammaPhiBehavesAsPlainPareto).

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

// RED: a previous implementation used plain Pareto on (theta, phi). With
// non-zero gamma_phi, plain Pareto silently widens the filter forbidden region:
// it both over-prunes existing entries and drops new entries as "redundant"
// when they are not. The IPOPT-style filter forbidden region of an entry
// (theta_e, phi_e) is theta_c >= (1-gamma_theta) theta_e AND
// phi_c >= phi_e - gamma_phi * theta_e. For dominance to hold, both axes have
// to use this shifted reading; psi(e) = phi_e - gamma_phi * theta_e is the
// correct ordinate.
//
// Setup pinned by this test:
//   gamma_phi = 5
//   e1 = (theta=0,   phi=10)  -> psi=10
//   e2 = (theta=1,   phi=11)  -> psi=6
// In plain (theta, phi) Pareto, e1 dominates e2 (theta_e1<=theta_e2 and
// phi_e1<=phi_e2), so the legacy code rejected e2 as redundant. In the
// shifted (theta, psi) Pareto e1 does NOT dominate e2 (psi(e1)=10>6=psi(e2));
// e2 must be inserted.
//
// The trial (theta=0.6, phi=7) is forbidden by e2 (theta>0.5 and phi>6) but
// not by e1 alone (phi=7 < 10). Without e2 in the filter, the legacy code
// silently accepted forbidden trials.
TEST(FilterParetoTest, NonZeroGammaPhiPreservesParetoIncomparableEntry)
{
    FilterLineSearch<ParetoFilterModel, 1> ls;
    const double gamma_phi = 5.0;

    const auto r1 = ls.try_insert_h_type_for_testing(0.0, 10.0, gamma_phi);
    EXPECT_EQ(r1.filter_redundant_inserts, 0);
    EXPECT_EQ(r1.filter_entries_pruned, 0);
    EXPECT_EQ(r1.filter_size_after, 1);

    const auto r2 = ls.try_insert_h_type_for_testing(1.0, 11.0, gamma_phi);
    EXPECT_EQ(r2.filter_redundant_inserts, 0)
        << "psi(e1)=10, psi(e2)=6; e1 does not dominate e2 in (theta, psi) space";
    EXPECT_EQ(r2.filter_entries_pruned, 0)
        << "e2 has theta_e2 > theta_e1; the new entry must not prune e1";
    ASSERT_EQ(r2.filter_size_after, 2);

    EXPECT_EQ(ls.filter_entry_for_testing(0).first, 0.0);
    EXPECT_EQ(ls.filter_entry_for_testing(0).second, 10.0);
    EXPECT_EQ(ls.filter_entry_for_testing(1).first, 1.0);
    EXPECT_EQ(ls.filter_entry_for_testing(1).second, 11.0);
}

// Symmetric guard: when the new entry psi-dominates an existing one, prune
// the old entry. With plain Pareto the dominance test missed this case any
// time the new entry had larger phi but a much smaller psi via gamma_phi.
TEST(FilterParetoTest, NonZeroGammaPhiPrunesEntriesDominatedInPsiSpace)
{
    FilterLineSearch<ParetoFilterModel, 1> ls;
    const double gamma_phi = 5.0;

    // Seed with an entry that has small theta and large phi, hence large psi.
    const auto r_seed = ls.try_insert_h_type_for_testing(0.5, 12.0, gamma_phi);
    ASSERT_EQ(r_seed.filter_size_after, 1);
    // psi_seed = 12 - 5*0.5 = 9.5

    // New entry has SMALLER theta, but larger phi than seed -> plain Pareto
    // would refuse the new entry as "dominated by something that doesn't
    // exist" or fail to prune the seed because phi_new > phi_seed. In
    // (theta, psi) space psi_new = 13 - 5*0 = 13 > psi_seed = 9.5, so the
    // seed is NOT dominated and the new one is NOT redundant either; both
    // stay on the frontier.
    const auto r_incomparable = ls.try_insert_h_type_for_testing(0.0, 13.0, gamma_phi);
    EXPECT_EQ(r_incomparable.filter_redundant_inserts, 0);
    EXPECT_EQ(r_incomparable.filter_entries_pruned, 0);
    EXPECT_EQ(r_incomparable.filter_size_after, 2);

    // Now insert an entry that genuinely psi-dominates the seed: theta_new <
    // theta_seed AND psi_new < psi_seed. theta_new=0.1, phi_new=4 ->
    // psi_new = 4 - 5*0.1 = 3.5 < 9.5. The new entry has theta=0.1 < 0.5 and
    // psi=3.5 < 9.5, so it psi-dominates the seed. The seed must be pruned;
    // the (theta=0, phi=13, psi=13) entry is incomparable so it stays.
    const auto r_dominator = ls.try_insert_h_type_for_testing(0.1, 4.0, gamma_phi);
    EXPECT_EQ(r_dominator.filter_redundant_inserts, 0);
    EXPECT_EQ(r_dominator.filter_entries_pruned, 1)
        << "(theta=0.1, psi=3.5) psi-dominates the seed (theta=0.5, psi=9.5)";
    EXPECT_EQ(r_dominator.filter_size_after, 2);
}

// Defense: gamma_phi = 0 still collapses to plain (theta, phi) Pareto, so
// strictly improving sequences still collapse to a single entry.
TEST(FilterParetoTest, ZeroGammaPhiBehavesAsPlainPareto)
{
    FilterLineSearch<ParetoFilterModel, 1> ls;
    const double gamma_phi = 0.0;

    ls.try_insert_h_type_for_testing(2.0, 20.0, gamma_phi);
    const auto r_strict = ls.try_insert_h_type_for_testing(1.0, 10.0, gamma_phi);
    EXPECT_EQ(r_strict.filter_entries_pruned, 1);
    EXPECT_EQ(r_strict.filter_redundant_inserts, 0);
    EXPECT_EQ(r_strict.filter_size_after, 1);

    const auto r_redundant = ls.try_insert_h_type_for_testing(2.0, 20.0, gamma_phi);
    EXPECT_EQ(r_redundant.filter_redundant_inserts, 1);
    EXPECT_EQ(r_redundant.filter_entries_pruned, 0);
    EXPECT_EQ(r_redundant.filter_size_after, 1);
}

// File-header contract case "strictly worsening (theta, phi) marks every
// new entry redundant" was previously documented but not pinned. The
// filter logic: once an entry (theta_seed, phi_seed) sits in the
// frontier, any later attempt with both theta and phi greater than or
// equal to the seed is redundant (the seed dominates the trial in both
// axes), so it must NOT enter the filter and must NOT prune anything.
// With gamma_phi = 0 (plain Pareto), this is the simplest form of the
// contract; the gamma_phi > 0 cases are pinned in
// NonZeroGammaPhi* above.
TEST(FilterParetoTest, StrictlyWorseningSequenceMarksEveryNewEntryRedundant)
{
    FilterLineSearch<ParetoFilterModel, 1> ls;
    const double gamma_phi = 0.0;

    const auto r_seed = ls.try_insert_h_type_for_testing(1.0, 10.0, gamma_phi);
    ASSERT_FALSE(r_seed.filter_redundant_inserts > 0);
    ASSERT_EQ(r_seed.filter_size_after, 1);

    int total_redundant = 0;
    int total_pruned = 0;
    for (int i = 1; i <= 5; ++i) {
        const double theta = 1.0 + 0.5 * static_cast<double>(i);
        const double phi = 10.0 + static_cast<double>(i);
        const auto r = ls.try_insert_h_type_for_testing(theta, phi, gamma_phi);
        EXPECT_EQ(r.filter_redundant_inserts, 1)
            << "(theta=" << theta << ", phi=" << phi
            << ") is dominated by the seed and must be redundant";
        EXPECT_EQ(r.filter_entries_pruned, 0)
            << "redundant inserts must not prune the surviving entry";
        EXPECT_EQ(r.filter_size_after, 1) << "the seed must be preserved, frontier size stays at 1";
        total_redundant += r.filter_redundant_inserts;
        total_pruned += r.filter_entries_pruned;
    }
    EXPECT_EQ(total_redundant, 5);
    EXPECT_EQ(total_pruned, 0);
    EXPECT_EQ(ls.filter_size(), 1u);
}
