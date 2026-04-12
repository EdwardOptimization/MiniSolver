#include "../examples/01_car_tutorial/generated/car_model.h"
#include "minisolver/algorithms/line_search.h"
#include "minisolver/solver/solver.h"
#include <atomic>
#include <cstdlib>
#include <gtest/gtest.h>
#include <iostream>
#include <new>

// --- Memory Instrumentation ---
std::atomic<bool> g_memory_check_active(false);
std::atomic<int> g_allocation_count(0);

// Weak linkage allows overriding in the executable
void* operator new(size_t size)
{
    if (g_memory_check_active) {
        g_allocation_count++;
    }
    void* p = malloc(size);
    if (!p)
        throw std::bad_alloc();
    return p;
}

// std::get_temporary_buffer (used by libstdc++/gtest internals) allocates via
// `::operator new(size, std::nothrow)` but frees via `::operator delete(void*)`.
// If we override delete to call free(), we must also override the nothrow new
// variants so allocation goes through malloc/free as well, otherwise ASan will
// flag alloc-dealloc mismatch.
void* operator new(size_t size, const std::nothrow_t&) noexcept
{
    if (g_memory_check_active) {
        g_allocation_count++;
    }
    return malloc(size);
}

void* operator new[](size_t size)
{
    if (g_memory_check_active) {
        g_allocation_count++;
    }
    void* p = malloc(size);
    if (!p)
        throw std::bad_alloc();
    return p;
}

void* operator new[](size_t size, const std::nothrow_t&) noexcept
{
    if (g_memory_check_active) {
        g_allocation_count++;
    }
    return malloc(size);
}

void operator delete(void* p) noexcept
{
    free(p);
}

void operator delete(void* p, const std::nothrow_t&) noexcept
{
    free(p);
}

void operator delete[](void* p) noexcept
{
    free(p);
}

void operator delete[](void* p, const std::nothrow_t&) noexcept
{
    free(p);
}

void operator delete(void* p, size_t) noexcept
{
    free(p);
}

void operator delete[](void* p, size_t) noexcept
{
    free(p);
}

using namespace minisolver;

namespace {
// A tiny model to force the Filter line-search into the SOC branch, so we can
// ensure that codepath does not perform heap allocations.
struct SocTriggerModel {
    static constexpr int NX = 1;
    static constexpr int NU = 1;
    static constexpr int NC = 0;
    static constexpr int NP = 0;

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
        kp.f_resid(0) = kp.x(0) + kp.u(0) * dt;
        kp.A(0, 0) = 1.0;
        kp.B(0, 0) = dt;
    }

    template <typename T> static void compute_constraints(KnotPoint<T, NX, NU, NC, NP>& /*kp*/) { }

    template <typename T> static void compute_cost_impl(KnotPoint<T, NX, NU, NC, NP>& kp)
    {
        // Quadratic objective so candidate always increases phi when x moves away from 0.
        kp.cost = kp.x(0) * kp.x(0) + kp.u(0) * kp.u(0);
        kp.q(0) = 2.0 * kp.x(0);
        kp.r(0) = 2.0 * kp.u(0);
        kp.Q(0, 0) = 2.0;
        kp.R(0, 0) = 2.0;
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

template <typename TrajArray> class SocNoAllocLinearSolver final : public LinearSolver<TrajArray> {
public:
    bool solve(TrajArray& /*traj*/, int /*N*/, double /*mu*/, double /*reg*/,
        InertiaStrategy /*strategy*/, const SolverConfig& /*config*/,
        const TrajArray* /*affine_traj*/) override
    {
        return true;
    }

    bool solve_soc(TrajArray& traj, const TrajArray& /*soc_rhs_traj*/, int N, double /*mu*/,
        double /*reg*/, InertiaStrategy /*strategy*/, const SolverConfig& /*config*/) override
    {
        for (int k = 0; k <= N; ++k) {
            traj[k].dx.setZero();
            traj[k].du.setZero();
            traj[k].ds.setZero();
            traj[k].dlam.setZero();
            traj[k].dsoft_s.setZero();
        }
        return true;
    }
};
} // namespace

TEST(MemoryTest, ZeroMalloc_Compliance_Test)
{
    // 1. Setup Solver
    int N = 10;
    SolverConfig config;
    config.print_level = PrintLevel::NONE; // Disable logging to avoid allocations
    config.max_iters = 5;
    config.enable_profiling = false;

    // Ensure we use Custom Matrix if possible, or verify Eigen doesn't allocate for fixed size
    // Eigen fixed-size matrices generally don't allocate on heap.

    MiniSolver<CarModel, 20> solver(N, Backend::CPU_SERIAL, config);
    solver.set_dt(0.1);

    // Warmup / Pre-allocation (if any lazy init exists)
    // The constructor should have done everything.

    // 2. Start Monitoring
    g_allocation_count = 0;
    g_memory_check_active = true;

    // 3. Run Solve
    solver.solve();

    // 4. Stop Monitoring
    g_memory_check_active = false;

    // 5. Verify
    // Note: If using Eigen, verify EIGEN_NO_MALLOC is respected or fixed-size used.
    // If using std::vector inside solver (e.g. for resizing), it might fail.
    // MiniSolver uses std::vector for maps, but those are only read during solve if lookup used.
    // solve() itself shouldn't look up names if we don't call set_param by name inside loop.
    // The loop uses direct access.

    EXPECT_EQ(g_allocation_count, 0)
        << "Detected " << g_allocation_count << " heap allocations during solve() loop!";
}

TEST(MemoryTest, ZeroMalloc_FilterSOC_Path)
{
    // Construct everything before enabling memory instrumentation.
    constexpr int MAX_N = 3;
    using TrajectoryType = Trajectory<KnotPoint<double, SocTriggerModel::NX, SocTriggerModel::NU,
                                          SocTriggerModel::NC, SocTriggerModel::NP>,
        MAX_N>;
    using TrajArray = typename TrajectoryType::TrajArray;

    FilterLineSearch<SocTriggerModel, MAX_N> line_search;
    SocNoAllocLinearSolver<TrajArray> linear_solver;

    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.enable_profiling = false;
    config.enable_soc = true;
    config.enable_line_search_rollout = false;
    config.line_search_max_iters = 2;
    config.soc_trigger_alpha = 0.5;

    TrajectoryType trajectory(/*initial_N=*/1);
    trajectory.N = 1;

    std::array<double, MAX_N> dt_traj;
    dt_traj.fill(1.0);

    auto& active = trajectory.active();
    // Active is perfectly feasible with zero cost/defect.
    active[0].x(0) = 0.0;
    active[0].u(0) = 0.0;
    active[1].x(0) = 0.0;
    active[1].u(0) = 0.0;

    // Provide a direction that makes the first candidate worse (rejected),
    // with alpha=1.0 so SOC branch is taken.
    active[0].dx(0) = 1.0;
    active[1].dx(0) = 0.0;
    active[0].du(0) = 0.0;
    active[1].du(0) = 0.0;

    // Ensure initial metrics are computed.
    for (int k = 0; k <= trajectory.N; ++k) {
        SocTriggerModel::compute(
            active[k], config.integrator, (k < trajectory.N) ? dt_traj[k] : 0.0);
    }

    g_allocation_count = 0;
    g_memory_check_active = true;
    const double alpha = line_search.search(trajectory, linear_solver, dt_traj, /*mu=*/0.1,
        /*reg=*/1e-4, config);
    g_memory_check_active = false;

    EXPECT_DOUBLE_EQ(alpha, 0.0);
    EXPECT_EQ(g_allocation_count, 0) << "Detected " << g_allocation_count
                                     << " heap allocations inside FilterLineSearch SOC path!";
}
