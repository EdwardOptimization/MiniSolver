#include <gtest/gtest.h>

#include "fusedeulerimplicitregressionmodel.h"
#include "fusedgaussimplicitregressionmodel.h"
#include "fusedmidpointimplicitregressionmodel.h"
#include "genericeulerimplicitregressionmodel.h"
#include "genericgaussimplicitregressionmodel.h"
#include "genericmidpointimplicitregressionmodel.h"
#include "minisolver/solver/solver.h"

using namespace minisolver;

namespace {

constexpr int kHorizon = 10;
constexpr int kMaxHorizon = 16;

bool is_success(SolverStatus status)
{
    return status == SolverStatus::OPTIMAL || status == SolverStatus::FEASIBLE;
}

template <typename SolverT> double total_cost(const SolverT& solver)
{
    double cost = 0.0;
    for (int k = 0; k <= kHorizon; ++k) {
        cost += solver.get_stage_cost(k);
    }
    return cost;
}

template <typename BaseModel> struct CountingFusedModel : public BaseModel {
    inline static int fused_calls = 0;

    template <typename T>
    static void compute_fused_riccati_step(
        const MSMat<T, BaseModel::NX, BaseModel::NX>& Vxx,
        const MSVec<T, BaseModel::NX>& Vx,
        KnotPoint<T, BaseModel::NX, BaseModel::NU, BaseModel::NC, BaseModel::NP>& kp)
    {
        ++fused_calls;
        BaseModel::template compute_fused_riccati_step<T>(Vxx, Vx, kp);
    }
};

SolverConfig make_config(IntegratorType integrator)
{
    SolverConfig config;
    config.print_level = PrintLevel::NONE;
    config.integrator = integrator;
    config.default_dt = 0.08;
    config.max_iters = 80;
    config.tol_con = 1e-7;
    config.tol_dual = 1e-7;
    config.tol_grad = 1e-7;
    config.mu_final = 1e-8;
    config.tol_mu = 1e-8;
    config.tol_cost = 1e-10;
    config.barrier_strategy = BarrierStrategy::MONOTONE;
    config.line_search_type = LineSearchType::FILTER;
    config.enable_profiling = false;
    return config;
}

template <typename Model> MiniSolver<Model, kMaxHorizon> make_initialized_solver(IntegratorType type)
{
    MiniSolver<Model, kMaxHorizon> solver(kHorizon, Backend::CPU_SERIAL, make_config(type));
    solver.set_dt(0.08);
    solver.set_initial_state("x0", 1.0);
    solver.set_initial_state("x1", -0.45);
    solver.set_initial_state("x2", 0.30);
    solver.set_initial_state("x3", -0.20);
    solver.set_initial_state("x4", 0.12);
    solver.rollout_dynamics();
    return solver;
}

template <typename FusedBaseModel, typename GenericModel>
void expect_implicit_sparse_riccati_matches_generic(IntegratorType integrator)
{
    using FusedModel = CountingFusedModel<FusedBaseModel>;

    FusedModel::fused_calls = 0;
    auto fused_solver = make_initialized_solver<FusedModel>(integrator);
    auto generic_solver = make_initialized_solver<GenericModel>(integrator);

    SolverStatus fused_status = fused_solver.solve();
    SolverStatus generic_status = generic_solver.solve();

    ASSERT_TRUE(is_success(fused_status));
    ASSERT_TRUE(is_success(generic_status));
    EXPECT_GT(FusedModel::fused_calls, 0)
        << "matching generated_integrator must dispatch to the implicit sparse Riccati kernel";

    EXPECT_NEAR(total_cost(fused_solver), total_cost(generic_solver), 1e-7);
    for (int k = 0; k <= kHorizon; ++k) {
        for (int i = 0; i < FusedModel::NX; ++i) {
            EXPECT_NEAR(fused_solver.get_state(k, i), generic_solver.get_state(k, i), 1e-7)
                << "state mismatch at stage " << k << ", index " << i;
        }
    }
    for (int k = 0; k < kHorizon; ++k) {
        for (int i = 0; i < FusedModel::NU; ++i) {
            EXPECT_NEAR(fused_solver.get_control(k, i), generic_solver.get_control(k, i), 1e-7)
                << "control mismatch at stage " << k << ", index " << i;
        }
    }
}

} // namespace

TEST(ImplicitSparseRiccatiTest, BackwardEulerSolveMatchesGenericRiccati)
{
    expect_implicit_sparse_riccati_matches_generic<FusedEulerImplicitRegressionModel,
        GenericEulerImplicitRegressionModel>(IntegratorType::EULER_IMPLICIT);
}

TEST(ImplicitSparseRiccatiTest, ImplicitMidpointSolveMatchesGenericRiccati)
{
    expect_implicit_sparse_riccati_matches_generic<FusedMidpointImplicitRegressionModel,
        GenericMidpointImplicitRegressionModel>(IntegratorType::RK2_IMPLICIT);
}

TEST(ImplicitSparseRiccatiTest, GaussLegendreSolveMatchesGenericRiccati)
{
    expect_implicit_sparse_riccati_matches_generic<FusedGaussImplicitRegressionModel,
        GenericGaussImplicitRegressionModel>(IntegratorType::RK4_IMPLICIT);
}
