# MiniSolver 测试覆盖缺口分析

**Date:** 2026-05-02
**Baseline:** 25 CTest entries after `18da9fc` test split (Eigen + Custom backend)
**Source:** `docs/review_2026-05-02.md` 中验证通过的 25 条 issue

**2026-05-02 follow-up:** `GAP-1` through `GAP-9` and `GAP-12` now have direct
regression coverage, except `GAP-6` which remains a benchmark/real-case trigger
candidate. Covered items were confirmed as coverage gaps rather than current
behavior failures.

---

## 现有覆盖概览

| 测试文件 | 测试数 | 覆盖领域 |
|----------|--------|----------|
| `test_bugfixes.cpp` | 31 | NaN 检测、Mehrotra、SOC、barrier update、dual recovery |
| `test_config_regressions.cpp` | 4 | config/backend/build-state/horizon/query guard |
| `test_barrier_residual_contract.cpp` | 6 | barrier residual snapshot、postsolve stale residual guard、Mehrotra target-mu edge cases |
| `test_integrator.cpp` | 18 | dispatch rejection、Newton 收敛、3 种积分器精度、stiff、A/B Jacobian、terminal implicit eval、warm start |
| `test_solver_quality.cpp` | 15 | FD Jacobian、KKT 最优性、解析解、MPC 闭环、Mehrotra、reference config |
| `test_line_search.cpp` | 14 | filter/merit acceptance、SOC candidate 语义、damping、model hook、rollout、filter capacity |
| `test_mini_matrix.cpp` | 13 | Cholesky、LDLT、LU、block views、dot、symmetrize、finite checks |
| `tests/minimodel/*` | 5 CTest entries | identifiers、implicit patterns、constraint packet/SOC、residual costs、terminal projection |
| `test_memory.cpp` | 6 | 零 malloc（6 种配置组合） |
| `test_serializer.cpp` | 7 | round-trip、soft_s、config fields、格式拒绝、原子性 |
| `test_soft_constraints.cpp` | 6 | L1/L2 收敛、invalid dual warm start、tiny L1 weight initialization |
| `test_features.cpp` | 4 | 基本功能 |
| `test_advanced.cpp` | 4 | 高级场景 |
| `test_solver.cpp` | 3 | 全收敛、不可行恢复、horizon resize |
| `test_implicit_sparse_riccati.cpp` | 3 | 3 种 implicit integrator fused vs generic |

**覆盖良好的区域**：
- 隐式积分器全链路（dispatch → Newton → Jacobian → Riccati）
- SOC 重构（candidate 语义、damping、model hook）
- NaN 传播（Jacobian/dynamics/soft slack）
- Mehrotra（mu gating、mu_aff L1 soft pair）
- 零 malloc（所有 line search × integrator 组合）
- 序列化（round-trip + 错误格式拒绝）

---

## 缺口分析

### 优先级 HIGH（对应 confirmed HIGH/MEDIUM issue，无任何测试覆盖）

#### GAP-1: `evaluate_model_stage` terminal dynamics 开销

- **对应 Issue**: HIGH-1 — terminal 点仍调用 dynamics
- **现状**: Covered by `ImplicitIntegratorTest.TerminalImplicitEvaluationAtZeroDtIsFinite`
- **缺失测试**: 构造一个隐式积分器模型，在 terminal knot 调用 `evaluate_model_stage`，验证：
  1. 不产生 NaN 或错误结果
  2. 可选：测量 Newton 迭代次数（应为 0 或 1，因为 `z - x = 0`）
- **用途**: 确认是否需要 `if (!is_terminal)` gate，以及性能影响
- **结论**: Terminal implicit evaluation at `dt=0` is finite and produces the
  identity/zero discrete Jacobian. The remaining question is performance, not
  correctness.

#### GAP-2: 收敛检查用 stale dual_inf 的影响

- **对应 Issue**: MEDIUM-2 — `check_convergence` 用 pre-line-search `max_dual_inf`
- **现状**: Covered by `BarrierResidualContractTest.PostsolveRechecksDualResidualAfterLoopOptimal`
- **缺失测试**: 构造一个 case：
  1. Pre-line-search dual_inf < tol（满足收敛）
  2. Post-line-search dual_inf > tol（实际不收敛）
  3. 验证 `postsolve()` 能纠正 in-loop 的 OPTIMAL 判定
- **用途**: 证明 `postsolve()` 兜底机制有效
- **结论**: A loop-level `OPTIMAL` verdict is downgraded by fresh postsolve dual
  residuals.

#### GAP-3: `HasNanImpl<true>` early return 行为

- **对应 Issue**: MEDIUM-3 — `HasNanImpl<true>` 用 plain for 而非 StaticFor
- **现状**: Covered by `MiniMatrixTest.Kernel_HasNanAndAllFiniteBoundaryCases`
- **缺失测试**:
  1. 矩阵第一个元素是 NaN → `has_nan` 立即返回 true
  2. 矩阵最后一个元素是 NaN → `has_nan` 扫描全部后返回 true
  3. 无 NaN → `has_nan` 返回 false
- **用途**: 确认 early return 语义正确（当前实现是 plain for，early return 是有意 trade-off）
- **结论**: Current behavior detects NaN in first and last positions and
  distinguishes Inf from NaN. The plain loop remains an intentional early-return
  trade-off.

---

### 优先级 MEDIUM（边界条件，现有测试未触及的路径）

#### GAP-4: Mehrotra `mu_curr==0` 除零

- **对应 Issue**: LOW-7 — `barrier_update.h:26` 无 guard
- **现状**: Covered by `BarrierResidualContractTest.MehrotraTargetMuHandlesZeroCurrentMu`
- **缺失测试**: 直接调用 `BarrierUpdateKernel::mehrotra_target_mu(avg_gap=0.1, mu_curr=0.0, config)`，验证返回值不是 NaN/Inf
- **用途**: 确认是否需要 `std::max(mu_curr, 1e-30)` guard
- **结论**: Current behavior returns `mu_final` for zero-current-mu edge cases;
  keep this as a regression guard unless a stronger barrier-update contract is
  designed.

#### GAP-5: L1 初始化极小权重

- **对应 Issue**: LOW-25 — `initialization.h:54` clamping range 为空
- **现状**: Covered by `SoftConstraintTest.L1TinyWeightInitializationStaysFinite`
- **缺失测试**: 构造 `w = 1e-10` 的 L1 soft constraint，验证 `initialize_primal_dual_slacks_` 不产生 NaN
- **用途**: 确认极小权重下的数值行为
- **结论**: Current initialization keeps finite positive hard-slack/dual values.
  Tiny L1 weights below the existing `w > 1e-6` threshold are not treated as an
  active L1 dual-box constraint.

#### GAP-6: Armijo 方向导数估计的 cancellation

- **对应 Issue**: LOW-12 — `eps_alpha` 下界 1e-10
- **现状**: `MeritLS_ArmijoRejectsTinyImprovement` 是 demo 性质，无严格断言
- **缺失测试**:
  1. 构造 barrier merit 函数
  2. 设置 `alpha = 1e-5`（使 `eps_alpha = alpha * 1e-6 = 1e-11 < 1e-10`，触发下界）
  3. 验证 `dphi` 估计值的符号正确（负 = 下降方向）
  4. 验证 `dphi` 的量级合理（不因 cancellation 变成 0 或反号）
- **用途**: 量化 cancellation 风险，决定是否需要提高下界到 `1e-8`

#### GAP-7: `should_stop_after_line_search_` stale primal inf

- **对应 Issue**: LOW-11 — `max_prim_inf` 在 line search 前计算
- **现状**: Covered by `BarrierResidualContractTest.PostsolveRechecksPrimalResidualAfterLoopOptimal`
- **缺失测试**: 构造一个 case：
  1. Pre-line-search `max_prim_inf > tol`（不满足）
  2. Line search 将 primal inf 降到 `tol` 以下
  3. 验证 `should_stop_after_line_search_` 是否正确判断 FEASIBLE
- **用途**: 量化 stale 值对 early exit 的影响
- **结论**: A loop-level `OPTIMAL` verdict is downgraded by fresh postsolve
  primal residuals. The in-loop early-exit condition still uses a snapshot, but
  final status is protected.

#### GAP-8: Filter circular buffer overflow

- **对应 Issue**: LOW（info） — filter capacity 1024 硬编码
- **现状**: Covered by `LineSearchTest.FilterHistoryWrapsAtFixedCapacity`
- **缺失测试**: 设置 `max_iters = 2000`，运行 filter line search，验证：
  1. 不 crash
  2. 最终收敛（filter 覆盖不影响正确性）
- **用途**: 确认 circular buffer 覆盖行为正确
- **结论**: Direct line-search stress covers 1100 accepted entries and verifies
  fixed capacity remains 1024.

---

### 优先级 LOW（代码质量，不影响正确性）

#### GAP-9: `all_finite` 与 `has_nan` 一致性

- **对应 Issue**: LOW-16 — `all_finite` 不走 policy system
- **现状**: Covered by `MiniMatrixTest.Kernel_HasNanAndAllFiniteBoundaryCases`
- **缺失测试**: 对同一矩阵验证 `has_nan(m) == !all_finite(m)`（对于无 Inf 的矩阵）
- **用途**: 确认语义一致性
- **结论**: NaN at first and last element is detected; Inf is non-finite but
  intentionally not reported by `has_nan`.

#### GAP-10: Restoration penalty rho 适应性

- **对应 Issue**: LOW-9 — 硬编码 rho=1000.0
- **现状**: `InfeasibleStartRecovery` 测试存在但不验证 rho 效果
- **缺失测试**: 构造不同 scale 的约束问题（`g(x) = 1` vs `g(x) = 1e6`），验证 restoration 收敛
- **用途**: 确认 rho=1000.0 对不同问题 scale 是否合适

#### GAP-11: Serializer stats 字段一致性

- **对应 Issue**: MEDIUM-6 — `write_pod` vs raw `out.write`
- **现状**: `FullRoundTrip` 测试覆盖了整体 round-trip，但不单独验证 stats 字段
- **缺失测试**: 在 `CaptureAndSaveAndLoad` 中增加对 `state.iterations`、`state.total_cost`、`state.mu` 的精确值断言
- **用途**: 确认 raw `out.write` 路径与 `write_pod` 路径产生相同结果

#### GAP-12: `MeritFunctionBacktracking` failure path 断言

- **对应 Issue**: LOW-20 — failure path 用 `EXPECT_TRUE(true)`
- **现状**: Covered by `LineSearchTest.MeritFunctionBacktracking`
- **修复**: Failure path now asserts the solver exits with an expected
  non-success status instead of accepting any result.
- **结论**: This was a pure test-quality gap; no solver behavior change was
  needed.

---

## 测试质量观察

### Demo vs Regression

`test_bugfixes.cpp` 中的 `ImprovementDemo` 测试类包含：
- `MeritLS_ArmijoRejectsTinyImprovement`
- `MeritLS_ArmijoVsSimpleDecrease_Iterations`

这些测试展示改进效果，但不是严格的回归测试——它们不 assert 具体的 Armijo 行为差异。建议将关键断言提取为 `BugfixTest` 类的正式测试。

### 隐式积分器 Jacobian 验证

`ImplicitIntegratorTest::JacobiansMatchFiniteDifferenceForAllImplicitSchemes`
now verifies A/B against finite differences for Backward Euler, Implicit
Midpoint, and Gauss-Legendre.

### Reference Config 覆盖

Reference/default agreement now covers:
- simple unconstrained QP
- L1 soft constraint QP
- implicit-integrator QP

Residual-cost codegen is covered separately by `tests/minimodel/test_residual_costs.py`.

---

## 当前剩余建议

1. **GAP-6** (eps_alpha cancellation): 暂不写脆弱 synthetic test。等出现真实 merit/Armijo 异常或 benchmark case 后，用 benchmark/reproducer 锁定。
2. **Serializer stats**: 暂缓。Serializer 是 snapshot/replay 机制，不只是求解日志；应在单独重构中一起处理格式、语义和兼容口径。
3. **Restoration rho scale**: 保留为算法/策略问题。需要先有多尺度 infeasible benchmark，再决定是否改固定 rho。
