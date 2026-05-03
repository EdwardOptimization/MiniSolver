# MiniSolver Testing Matrix

MiniSolver tests are organized around solver contracts, not around a full
Cartesian product of every option. The goal is to keep per-commit CI fast while
still locking the semantics that matter for NMPC correctness, zero-allocation
claims, and generated-model compatibility.

## Scope

In scope:

- Core math kernels: matrix backends, factorization, finite checks, autodiff.
- Solver contracts: primal/dual residuals, barrier updates, line search, SOC,
  implicit integration, restoration behavior, zero-malloc solve paths.
- Code generation: MiniModel identifiers, terminal projection, residual costs,
  constraint packets, implicit Riccati sparsity.
- Integration quality: reference solver path, default solver path, generated
  asset regressions, benchmark smoke.

Out of scope for this matrix:

- Full NMPC performance comparisons against acados/CasADi. Those belong in the
  separate `MiniSolver-Bench` repository.
- Cross-version snapshot schema migration. The current solver snapshot I/O is a
  replay/debug tool, not a stable public interchange format. Current-format
  round-trip behavior is tested, while old formats are rejected explicitly.

## Test Tiers

| Tier | Purpose | Examples | Required Frequency |
| --- | --- | --- | --- |
| Unit | Small deterministic contracts | matrix kernels, barrier update, termination, initialization | Every commit |
| Component | One solver subsystem with controlled inputs | line search, Riccati, implicit integrator, SOC | Every commit |
| Integration | Full `MiniSolver::solve()` on small models | reference/default agreement, soft constraints, memory tests | Every commit |
| Codegen | Python MiniModel generates compilable and correct C++ | residual costs, constraint packets, terminal projection | Every commit |
| Asset regression | Generated real-ish models against stored references | bicycle, double-integrator 3D | Every commit |
| Stress | Long iteration or capacity tests | filter ring-buffer wrap, zero-malloc variants | Every commit if fast; otherwise nightly |
| Benchmark | Runtime and accuracy comparisons | nmpc-bench, matrix microbenchmarks | Manual/nightly, not unit CI |

## Required Matrix

### Backends

| Backend | Coverage |
| --- | --- |
| Eigen | Full `ctest` in `.build` |
| MiniMatrix | Full `ctest` in `.build_custom` |
| CUDA/GPU | No required CI until the GPU backend is made functional |

### Solver Routes

| Route | Must Cover | Current Tests |
| --- | --- | --- |
| Reference path | Conservative IPM semantics without advanced heuristics | `test_solver_quality` |
| Default path | Recommended production defaults | `test_solver_quality`, `test_solver` |
| Filter line search | Acceptance, SOC, fixed-capacity history | `test_line_search` |
| Merit line search | Armijo/failure semantics | `test_line_search`, `test_bugfixes` |
| No line search | RTI-style full step and rollout semantics | `test_line_search` |
| Postsolve verdict | Fresh residuals override stale loop status | `test_barrier_residual_contract` |

### Model Features

| Feature | Must Cover | Current Tests |
| --- | --- | --- |
| Hand-written model | Core C++ API remains usable without codegen | most C++ tests |
| MiniModel generated model | Header compiles and exposes expected symbols | `tests/minimodel/*`, asset regressions |
| Terminal stage projection | Terminal cost/constraints do not depend on `u_N` semantics | `tests/minimodel/test_terminal.py` |
| True/QP/SOC constraint packets | Globalization uses true residuals; QP/SOC may use overrides | `tests/minimodel/test_constraints.py`, `test_line_search` |
| L1/L2 soft constraints | Initialization, convergence, dual/slack safety | `test_soft_constraints`, `test_bugfixes` |
| Residual costs | True Gauss-Newton Hessian and parameter weights | `tests/minimodel/test_residual_costs.py` |
| Implicit integrators | BE, midpoint, Gauss-Legendre accuracy and A/B Jacobians | `test_integrator`, `test_implicit_sparse_riccati` |

### Memory And Diagnostics

| Claim | Required Coverage |
| --- | --- |
| Zero dynamic allocation during production `solve()` | `test_memory` with profiling/logging disabled and multiple line-search modes |
| Profiling/logging | Explicitly not part of the zero-malloc guarantee unless converted to fixed buffers |
| Solver snapshot/replay | Round-trip config, trajectory, soft slacks, backend policy, model fingerprint, and atomic failure behavior |

## CI Commands

Per-commit local validation should run both configured build trees:

```bash
cmake --build .build -j$(nproc)
ctest --test-dir .build --output-on-failure

cmake --build .build_custom -j$(nproc)
ctest --test-dir .build_custom --output-on-failure
```

Focused development may run a narrower target first, but full Eigen and
MiniMatrix `ctest` should pass before commit or push.

## Current Follow-Up Items

These are testing obligations, not a request to delete old gap records. When a
gap is closed, keep the discovery in the dated gap/review document and add the
regression test that now protects it.

P0:

- Keep implicit integrator A/B Jacobian checks for all implicit schemes.
- Keep reference/default agreement on at least unconstrained QP, L1 soft, and
  implicit-integrator QP.

P1:

- Add stricter Armijo directional-derivative regression if cancellation is
  observed in a real solver case.
- Expand reference/default agreement as new public model features are added.

P2:

- Keep fixed-capacity line-search history tests lightweight.
- Move long horizon or slow stress coverage to nightly if it becomes expensive.
- Keep snapshot/replay tests current whenever `SolverConfig` or solver runtime
  state changes.
