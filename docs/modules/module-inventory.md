# Module Inventory

Status: draft

This file lists the tracked MiniSolver implementation modules. Each row points
to a module document that defines purpose, inputs, outputs, owner layer, owned
state, hot-path status, allocation policy, failure semantics, tests, and known
gaps.

| Module ID | Files | Module doc | Status |
| --- | --- | --- | --- |
| `MOD-CORE-CONFIG` | `include/minisolver/core/solver_options.h`, `include/minisolver/core/config_fields.h`, `include/minisolver/core/config_validation.h` | [`core-config.md`](core-config.md) | `draft` |
| `MOD-CORE-TYPES` | `include/minisolver/core/types.h`, `include/minisolver/core/solver_context.h`, `include/minisolver/core/solver_plan.h`, `include/minisolver/core/gpu_types.h` | [`core-types.md`](core-types.md) | `draft` |
| `MOD-CORE-TRAJ` | `include/minisolver/core/trajectory.h` | [`core-trajectory.md`](core-trajectory.md) | `draft` |
| `MOD-CORE-SEMANTICS` | `include/minisolver/core/constraint_semantics.h`, `include/minisolver/core/model_traits.h` | [`core-semantics.md`](core-semantics.md) | `draft` |
| `MOD-SOLVER-ROUTE` | `include/minisolver/solver/solver.h` | [`solver-route.md`](solver-route.md) | `draft` |
| `MOD-SOLVER-RICCATI` | `include/minisolver/solver/riccati.h`, `include/minisolver/solver/kkt_assembler.h`, `include/minisolver/algorithms/riccati_solver.h`, `include/minisolver/algorithms/linear_solver.h`, `include/minisolver/algorithms/linear_solve_result.h` | [`solver-riccati.md`](solver-riccati.md) | `draft` |
| `MOD-SOLVER-LSUTIL` | `include/minisolver/solver/line_search_utils.h` | [`solver-line-search-utils.md`](solver-line-search-utils.md) | `draft` |
| `MOD-ALG-INIT` | `include/minisolver/algorithms/initialization.h` | [`alg-initialization.md`](alg-initialization.md) | `draft` |
| `MOD-ALG-LS` | `include/minisolver/algorithms/line_search.h` | [`alg-line-search.md`](alg-line-search.md) | `draft` |
| `MOD-ALG-EVAL` | `include/minisolver/algorithms/model_evaluation.h` | [`alg-model-evaluation.md`](alg-model-evaluation.md) | `draft` |
| `MOD-ALG-TERM` | `include/minisolver/algorithms/termination.h`, `include/minisolver/algorithms/residual_stagnation_monitor.h` | [`alg-termination.md`](alg-termination.md) | `draft` |
| `MOD-ALG-BARRIER` | `include/minisolver/algorithms/barrier_update.h` | [`alg-barrier.md`](alg-barrier.md) | `draft` |
| `MOD-INTEGRATOR` | `include/minisolver/integrator/*.h` | [`integrator.md`](integrator.md) | `draft` |
| `MOD-MATRIX` | `include/minisolver/matrix/*.h` | [`matrix.md`](matrix.md) | `draft` |
| `MOD-MODEL-CODEGEN` | `python/minisolver/MiniModel.py`, `python/minisolver/__init__.py`, `python/minisolver/templates/model.h.in` | [`model-codegen.md`](model-codegen.md) | `draft` |
| `MOD-DEBUG-SNAPSHOT` | `include/minisolver/debug/solver_snapshot.h`, replay tools if present | [`debug-snapshot.md`](debug-snapshot.md) | `draft` |
| `MOD-RUNTIME` | `include/minisolver/core/logger.h`, `include/minisolver/backend/backend_interface.h`, `CMakeLists.txt`, build options | [`runtime.md`](runtime.md) | `draft` |
| `MOD-TESTING` | `tests/**`, `docs/testing/**` | [`testing.md`](testing.md) | `draft` |
