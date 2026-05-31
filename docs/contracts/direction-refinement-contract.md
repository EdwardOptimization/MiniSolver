# Direction Refinement Contract

Status: draft

Owner modules:

- `MOD-SOLVER-RICCATI`
- `MOD-CORE-CONFIG`

Related modules:

- `MOD-CORE-TRAJ`
- `MOD-SOLVER-ROUTE`

Coverage matrix:

- [`../testing/contract-coverage-matrix.md`](../testing/contract-coverage-matrix.md)

## Scope

Optional post-Riccati direction correction for linearized dynamics defects.

## Contract IDs

| ID | Requirement | Evidence status |
| --- | --- | --- |
| `RIC/REFINE-001` | Direction refinement is disabled by default unless `DirectionRefinementMode` requests it. | `partial` |
| `RIC/REFINE-002` | Dynamics-defect rollout refinement may correct `dx/du` using existing feedback gains and original linearized system. | `partial` |
| `RIC/REFINE-003` | Direction refinement must not rebuild slack/dual directions or reinterpret KKT semantics. | `partial` |
| `RIC/REFINE-004` | Refinement uses a full-copy backup when it needs original system matrices. | `partial` |

## Inputs And Outputs

| Kind | Values |
| --- | --- |
| Inputs | Direction trajectory, original system backup, config |
| Outputs | Refined `dx/du`, unchanged public status unless later phases fail |

## Failure Semantics

| Failure | Required behavior |
| --- | --- |
| Unsupported refinement mode | Config validation rejects it. |
| Refinement disabled | Return success/no-op. |

## Tests And Evidence

| Contract ID | Evidence | Status |
| --- | --- | --- |
| `RIC/REFINE-001` | `tests/test_config_regressions.cpp` | `partial` |
| `RIC/REFINE-002` | `tests/test_riccati.cpp` | `partial` |
| `RIC/REFINE-003` | `tests/test_riccati.cpp` | `partial` |
| `RIC/REFINE-004` | `tests/test_riccati.cpp`, trajectory tests | `partial` |

## Open Gaps

- Need explicit benchmark evidence before enabling refinement by default.
