# Implicit Integrator Contract

Status: draft

Owner modules:

- `MOD-INTEGRATOR`
- `MOD-ALG-EVAL`

Related modules:

- `MOD-MODEL-CODEGEN`
- `MOD-MATRIX`

Coverage matrix:

- [`../testing/contract-coverage-matrix.md`](../testing/contract-coverage-matrix.md)

## Scope

Implicit Euler, implicit midpoint, Gauss-Legendre two-stage integration, Newton
solve behavior, Jacobian inversion, and invalid dynamics marking.

## Contract IDs

| ID | Requirement | Evidence status |
| --- | --- | --- |
| `INT-020` | Implicit integrator only accepts implicit integrator types. | `covered` |
| `INT-021` | Newton solve uses analytical continuous Jacobians when available and numerical Jacobians otherwise. | `covered` |
| `INT-022` | Failed Newton solve marks dynamics and Jacobian packets invalid. | `covered` |
| `INT-023` | Failed Jacobian inversion marks Jacobian packets invalid. | `covered` |
| `INT-024` | Implicit integrator writes `f_resid`, `A`, and `B` for Riccati/model evaluation. | `covered` |
| `INT-025` | Newton regularization is a fallback for singular Jacobians, not unconditional damping. | `covered` |

## Failure Semantics

| Failure | Required behavior |
| --- | --- |
| Newton failure | Mark packet NaN; solver numeric boundary handles solve failure. |
| Unsupported type | Throw invalid argument for internal misuse. |
| Singular Jacobian with no successful damping | Return failure/mark invalid. |

## Tests And Evidence

| Contract ID | Evidence | Status |
| --- | --- | --- |
| `INT-020` | `tests/test_integrator.cpp::ImplicitIntegratorTest.RejectsUnsupportedDirectIntegratorType`, `tests/test_integrator.cpp::ImplicitIntegratorTest.DispatchRejectsImplicitWithoutContinuousDynamics` | `covered` |
| `INT-021` | `tests/test_integrator.cpp::ImplicitIntegratorTest.JacobianAccuracy`, `tests/test_integrator.cpp::ImplicitIntegratorTest.JacobiansMatchFiniteDifferenceForAllImplicitSchemes` | `covered` |
| `INT-022` | `tests/test_integrator.cpp::ImplicitIntegratorTest.FailedNewtonSolveInvalidatesDynamics`, `tests/test_integrator.cpp::ImplicitIntegratorTest.FailedNewtonSolveInvalidatesStandaloneIntegrate` | `covered` |
| `INT-023` | `tests/test_integrator.cpp::ImplicitIntegratorTest.SingularJacobianMarksDynamicsInvalid` | `covered` |
| `INT-024` | `tests/test_integrator.cpp::ImplicitIntegratorTest.JacobiansMatchFiniteDifferenceForAllImplicitSchemes`, `tests/test_implicit_sparse_riccati.cpp`, `tests/test_replay_corpus.cpp::ReplayCorpusTest.GeneratedImplicitIntegratorConvergesAndReplaysPreSolveSnapshot` | `covered` |
| `INT-025` | `tests/test_integrator.cpp::ImplicitIntegratorTest.NewtonDoesNotDampSolvableIllConditionedJacobian` | `covered` |

## Open Gaps

- No open P1 evidence gaps. Stiff-model replay remains useful before
  production implicit-integrator claims.
