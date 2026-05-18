# Replay Corpus Plan

Last updated: 2026-05-06

MiniSolver needs a small internal replay/correctness corpus before adding more
solver features. The corpus is not a cross-solver benchmark; that belongs in
MiniSolver-Bench. This corpus exists to keep MiniSolver behavior reproducible
across refactors and to make failure snapshots first-class debugging artifacts.

## Goals

- Exercise representative MiniSolver paths with lightweight in-repository
  models.
- Record the same diagnostic fields for each scenario.
- Verify that pre-solve snapshots can reproduce a failing solve setup.
- Keep the corpus cheap enough to run in normal `ctest`.

## Non-Goals

- No comparison against acados, CasADi, IPOPT, or external datasets.
- No performance claims beyond detecting local regressions.
- No new solver API or public strategy framework.
- No piecewise/ppoly or large benchmark assets in this repository.

## Corpus Metrics

Every scenario should expose or assert the following fields when relevant:

- `SolverStatus`
- `TerminationReason`
- iteration count
- total unscaled stage cost
- internal `primal_inf`
- `unscaled_primal_inf`
- `dual_inf`
- `complementarity_inf`
- accepted `alpha`
- regularization escalation count
- SOC attempt/accept/reject counts
- restoration attempt/success counts
- degraded Riccati freeze count
- scaling active flags

## Initial In-Repository Scenarios

| Scenario | Purpose | Current Artifact |
| --- | --- | --- |
| Unconstrained tracking | Basic reference-path correctness and finite diagnostics | `test_replay_corpus` |
| Warm-start two-frame tracking | Continuous-control replay: user-side shifted previous solution should reach acceptable quality in two iterations | `test_replay_corpus` |
| L1 soft constraint | Soft-constraint interior and diagnostics under the default solve path | `test_replay_corpus` |
| Badly scaled equivalent constraints | Scaled vs unscaled feasibility reporting | `test_replay_corpus` |
| Failure snapshot workflow | Save pre-solve state only when solve fails and reload it for replay | `test_replay_corpus` |
| Generated implicit integrator | Generated model using the implicit integrator path, not a handwritten shortcut | `test_replay_corpus` |
| SOC nonlinear obstacle | Deterministic filter/SOC path on a nonlinear circle obstacle residual | `test_replay_corpus` |

## Current Decision From Initial Corpus

The initial corpus is a correctness and replay baseline, not a license to enable
new default algorithms.

Current evidence supports:

- keeping snapshot failure capture as the standard bug-report workflow;
- using explicit application-side trajectory shifting for consecutive MPC
  frames, rather than reintroducing a solver-owned `shift_trajectory()` API;
- requiring new solver diagnostics to appear in corpus metrics when they affect
  solve decisions;
- keeping generated implicit-integrator coverage in the normal replay corpus,
  while detailed fused-vs-generic comparisons remain in
  `test_implicit_sparse_riccati`;
- using deterministic SOC seam coverage for nonlinear obstacle residuals before
  relying on end-to-end solves that may or may not trigger SOC;
- continuing to expand deterministic in-repository scenarios before adding new
  theory features.

Current evidence does not yet justify:

- enabling `RUIZ_EQUILIBRATION` or other scaling modes in the default
  `SolverConfig`;
- implementing Pareto-frontier filter history;
- implementing full KKT iterative refinement;
- adding inertia-detection KKT factorization;
- changing SOC semantics beyond already-covered regression fixes.

Reopen those algorithm decisions only when a corpus case or MiniSolver-Bench
scenario shows a concrete failure, accuracy gap, or runtime regression.

## Expansion Order

1. Add warm-start active-set or constraint-status change cases once the
   two-frame tracking baseline exposes stable diagnostics.
2. Add end-to-end nonlinear obstacle solves when they deterministically exercise
   SOC/globalization decisions; until then, keep SOC seam coverage deterministic.
3. Add benchmark-backed scenarios only after they are proven useful in
   MiniSolver-Bench.

## Acceptance Rules

- A corpus case must be deterministic and run in normal `ctest`.
- New metrics must be added to the test and this plan in the same commit.
- If a corpus case becomes flaky, reduce the assertion to a solver invariant or
  move the scenario to benchmark/nightly coverage.
- If a scenario suggests a new solver feature, route it through the review
  triage checklist and overdesign ledger before coding.
