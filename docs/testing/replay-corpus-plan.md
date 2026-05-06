# Replay Corpus Plan

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
| L1 soft constraint | Soft-constraint interior and diagnostics under the default solve path | `test_replay_corpus` |
| Badly scaled equivalent constraints | Scaled vs unscaled feasibility reporting | `test_replay_corpus` |
| Failure snapshot workflow | Save pre-solve state only when solve fails and reload it for replay | `test_replay_corpus` |

## Expansion Order

1. Add a nonlinear obstacle/SOC scenario after the current SOC semantics are
   stable.
2. Add warm-start active-set change after replay metrics are stable across two
   consecutive solves.
3. Add an implicit-integrator scenario that uses generated models, not a custom
   hand-written shortcut.
4. Add benchmark-backed scenarios only after they are proven useful in
   MiniSolver-Bench.

## Acceptance Rules

- A corpus case must be deterministic and run in normal `ctest`.
- New metrics must be added to the test and this plan in the same commit.
- If a corpus case becomes flaky, reduce the assertion to a solver invariant or
  move the scenario to benchmark/nightly coverage.
- If a scenario suggests a new solver feature, route it through the review
  triage checklist and overdesign ledger before coding.
