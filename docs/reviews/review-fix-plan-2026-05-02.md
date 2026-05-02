# Review Issue Ledger: 2026-05-02 Static Review Follow-Up

This document tracks stale-or-current static review findings with their current
status, evidence, and resolution. It intentionally preserves both the original
problem discovery and the later fix record so the solver's evolution remains
auditable.

Each item must follow:

1. Confirm the claim on the current checkout.
2. Add a focused red test, reproducer, benchmark, or static check.
3. Capture the failing baseline.
4. Apply the smallest behavior change.
5. Re-run the same evidence path and nearby regressions.

## Status Legend

| Status | Meaning |
| --- | --- |
| `confirmed` | The current code still exhibits the issue or design mismatch. |
| `partly-confirmed` | The core concern exists, but later guards/tests reduce impact. |
| `deferred-design` | Real design debt, but needs a separate design pass before code. |
| `intentional` | Current behavior is deliberate and covered, but may need documentation. |
| `fixed` | Red test and fix have landed. |

## Resolution Summary

Completed in the 2026-05-02 hardening batch:

- `4af5b04 docs: track review fix plan`
- `02ce443 docs: fix quick start config example`
- `eb31888 fix: harden solver recovery invariants`

Validation recorded before landing:

- `.build`: full `ctest` passed, 25/25.
- `.build_custom`: full `ctest` passed, 25/25.
- `clang-format` pre-commit hook passed on modified C++ files.

## P0: Fix First

| Finding | Status | Evidence Path | Resolution / Next Action |
| --- | --- | --- | --- |
| L1 restoration dual-box rebuild can push `lam >= w`, making `w - lam <= 0`. | fixed | `BugfixTest.L1RestorationRebuildKeepsDualInsideBoxWhenSlackIsTiny` failed before the fix with negative `w - lam` and passes after the clamp. | Clamp by first ensuring `s` is large enough for a non-empty dual box, then rebuild `lam` and `soft_s` from a valid box. |
| README Quick Start uses private `solver.config`. | fixed | Static check for `solver.config` in README failed before the patch and now passes. | README now uses `get_config()` / `set_config()` and the stale Python snippet variables are fixed. |
| Post-line-search early stop uses pre-line-search residual snapshot. | fixed | `BugfixTest.PostLineSearchStopRejectsStaleFeasibilitySnapshot` failed before the fix and now passes. | The helper now recomputes primal violation from the accepted trajectory before allowing the shortcut. |

## P1: Hardening And Semantics

| Finding | Status | Evidence Path | Resolution / Next Action |
| --- | --- | --- | --- |
| Restoration success means "a step was applied", not "violation improved". | fixed | `BugfixTest.FeasibilityRestorationRequiresViolationImprovement` failed before the fix and now passes. | Restoration now returns success only when violation is finite and improved or below the feasibility bound. |
| Finite validation is incomplete for guesses and model outputs. | fixed | `BugfixTest.PrimalDualGuessRejectsInfState` and `BugfixTest.HasNansRejectsInfDynamicsJacobian` failed before the fix and now pass. | Warm-start guesses and model outputs now use bit-level finite checks where values enter Riccati/barrier logic. |
| GPU backend silently falls back to CPU. | confirmed/intentional | Existing tests expect fallback; benchmark semantics are risky. | Document actual backend or introduce an explicit unsupported/degraded status before GPU benchmarks. |
| Global `-ffast-math` and `-march=native` are always enabled. | confirmed | CMake inspection plus both-backend CI. | Make both flags opt-in and keep benchmark builds explicit. |

## P2: Performance And Architecture Debt

| Finding | Status | Evidence Path | Resolution / Next Action |
| --- | --- | --- | --- |
| SOC uses `TrajArray soc_data = active` in the solve path. | confirmed | Zero heap tests pass; stack/cache pressure remains unmeasured. | Move SOC scratch into preallocated solver/line-search workspace after a stack/copy benchmark or size audit. |
| `solve()` always resets `mu/reg`, reducing MPC warm-start reuse. | confirmed | MPC-style warm-start benchmark comparing reset vs reuse. | Add an explicit config once a benchmark shows benefit and safe failure reset policy. |
| Defect rollout refinement only updates `dx/du`, not constraint directions. | confirmed/intentional | Existing docs say it is not full KKT refinement. | Either rename/deprecate the option or recompute dependent directions after a targeted constrained test. |
| Small-`NU` Riccati fallback freezes control dimensions without external degraded flag. | intentional | Existing tests cover the fallback. | Add diagnostics if benchmark/debug use needs to distinguish exact vs degraded directions. |
| Terminal stage still evaluates dynamics with `dt=0`. | partly-confirmed | Existing test proves finite implicit behavior; cost remains performance-only. | Consider a performance-motivated gate after profiling terminal overhead. |

## Open Follow-Up Order

1. GPU backend degraded/unsupported semantics before any GPU benchmark claims.
2. CMake `-ffast-math` and `-march=native` opt-in behavior.
3. SOC scratch trajectory stack/copy measurement before changing storage.
4. Warm-start `mu/reg` reuse benchmark before adding config.
5. Defect rollout refinement naming or constrained-direction consistency test.
