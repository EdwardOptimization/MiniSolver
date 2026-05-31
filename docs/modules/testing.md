# Testing And Evidence

Module ID: `MOD-TESTING`

Status: draft

Files:

- `tests/**`
- `docs/testing/**`
- generated test model scripts under `tests/models/**`
- reference assets under `tests/reference/**`

Owner layer:

- Unit, component, regression, memory, replay, and MiniModel evidence.

## Purpose

Own the in-repository evidence structure used to protect solver contracts:
C++ tests, Python MiniModel tests, generated model assets, reference data,
memory/allocation tests, replay corpus tests, and coverage/gap documentation.

## Inputs

| Input | Source | Assumptions |
| --- | --- | --- |
| Source code under test | Repository | Current checkout. |
| Generated assets | MiniModel scripts and committed references | Regenerated only when intentionally updated. |
| Contract IDs | `docs/contracts/**` | Added in later phases. |
| Benchmark/replay evidence | `nmpc-bench` or in-repo replay tests | Linked, not dumped wholesale. |

## Outputs

| Output | Consumer | Meaning |
| --- | --- | --- |
| Test pass/fail | Developer/CI | Regression evidence. |
| Coverage matrix rows | Maintainer/reviewer | Contract-to-evidence mapping. |
| Reference assets | Tests | Stable generated-code behavior. |

## Owned State

| State | Lifetime | Notes |
| --- | --- | --- |
| Reference generated data | Repository | Must be intentionally updated. |
| Coverage/gap docs | Repository | Updated with contract changes. |

## Public API Surface

No runtime API. Test utilities such as `tests/solver_internal_access.h` are
test-only friend boundaries and must not grow public solver API.

## Internal Contracts

- Behavior fixes should have red tests or an explicit evidence path first.
- Benchmark data belongs in benchmark repositories unless it explains a
  MiniSolver design decision.
- Coverage matrix owns missing/partial/deferred status for contract IDs.

## Hot-Path And Allocation Policy

- Hot path: no
- Solve-time allocation allowed: test-dependent
- Notes: `test_memory` is the evidence path for zero-allocation claims.

## Failure Semantics

| Failure | Status/reason path | Notes |
| --- | --- | --- |
| Red test | Blocks behavior fix completion until addressed | Evidence-driven workflow. |
| Missing contract coverage | `missing`/`partial` in coverage matrix | Not necessarily a code failure. |
| Benchmark regression | Requires benchmark-specific analysis | Should not be inferred from raw CSV alone. |

## Tests And Evidence

| Evidence | Scope |
| --- | --- |
| `ctest` | Main C++ test suite. |
| `pytest tests/minimodel` | MiniModel/codegen tests. |
| `tests/test_memory.cpp` | Zero-allocation evidence. |
| `tests/test_replay_corpus.cpp` | In-repo replay evidence. |

## Known Gaps

- Contract coverage matrix is scaffolded but not filled.
