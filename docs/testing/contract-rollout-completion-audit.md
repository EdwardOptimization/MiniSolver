# Contract Rollout Completion Audit

Status: complete for the initial contract-driven development rollout

Date: 2026-05-31

This audit records why the initial contract rollout is considered complete. It
does not mean every future `P2` replay, benchmark, or nightly item is already
implemented; it means those rows have owners and explicit evidence paths, while
all current `P0` and `P1` contract rows are covered.

## Completion Criteria

| Criterion | Evidence | Result |
| --- | --- | --- |
| Every module in the inventory has a module document. | `docs/modules/module-inventory.md` links each `MOD-*` row to an existing file under `docs/modules/`. | Complete |
| Every major behavior domain has at least one contract file. | `docs/contracts/` contains solver, numeric, algorithm, constraint, model/codegen, runtime, debug, and integrator contracts. | Complete |
| Every contract file has stable contract IDs. | Contract IDs follow `docs/contracts/contract-id-policy.md`; ID consistency check reports 246 unique contract IDs. | Complete |
| Every contract ID appears in the coverage matrix. | ID consistency check: `contract_ids 246`, `matrix_ids 246`, `missing []`, `extra []`. | Complete |
| All `P0` rows are `covered`. | Matrix check reports `p0_partial_rows 0`. | Complete |
| All `P1` rows are `covered`. | Matrix check reports `p1_partial_rows 0` after P1 evidence closure. | Complete |
| `P1` closure uses direct evidence. | `contract-coverage-matrix.md` records focused or specialized evidence as primary; integration/replay evidence is supplemental unless the row is a compile/build/selection/static-policy contract. | Complete |
| Full Eigen `ctest` passes. | `ctest --test-dir build -j16 --output-on-failure`: 32/32 passed. | Complete |
| Full MiniMatrix `ctest` passes before push for matrix/Riccati/backend contract changes. | `cmake --build .build_ci_custom_release -j16 && ctest --test-dir .build_ci_custom_release -j16 --output-on-failure`: 32/32 passed. | Complete |
| `test_memory` backs all zero-allocation claims. | `test_memory` is part of both full Eigen and custom MiniMatrix runs; `memory-allocation-contract.md` records exclusions for logging/profiling/snapshot/codegen. | Complete |
| Replay corpus covers termination, soft constraints, scaling, callbacks/snapshots, and a real-ish generated model. | `tests/test_replay_corpus.cpp` covers warm-start/acceptable-NMPC termination, L1 soft constraints, badly scaled reporting, snapshot/failure replay, generated implicit model replay, and SOC nonlinear obstacle seam coverage. | Complete |

## Replay Coverage Map

| Required area | Evidence |
| --- | --- |
| Termination / acceptable NMPC | `ReplayCorpusTest.WarmStartTwoFrameSolveReachesAcceptableQualityInTwoIterations`, `ReplayCorpusTest.AcceptableNmpcEarlyStopCoversSoftConstrainedProblem` |
| Soft constraints | `ReplayCorpusTest.L1SoftConstraintConvergesWithFiniteInteriorMetrics`, `ReplayCorpusTest.AcceptableNmpcEarlyStopCoversSoftConstrainedProblem` |
| Scaling | `ReplayCorpusTest.BadScalingCaseReportsScaledAndUnscaledFeasibility`, `ReplayCorpusTest.FailureSnapshotWorkflowPersistsPreSolveReplayState` |
| Snapshot/replay | `ReplayCorpusTest.UnconstrainedTrackingConvergesAndReplaysPreSolveSnapshot`, `ReplayCorpusTest.FailureSnapshotWorkflowPersistsPreSolveReplayState` |
| Generated model | `ReplayCorpusTest.GeneratedImplicitIntegratorConvergesAndReplaysPreSolveSnapshot` |
| SOC nonlinear obstacle seam | `ReplayCorpusTest.SocNonlinearObstaclePathAttemptsAndAcceptsCorrection` |

## Current Validation Commands

```bash
git diff --check

python3 - <<'PY'
from pathlib import Path
import re
contract_dir=Path('docs/contracts')
matrix=Path('docs/testing/contract-coverage-matrix.md').read_text()
contract_ids=[]
for p in contract_dir.glob('*.md'):
    if p.name == '_template.md':
        continue
    contract_ids += re.findall(r'`([A-Z]+(?:/[A-Z]+)?-\d{3})`', p.read_text())
matrix_ids=re.findall(r'`([A-Z]+(?:/[A-Z]+)?-\d{3})`', matrix)
print('contract_ids', len(set(contract_ids)))
print('matrix_ids', len(set(matrix_ids)))
print('missing', sorted(set(contract_ids)-set(matrix_ids)))
print('extra', sorted(set(matrix_ids)-set(contract_ids)))
print('p0_partial_rows', sum(
    1 for line in matrix.splitlines()
    if '| [`../contracts/' in line and '`P0`' in line and '`partial`' in line))
print('p1_partial_rows', sum(
    1 for line in matrix.splitlines()
    if '| [`../contracts/' in line and '`P1`' in line and '`partial`' in line))
PY

ctest --test-dir build -j16 --output-on-failure

cmake --build .build_ci_custom_release -j16
ctest --test-dir .build_ci_custom_release -j16 --output-on-failure
```

## Remaining Work Policy

Remaining `partial` or `deferred` rows are `P2`. They become blocking only when
a future behavior change touches that contract, or when a release scope
explicitly promotes the row to `P1` or `P0`.
