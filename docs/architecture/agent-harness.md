# Agent Harness

MiniSolver is developed with multiple agents, but the project should not rely on
the human maintainer repeatedly correcting the same classes of mistakes. This
document records the intervention patterns found in the Codex session history
and the harness rules agents should apply before acting.

Source reviewed:

- Codex rollout:
  `/home/quyaonan/.codex/sessions/2026/04/09/rollout-2026-04-09T22-22-25-019d729f-71e2-72f2-b128-a35b2742dddd.jsonl`
- The main MiniSolver session contained 696 user messages.
- Approximate intervention pattern counts from user messages:
  - process / sequencing control: 227
  - default semantics and solver contract guidance: 192
  - evidence / benchmark / test demands: 122
  - design-boundary corrections: 85
  - explicit negative corrections: 55

The counts are not a scientific metric. They are useful because they show that
the main failure mode is not isolated implementation skill; it is missing
project-specific pre-flight checks.

## Repeated Intervention Patterns

| Pattern | Typical maintainer correction | Preventive rule |
| --- | --- | --- |
| Overdesign | Avoid adapters, public plugin layers, bulk iterate import/export, or oversized APIs. | Start with existing setters, `SolverConfig`, and internal helpers. Add public API only with a concrete current use case. |
| Boundary drift | Keep benchmark comparison outside MiniSolver; keep model semantics out of solver core. | MiniSolver core is generic NMPC/IPM/Riccati. MiniModel/codegen owns model-specific semantics. nmpc-bench owns cross-solver comparison. |
| Evidence gaps | Do not patch directly from review prose; add red tests or benchmarks first. | Use `claim -> evidence -> fix -> same validation -> commit`. |
| Default semantics | Defaults should be correctness-first; realtime shortcuts must be explicit. | New behavior goes behind `SolverConfig` and is resolved at the build boundary. |
| Zero-malloc ambiguity | Stack POD locals are not heap allocation; solve-time dynamic allocation is the issue. | Run `test_memory` for hot-path allocation claims. |
| Overdefense | Branches for impossible internal states hide invariant problems. | Validate at boundaries, fix state transitions, and avoid hot-path checks for invalid states that should be impossible. |
| Commit churn | History became noisy when mechanical extraction and behavior fixes mixed. | Split commits by behavior and preserve test evidence. |

## Required Agent Pre-Flight

Before changing MiniSolver, an agent should answer these questions:

1. Which layer owns the problem: solver core, MiniModel/codegen, examples/tools,
   tests, docs, or nmpc-bench?
2. Is this a confirmed bug, a design decision, a performance question, a docs
   mismatch, or an unconfirmed review claim?
3. What is the narrowest red test, benchmark, compile check, or allocation test?
4. Does the change preserve `SolverConfig` as the user-facing strategy surface?
5. Does the change preserve solve-time zero-malloc?
6. Does the commit mix unrelated behavior, formatting, docs, or benchmark assets?

If any answer is unclear, stop and write a short design note or test plan before
editing solver behavior.

## Harness Rules

### Public API

- Users should configure solver behavior through `SolverConfig` and existing
  setter/getter methods.
- Do not add a public OOP plugin framework for strategy selection.
- Internal kernels or build-state objects are allowed when they reduce coupling,
  but they should be selected from config and frozen before hot solve loops.

### Solver Core

- Core should know residuals, derivatives, slacks, duals, Riccati/KKT solves,
  line search, restoration, and termination contracts.
- Core should not know circle, ellipse, obstacle, race track, dataset, or other
  application semantics.
- If a feature needs model semantics, represent it through generated numerical
  packets rather than solver-core type checks.

### Evidence

- Review findings with a standard mathematical route should get red tests first.
- Design-sensitive findings should get a short design document first.
- Performance claims need before/after numbers and correctness metrics.
- Zero-malloc claims need allocation instrumentation.
- Benchmark fairness requires same model semantics and same runtime class.

### Defaults

- Defaults should favor correctness, reproducibility, and debug clarity.
- RTI, acceptable NMPC termination, warm-start reuse, and aggressive
  performance modes should be explicit config choices until benchmark evidence
  supports making them presets.

### Commits

- Commit tests with the behavior they protect.
- Keep docs-only changes separate unless they describe the same behavior.
- Re-run targeted tests after pre-commit formatting modifies files.
- Run full CTest before push for solver-core changes.

## Local Skill

The local agent skill implementing this harness is:

`/home/quyaonan/.agents/skills/minisolver-engineering-harness/SKILL.md`

It should be loaded automatically whenever an agent works in MiniSolver or
nmpc-bench on solver design, debugging, benchmarks, codegen, tests, CI, or
repository hygiene.
