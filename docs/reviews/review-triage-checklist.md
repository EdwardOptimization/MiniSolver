# Review Triage Checklist

Last updated: 2026-05-06

Use this checklist before implementing findings from LLM reviews, static
analysis, deep reviews, or external solver comparisons.

The goal is to avoid two failure modes:

- fixing prose without proving the issue exists;
- turning modeling choices or product-boundary concerns into solver-core
  restrictions.

## Required Triage Table

Create a short table in the review plan, commit notes, or follow-up document:

| Claim | Category | Evidence | API/config impact | Modeling freedom impact | Decision | Next artifact |
| --- | --- | --- | --- | --- | --- | --- |
| Short review finding | `hard-invariant` / `algorithm-convention` / `modeling-choice` / `product-boundary` | Red test, benchmark, code evidence, external solver reference, or none | none / private / public config / public API | none / low / restricts valid models | fix / design / docs-only / defer / reject | test, benchmark, doc, ledger, or issue |

## Categories

- `hard-invariant`: valid MiniSolver states must satisfy it. Examples include
  finite solver data, enum ranges, positive barrier interior variables, and
  line-search factors in `(0, 1)`.
- `algorithm-convention`: a mature solver practice that may be valuable but
  still needs MiniSolver-specific evidence. Examples include filter switching
  variants, Mehrotra step choices, scaling kernels, and globalization profiles.
- `modeling-choice`: a user or MiniModel formulation decision. Examples include
  time-step sign, physical units, static stages, pseudo-time, and whether a
  handwritten model defines an explicit replay fingerprint.
- `product-boundary`: release, replay, observability, embedded, docs, or user
  expectation concerns. These usually start as docs, diagnostics, report fields,
  or deferred roadmap items.

## Decision Rules

- Fix `hard-invariant` issues with red tests first.
- Do not implement `algorithm-convention` changes without research, a design
  note, and a failing case or benchmark target.
- Do not turn `modeling-choice` items into core validation without explicit
  maintainer approval.
- Prefer docs, diagnostics, or deferred roadmap entries for
  `product-boundary` items before adding public knobs.
- Update `docs/reviews/overdesign-ledger.md` for rejected, deferred, docs-only,
  or modified suggestions.

## Patch Review Questions

Before coding, answer:

1. Which layer owns the concept: MiniSolver core, MiniModel/codegen,
   MiniSolver-Bench, docs/release, or local developer harness?
2. Does the change add public config or API surface?
3. Does the change add hot-path branches or dynamic allocation risk?
4. Does the change restrict a model that could be valid?
5. Is the evidence local and repeatable?
6. Does the smallest fix require code, or is a document/report/diagnostic enough?

## Evidence Expectations

- Solver behavior: red regression test, then targeted test and full `ctest`.
- Numerical algorithm: red case or benchmark, neighboring tests, and
  before/after metrics.
- Performance: microbenchmark or scenario benchmark before/after.
- Zero-malloc: allocation test plus relevant configuration coverage.
- Docs-only: `git diff --check` and stale-claim inspection.

## Anti-Patterns

- Adding a public knob because a review suggested a possible future need.
- Adding defensive hot-path branches for states that valid configuration should
  already exclude.
- Treating old review ledgers as current truth without re-anchoring to code and
  tests.
- Fixing MiniSolver core for a MiniModel or benchmark-dataset responsibility.
- Calling a feature missing when only a helper profile or ergonomic setter is
  missing.
