# Model And Codegen

Module ID: `MOD-MODEL-CODEGEN`

Status: draft

Files:

- `python/minisolver/MiniModel.py`
- `python/minisolver/__init__.py`
- `python/minisolver/templates/model.h.in`

Owner layer:

- Python modeling DSL and generated C++ model packets.

## Purpose

Own user modeling syntax, symbolic validation, objective/residual/constraint
normalization, soft-constraint structural metadata and weight updater emission,
generated dynamics/cost/constraint packets, fingerprints, and optional fused
Riccati code generation.

## Inputs

| Input | Source | Assumptions |
| --- | --- | --- |
| State/control/parameter declarations | User Python model | Names must be valid C++ identifiers and unique. |
| Dynamics equations | User model | Must match supported `Dot`/`Next` patterns. |
| Objectives/residuals/constraints | User model | Symbolic expressions are validated before generation. |
| Soft weights/losses | User model | Weight may be parameter expression but not decision-variable dependent. |
| Integrator choice | User or default | Reflected in generated metadata. |

## Outputs

| Output | Consumer | Meaning |
| --- | --- | --- |
| Generated C++ header | MiniSolver core | Static model interface implementation. |
| Static dimensions and names | Solver constructor/setters | Model metadata. |
| Cost/dynamics/constraint functions | Model evaluation and Riccati | Numerical packets. |
| Soft metadata and updater | Constraint semantics/model evaluation | Structural flags and per-knot weights. |
| Model fingerprint | Snapshot/replay | Model compatibility check. |

## Owned State

| State | Lifetime | Notes |
| --- | --- | --- |
| Python symbolic model object | Generation time | Not used by solver runtime. |
| Generated header | Build/runtime | Header-only model implementation. |

## Public API Surface

- `OptimalControlModel`
- `state()`, `control()`, `parameter()`
- `minimize()`, `add_residual()`, `subject_to()`, `subject_to_quad()`
- `generate()`

## Internal Contracts

- MiniModel owns model-specific symbolic semantics.
- Generated C++ should overwrite packets it owns and avoid redundant clears when
  full overwrite is provable.
- Solver core should consume structural metadata and numerical packets without
  reinterpreting user modeling intent.

## Hot-Path And Allocation Policy

- Hot path: Python codegen no; generated C++ yes
- Solve-time allocation allowed: no in generated C++ packets
- Notes: generated code shape can affect benchmark timing and must stay simple.

## Failure Semantics

| Failure | Status/reason path | Notes |
| --- | --- | --- |
| Invalid Python model | Python exception | Codegen boundary. |
| Unsupported symbolic dependency | Python exception | Prevents unsupported runtime KKT semantics. |
| Generated runtime non-finite packet | Solver numerical boundary | Core reports numerical failure where appropriate. |

## Tests And Evidence

| Evidence | Scope |
| --- | --- |
| `tests/minimodel/*.py` | DSL and codegen behavior. |
| `tests/test_asset_regressions.cpp` | Generated asset behavior. |
| `tests/test_soft_constraints.cpp` | Generated soft metadata/weights through solver. |
| `tests/test_implicit_sparse_riccati.cpp` | Generated implicit/fused paths. |

## Known Gaps

- MiniModel/codegen, generated packet ownership, and the generated C++ model
  interface are covered by
  [`../contracts/minimodel-codegen-contract.md`](../contracts/minimodel-codegen-contract.md),
  [`../contracts/generated-packets-contract.md`](../contracts/generated-packets-contract.md),
  and
  [`../contracts/model-interface-contract.md`](../contracts/model-interface-contract.md).
  Split public DSL and generated-runtime contracts further only when a future
  modeling feature needs separate ownership.
