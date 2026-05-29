# Soft Constraint Weight API Plan

## Problem

MiniModel currently stores one static `(type, weight)` pair per constraint row.
That makes two user-facing cases awkward:

1. a soft weight cannot be a model parameter updated at runtime;
2. one physical constraint cannot carry both L1 and L2 soft penalties without
   duplicating the constraint row.

## Contract

MiniModel owns the modeling semantics. C++ must not decide whether a row is L1
or L2 by inspecting the runtime weight value.

The generated model should expose fixed structure:

```cpp
static constexpr std::array<bool, NC> constraint_has_l1 = {...};
static constexpr std::array<bool, NC> constraint_has_l2 = {...};
static constexpr bool any_l1_constraints = ...;
static constexpr bool any_l2_constraints = ...;
```

Runtime soft weights live on each knot:

```cpp
MSVec<T, NC> l1_weight;
MSVec<T, NC> l2_weight;
```

Together, the trajectory stores two `(N + 1) x NC` weight tables. A generated
model function refreshes both tables for one knot:

```cpp
template <typename T>
static void update_soft_constraint_weights(KnotPoint<T, NX, NU, NC, NP>& kp);
```

The function assigns constants, parameters, or parameter-only expressions into
`kp.l1_weight` and `kp.l2_weight`. It does not decide whether a row has L1 or L2
semantics.

## MiniModel API

Keep the existing simple form:

```python
model.subject_to(expr <= 0, weight=a, loss="L1")
model.subject_to(expr <= 0, weight=b, loss="L2")
```

Add combined soft penalties on one row:

```python
model.subject_to(expr <= 0, weight=[a, b], loss=["L1", "L2"])
```

For batches, normalize each row independently:

```python
model.subject_to(
    [expr1 <= 0, expr2 <= 0],
    weight=[[a, b], c],
    loss=[["L1", "L2"], "L2"],
)
```

Internally, normalize to per-row metadata:

```python
{"L1": a, "L2": b}
{"L2": c}
```

Soft weight expressions may be numeric constants, parameters, or parameter-only
expressions. They must not depend on state or control variables.

## Solver Integration

The solver reads structure and values separately:

```cpp
if constexpr (Model::any_l1_constraints) {
    if (Model::constraint_has_l1[i]) {
        const double w = kp.l1_weight(i);
        // existing L1 path
    }
}
if constexpr (Model::any_l2_constraints) {
    if (Model::constraint_has_l2[i]) {
        const double w = kp.l2_weight(i);
        // existing L2 path
    }
}
```

Hand-written test or example models must use the same structure arrays and
weight updater. The solver core does not infer L1/L2 semantics from legacy
`constraint_types` or `constraint_weights`.

## Validation

- MiniModel test: parameter soft weights generate structure arrays and a weight
  update function.
- MiniModel test: one constraint row can have both L1 and L2 soft weights.
- Solver test: changing a parameter weight at runtime changes the per-stage
  stored weight before solving.
- Regression tests: existing L1/L2 soft constraint behavior remains unchanged
  after internal test models migrate to the new structure/updater contract.
