# MiniSolver Module Inventory

This directory documents implementation modules and their boundaries. Module
docs describe what each part of the code owns before behavior contracts assign
stable requirements.

Use module docs to reduce change amplification. When a feature would require
many files to change, first identify the owning module and the narrow contract
that should carry the behavior.

## Files

| File | Purpose |
| --- | --- |
| [`_template.md`](_template.md) | Starting point for module documents. |
| [`module-inventory.md`](module-inventory.md) | Index of all tracked modules and their documentation status. |

## Module Rules

1. A module document describes current ownership; it is not a wishlist.
2. Define inputs and outputs before proposing a new contract.
3. Mark hot-path and allocation expectations explicitly.
4. Link related tests and known gaps.
5. Keep public API, internal workspace, and generated model responsibilities
   separate.
