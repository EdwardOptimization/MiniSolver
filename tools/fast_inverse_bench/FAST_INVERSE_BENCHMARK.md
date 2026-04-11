# fast_inverse Microbenchmark

This microbenchmark exists to quantify the overhead of adding SPD (positive definite) checks to `minisolver::fast_inverse()`.

Background:
- `fast_inverse()` is used as a fast path when `NU <= 3` in the Riccati backward pass.
- We changed `fast_inverse()` to be **SPD-only** (Sylvester leading principal minors) and added a fallback to **freeze** control dimensions when `Quu` is not SPD, rather than hard-failing or producing an unstable direction.

This benchmark isolates only the `fast_inverse()` routine itself by comparing:
- **legacy_fast_inverse**: the previous implementation (invertible check only via `abs(det)`).
- **minisolver::fast_inverse**: the current implementation (SPD checks).

## Build

```bash
cmake --build build -j --target fast_inverse_bench
```

## Run

```bash
./build/fast_inverse_bench
./build/fast_inverse_bench --iters 50000000 --mats 4096 --repeats 7 --epsilon 1e-9
```

Arguments:
- `--iters`: number of inversions per case (default `20,000,000`)
- `--mats`: number of pre-generated SPD matrices to cycle through (default `4096`)
- `--repeats`: repeat count, best run is reported (default `5`)
- `--epsilon`: singular/SPD threshold (default `1e-9`)

## Sample Output (Machine-Dependent)

Example (Release build, best-of repeats) on:
- CPU: AMD Ryzen 9 9950X3D 16-Core Processor
- Command: `./build/fast_inverse_bench --iters 20000000 --mats 4096 --repeats 5 --epsilon 1e-9`

```text
fast_inverse microbenchmark (before/after SPD checks)
iters=20000000 mats=4096 repeats=5 epsilon=1.000000e-09

1x1 SPD    legacy 2.303 ns/call | new 2.263 ns/call | delta -1.74%
2x2 SPD    legacy 2.518 ns/call | new 2.496 ns/call | delta -0.89%
3x3 SPD    legacy 3.259 ns/call | new 3.453 ns/call | delta +5.96%
```

Notes:
- Expect deltas to be small (a few extra multiplies/comparisons).
- Use this benchmark to catch accidental slowdowns (e.g. introducing a decomposition in the `NU<=3` fast path).
