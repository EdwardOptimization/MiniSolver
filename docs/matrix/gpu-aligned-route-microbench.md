# Aligned CUDA Route Microbenchmarks

Last updated: 2026-05-18

This note supersedes cross-route conclusions drawn from earlier route-specific
tables. The earlier probes used different dimensions, horizons, and batch
counts. Those results remain useful as individual smoke tests, but they do not
support a fair route comparison.

## Contract

The aligned re-run uses this common grid where each route can support it:

```text
(NX, NU) in {(4, 2), (8, 4)}
N in {32, 128}
batch in {1, 256, 4096}
```

Large-horizon stress rows such as `N=65536` are intentionally not part of this
aligned grid. They remain useful for scan-specific crossover investigation, but
they cannot be compared fairly against structured block-tridiagonal
factorization, generated packet upload, or batched Riccati rows that do not
cover the same shape.

Mappings and limits:

- stage-local Cholesky uses `DIM = NX + NU`, so `DIM in {6, 12}`;
- block-tridiagonal factorization uses `block_dim = NX + NU`, so
  `block_dim in {6, 12}`;
- batched block Riccati uses the full `(NX, NU, N, batch)` grid;
- generated packet upload has exact aligned `CarModel` coverage for `(4, 2)`,
  and real generated `BicycleExtModel` coverage for `(6, 2)`;
- prefix/block-LFT scan probes only cover `NX=4` in the aligned batched run;
- dense full-KKT is not a candidate strategy. It is retained only as a
  historical rejected probe because it does not exploit the OCP block structure.

Machine:

```text
GPU: NVIDIA GeForce RTX 5080
CUDA compiler: NVIDIA 13.1.115
CUDA architecture: native, resolved to sm_120
Build: .build_cuda_bench, host/device transfer excluded unless stated
```

Commands:

```bash
cmake --build .build_cuda_bench --target \
  cuda_batched_factor_bench \
  cuda_batched_scalar_riccati_bench \
  cuda_batched_lqr_riccati_bench \
  cuda_generated_packet_upload_bench \
  parallel_scan_gpu_bench \
  cuda_block_lft_scan_bench \
  cuda_block_tridiag_factor_bench -j

.build_cuda_bench/cuda_batched_factor_bench
.build_cuda_bench/cuda_batched_scalar_riccati_bench
.build_cuda_bench/cuda_batched_lqr_riccati_bench
.build_cuda_bench/cuda_generated_packet_upload_bench
.build_cuda_bench/parallel_scan_gpu_bench
.build_cuda_bench/cuda_block_lft_scan_bench
.build_cuda_bench/cuda_block_tridiag_factor_bench
```

## Route 1: Structured Block-Tridiagonal KKT Factorization

Strategy 1 means exploiting the OCP block structure directly. The aligned
candidate route is therefore the block-tridiagonal benchmark, not dense
full-KKT assembly.

| Block dim | N | Batch | Best CPU us | GPU us | GPU vs best CPU | Error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 32 | 1 | 8.31 | 1008.10 | 0.01x | 6.94e-18 |
| 6 | 32 | 256 | 1524.27 | 1808.46 | 0.84x | 1.04e-17 |
| 6 | 32 | 4096 | 3495.36 | 2231.81 | 1.57x | 1.04e-17 |
| 6 | 128 | 1 | 33.87 | 4172.82 | 0.01x | 6.94e-18 |
| 6 | 128 | 256 | 1797.35 | 7407.83 | 0.24x | 1.04e-17 |
| 6 | 128 | 4096 | 11619.64 | 9437.02 | 1.23x | 1.04e-17 |
| 12 | 32 | 1 | 38.44 | 4449.93 | 0.01x | 6.94e-18 |
| 12 | 32 | 256 | 1791.05 | 12337.86 | 0.15x | 1.04e-17 |
| 12 | 32 | 4096 | 12543.33 | 14176.67 | 0.88x | 1.04e-17 |
| 12 | 128 | 1 | 157.27 | 18608.62 | 0.01x | 1.04e-17 |
| 12 | 128 | 256 | 3496.20 | 53830.88 | 0.06x | 1.04e-17 |
| 12 | 128 | 4096 | 37306.24 | 56851.20 | 0.66x | 1.04e-17 |

The block-sparse route has limited positive signal only at very large batch and
small block size. It does not support a single-horizon GPU backend.

Extended single-horizon stress rows for Strategy 1 were run separately from the
aligned gate to answer whether much larger horizons change that conclusion:

| Block dim | N | Batch | Best CPU us | GPU us | GPU vs best CPU | Error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 512 | 1 | 140.02 | 19768.11 | 0.01x | 1.04e-17 |
| 6 | 4096 | 1 | 1094.02 | 181064.85 | 0.01x | 1.04e-17 |
| 6 | 16384 | 1 | 4670.54 | 725664.06 | 0.01x | 1.04e-17 |
| 6 | 65536 | 1 | 19477.69 | 2976203.12 | 0.01x | 1.04e-17 |
| 12 | 512 | 1 | 633.59 | 92774.88 | 0.01x | 1.04e-17 |
| 12 | 4096 | 1 | 5283.87 | 763560.55 | 0.01x | 1.04e-17 |
| 12 | 16384 | 1 | 22551.46 | 3055434.57 | 0.01x | 1.04e-17 |
| 12 | 65536 | 1 | 153913.46 | 12233009.77 | 0.01x | 1.04e-17 |

These rows validate the structured block-tridiagonal route at `N=65536`, but
they also show that the current one-thread-per-system GPU kernel does not cross
over even at very large single-horizon `N`. A credible Strategy 1 backend would
need intra-system parallelism such as block-parallel factorization, cyclic
reduction, or factorization reuse.

Dense full-KKT assembly/factorization is excluded from Strategy 1 because it
destroys the OCP sparsity. The historical dense probe remains documented in
`gpu-full-kkt-factor-microbench.md` only as a rejected anti-pattern.

## Route 2: Stage-Local / Batched Riccati Work

### Batched Stage-Local Cholesky

| DIM | Batch | Best CPU us | GPU simple us | GPU coop us | Best GPU speedup | Error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 1 | 0.05 | 10.70 | 9.01 | 0.01x | 0.00e+00 |
| 6 | 256 | 11.47 | 15.69 | 9.19 | 1.25x | 3.47e-18 |
| 6 | 4096 | 43.45 | 14.91 | 39.15 | 2.91x | 2.22e-16 |
| 12 | 1 | 0.13 | 28.90 | 12.90 | 0.01x | 3.47e-18 |
| 12 | 256 | 34.00 | 74.68 | 14.70 | 2.31x | 2.22e-16 |
| 12 | 4096 | 105.63 | 76.11 | 102.79 | 1.39x | 2.22e-16 |

### Batched Scalar Riccati

| N | Batch | CPU us | GPU us | GPU speedup | Error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 1 | 0.09 | 16.53 | 0.01x | 0.00e+00 |
| 32 | 256 | 20.79 | 18.78 | 1.11x | 3.55e-15 |
| 32 | 4096 | 329.05 | 19.32 | 17.03x | 5.33e-15 |
| 128 | 1 | 0.45 | 59.78 | 0.01x | 0.00e+00 |
| 128 | 256 | 115.40 | 63.93 | 1.81x | 3.55e-15 |
| 128 | 4096 | 1898.18 | 63.80 | 29.75x | 5.33e-15 |

### Batched Barrier-Affine Block Riccati

| NX | NU | N | Batch | Threaded CPU us | GPU us | GPU vs threaded CPU | Error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 2 | 32 | 1 | 3.59 | 693.67 | 0.01x | 8.88e-16 |
| 4 | 2 | 32 | 256 | 491.49 | 786.56 | 0.62x | 8.88e-16 |
| 4 | 2 | 32 | 4096 | 1770.68 | 788.28 | 2.25x | 8.88e-16 |
| 4 | 2 | 128 | 1 | 14.41 | 2738.43 | 0.01x | 1.78e-15 |
| 4 | 2 | 128 | 256 | 1914.96 | 3097.82 | 0.62x | 1.78e-15 |
| 4 | 2 | 128 | 4096 | 5269.82 | 3099.48 | 1.70x | 1.78e-15 |
| 8 | 4 | 32 | 1 | 22.72 | 4054.17 | 0.01x | 1.78e-15 |
| 8 | 4 | 32 | 256 | 2950.36 | 4506.34 | 0.65x | 1.78e-15 |
| 8 | 4 | 32 | 4096 | 6426.10 | 4504.15 | 1.43x | 1.78e-15 |
| 8 | 4 | 128 | 1 | 92.79 | 16190.26 | 0.01x | 1.78e-15 |
| 8 | 4 | 128 | 256 | 11784.36 | 17983.62 | 0.66x | 1.78e-15 |
| 8 | 4 | 128 | 4096 | 25941.09 | 17966.39 | 1.44x | 1.78e-15 |

This is the strongest aligned route-2 evidence: GPU only becomes useful once
there are thousands of independent horizons.

### Generated Packet Integration Cost

For `CarModel` `(NX, NU) = (4, 2)`, CPU eval+pack plus H2D remains a material
cost, while hand-transcribed CUDA packet assembly is correct and faster for
large batches.

| N | Batch | CPU eval+pack us | Pinned H2D us | CUDA exact packet us | Max packet error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 1 | 13.69 | 29.81 | 16.58 | 8.88e-16 |
| 32 | 256 | 955.37 | 298.08 | 29.58 | 1.78e-15 |
| 32 | 4096 | 16075.13 | 2329.84 | 586.32 | 3.55e-15 |
| 128 | 1 | 27.94 | 32.72 | 22.92 | 8.88e-16 |
| 128 | 256 | 3986.15 | 713.14 | 95.76 | 7.11e-15 |
| 128 | 4096 | 65811.93 | 9186.00 | 4471.84 | 7.11e-15 |

The device-resident Riccati kernel speedups are not end-to-end backend speedups
unless generated packet assembly and transfer are fused or amortized.

## Route 3: Horizon-Parallel Scan / MPX-PCR

### Batched Affine Prefix Scan

| NX | N | Batch | CPU us | MPX-like us | PCR-like us | Best GPU speedup | Error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 32 | 1 | 0.36 | 68.21 | 40.50 | 0.01x | 7.77e-16 |
| 4 | 32 | 256 | 97.32 | 61.53 | 52.57 | 1.85x | 1.22e-15 |
| 4 | 32 | 4096 | 1713.41 | 238.50 | 277.67 | 7.18x | 1.44e-15 |
| 4 | 128 | 1 | 1.45 | 69.92 | 56.65 | 0.03x | 9.44e-16 |
| 4 | 128 | 256 | 397.92 | 147.35 | 112.93 | 3.52x | 1.22e-15 |
| 4 | 128 | 4096 | 7003.13 | 1817.10 | 1424.99 | 4.91x | 1.44e-15 |

`NX=8` is not reported for this batched `scan_by_key` probe because the CUB
kernel for the larger operator exceeded the device shared-memory limit during
compilation. This is itself a route-3 coverage limitation, not a speed result.

### Batched Block-LFT Scan

| NX | N | Batch | CPU us | MPX-like us | PCR-like us | PCR speedup | Error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 32 | 1 | 1.31 | 146.63 | 101.02 | 0.01x | 2.89e-15 |
| 4 | 32 | 256 | 386.12 | n/a | 148.54 | 2.60x | 4.22e-15 |
| 4 | 32 | 4096 | 8004.93 | n/a | 1182.06 | 6.77x | 4.88e-15 |
| 4 | 128 | 1 | 5.22 | 241.76 | 177.69 | 0.03x | 4.44e-15 |
| 4 | 128 | 256 | 1400.38 | n/a | 520.78 | 2.69x | 8.66e-15 |
| 4 | 128 | 4096 | 24472.44 | n/a | 5933.55 | 4.12x | 1.04e-14 |

The batched scan route has positive large-batch microkernel signal, but it does
not yet cover a full block Riccati direction with feedback/RHS recovery.

## Corrected Gate Decision

The aligned run changes the confidence level of the previous conclusion:

- `batch=1`: every GPU route is slower by orders of magnitude. Do not enable a
  normal single-horizon GPU backend.
- `batch=256`: results are mixed. Some stage-local or scan kernels cross over,
  but batched block Riccati is still slower than threaded CPU.
- `batch=4096`: several GPU kernels cross over, especially scalar/batched
  Riccati, packet assembly lower bounds, and batched scans.
- dense full-KKT is excluded from the candidate set because it does not exploit
  structure; Strategy 1 means the structured block-tridiagonal route.
- route 3 currently lacks full `(NX, NU) = (8, 4)` batched scan coverage.

Therefore the backend gate remains closed:

```text
Do not enable Backend::GPU_MPX or Backend::GPU_PCR.
Keep MiniSolver's normal single-OCP NMPC solve path on CPU.
```

The credible next GPU target is not a drop-in backend for one NMPC problem. It
is a separate batched/differentiable workload path:

```text
MiniModel CUDA packet emission
  -> device-resident batched packets
  -> batched structured Riccati/KKT kernels
  -> explicit implicit-differentiation or sampled-control workloads
```

Any future cross-route claim must use the aligned grid above or explicitly state
why a route cannot cover it.

## Extended Route-Specific Stress Rows

The prefix-scan and block-tridiagonal tools still emit single-horizon stress
rows after the aligned rows. These include `N=65536`, but they answer a
different question:

```text
At what very large horizon does a route-specific primitive itself cross over?
```

They should not be used to decide whether a normal MiniSolver NMPC backend is
worth enabling.
