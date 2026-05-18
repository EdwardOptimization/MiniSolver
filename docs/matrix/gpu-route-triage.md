# GPU Route Triage

Last updated: 2026-05-18

This branch investigates whether MPX/PCR and related GPU routes are worth
integrating into MiniSolver. It is intentionally scoped to standalone
microbenchmarks. `Backend::GPU_MPX` and `Backend::GPU_PCR` remain unsupported.

## Final Gate Conclusion

Do not integrate a normal GPU backend for MiniSolver's ordinary fixed-size
single-OCP NMPC solve path.

The current evidence says:

- for `batch=1`, every measured GPU route is slower by orders of magnitude;
- the structured block-tridiagonal Strategy 1 route remains around `0.01x`
  GPU-vs-CPU speedup even at `N=65536`;
- MPX/PCR-style scan primitives are mathematically relevant, but the measured
  scan routes do not cover a complete OCP direction solve and still do not
  justify `Backend::GPU_MPX` or `Backend::GPU_PCR`;
- positive GPU signal appears only when there are many independent horizons,
  generated packets are device-resident or amortized, and the workload is
  explicitly batched.

Therefore the product decision is:

```text
Keep the normal MiniSolver OCP backend on CPU.
Do not enable Backend::GPU_MPX or Backend::GPU_PCR.
Keep GPU work limited to batched/differentiable/replay workloads until a new
benchmark corpus proves an end-to-end backend benefit.
```

This does not mean GPU acceleration is useless for all OCP-related workloads.
It means the current branch found no useful GPU backend path for the normal
single-problem online NMPC tick that MiniSolver primarily targets.

## Alignment Correction

Earlier route-specific tables used different dimensions, horizons, and batch
counts. They are valid smoke/probe results for their own tools, but they should
not be used for cross-route speed conclusions.

The current cross-route gate is based on the aligned re-run in
`docs/matrix/gpu-aligned-route-microbench.md`:

```text
(NX, NU) in {(4, 2), (8, 4)}
N in {32, 128}
batch in {1, 256, 4096}
```

Strategy 1 is the structured block-tridiagonal route. Dense full-KKT
assembly/factorization is excluded from the candidate set because it does not
exploit the OCP block structure. Prefix/block-LFT scan probes only cover the
`NX=4` batched scan subset. These coverage gaps are part of the route decision,
not missing positive results.

Route-specific large-horizon stress rows, including `N=65536`, are still useful
for primitive crossover analysis. They now cover both the prefix-scan route and
the structured block-tridiagonal route, but they remain excluded from the
aligned cross-route gate because they answer single-route stress questions.

## Current Artifacts

| Artifact | Purpose | Evidence |
| --- | --- | --- |
| `tools/parallel_scan_gpu_bench.cu` | MPX/PCR-style affine prefix scan benchmark | Correctness error around `1e-15`; normal NMPC-scale horizons are slower on GPU |
| `docs/matrix/gpu-prefix-scan-microbench.md` | Scan benchmark contract and RTX 5080 results | Records reproduction commands and scale crossover observations |
| `tools/cuda_full_kkt_factor_bench.cu` | Historical rejected dense full-KKT probe | Correctness and residual around `1e-16`; dense full-matrix route is excluded because it does not exploit OCP block structure |
| `docs/matrix/gpu-full-kkt-factor-microbench.md` | Dense full-KKT anti-pattern note | Records why dense full-KKT assembly is not Strategy 1 |
| `tools/cuda_block_tridiag_factor_bench.cu` | Explicit block-tridiagonal Cholesky/solve for synthetic block-sparse Newton systems | Correctness around `1e-17`; simple one-thread-per-system GPU route approaches CPU only for small blocks and medium batch |
| `docs/matrix/gpu-block-tridiag-factor-microbench.md` | Block-tridiagonal factorization route contract and RTX 5080 results | Records why sparse/block KKT work needs block-parallel kernels or cyclic reduction before backend integration |
| `tools/cuda_batched_factor_bench.cu` | Batched small dense Cholesky benchmark with sequential/threaded/Eigen CPU, simple GPU, and tuned cooperative GPU baselines | Correctness error around `1e-15`; GPU wins are workload-shape dependent |
| `docs/matrix/gpu-batched-factor-microbench.md` | Batched factorization benchmark contract and RTX 5080 results | Records reproduction commands and batch-size crossover observations |
| `tools/cuda_scalar_riccati_scan_bench.cu` | Scalar Riccati recurrence as MPX/PCR-style fractional-linear scan | Correctness error around `1e-14`; large `N` crossover only |
| `docs/matrix/gpu-scalar-riccati-scan-microbench.md` | Riccati-specific scan benchmark contract and RTX 5080 results | Records reproduction commands and why this still is not a backend |
| `tools/cuda_batched_scalar_riccati_bench.cu` | Many independent short scalar Riccati recursions | Correctness error around `1e-14`; strong speedup once batch reaches hundreds/thousands |
| `docs/matrix/gpu-batched-scalar-riccati-microbench.md` | Batched short-horizon Riccati benchmark contract and RTX 5080 results | Records reproduction commands and why batched workloads are the strongest GPU signal |
| `tools/cuda_block_lft_scan_bench.cu` | Block linear-fractional transform scan as a block-Riccati-adjacent MPX/PCR route | Correctness error around `1e-14` to `1e-13`; only marginal large-`N` PCR crossover |
| `docs/matrix/gpu-block-lft-scan-microbench.md` | Block-LFT scan benchmark contract and RTX 5080 results | Records why block operator scans still do not justify a normal GPU backend |
| `tools/cuda_batched_lqr_riccati_bench.cu` | Batched barrier-affine block Riccati direction recursion with synthetic defect RHS and mixed hard/L1/L2 recovery | Correctness error around `1e-15`; large batches show GPU speedup, small batches do not |
| `docs/matrix/gpu-batched-lqr-riccati-microbench.md` | Batched barrier-affine block Riccati benchmark contract and RTX 5080 results | Records the strongest Riccati-specific evidence for batched GPU workloads |
| `tools/cuda_generated_packet_upload_bench.cu` | Generated-model packet eval/pack, persistent pinned staging, synthetic device packet fill, hand-transcribed `CarModel` CUDA packet assembly, and H2D upload benchmark | Pinned H2D approaches `50 GB/s`; synthetic device fill reaches roughly `850 GB/s`; CUDA `CarModel` packet assembly matches CPU packets to around `1e-14` and is faster at large batch |
| `docs/matrix/gpu-generated-packet-upload-microbench.md` | Generated packet upload benchmark contract and RTX 5080 results | Records why a GPU backend must fuse or amortize packet assembly and transfer |
| `docs/matrix/gpu-aligned-route-microbench.md` | Aligned cross-route re-run over the shared route grid | Supersedes earlier cross-route interpretations from heterogeneous case tables |

The branch deliberately does not modify:

- `include/minisolver/algorithms/riccati_solver.h`
- `include/minisolver/solver/solver.h`
- `src/cuda/gpu_ops.cu`
- `SolverConfig`
- `Backend::GPU_*` behavior

## Prompt-To-Artifact Audit

Original request:

```text
Create a branch to investigate MPX/PCR-style GPU routes, using the old GPU
branch as reference if useful. Confirm speedup across problem scales. Start by
comparing matrix decomposition or recurrence kernels instead of full end-to-end
solver timing, and explore other candidate routes if relevant.
```

| Requirement | Current evidence | Status |
| --- | --- | --- |
| New branch | `feat/gpu-mpx-pcr-microbench` exists as the GPU research branch; local evidence commits should be pushed only after commit cleanup | Done |
| Implement MPX/PCR | Standalone MPX-like Thrust scan and PCR-like Hillis-Steele scan exist in `parallel_scan_gpu_bench.cu`; scalar Riccati and block-LFT scan variants cover Riccati-adjacent recurrences | Done for exploratory benchmarks |
| Reference old GPU branch | Old branch was inspected; direct cherry-pick was rejected because it is highly stale | Done as design input |
| Confirm speedup at different scales | Aligned route re-run covers `(NX,NU)={(4,2),(8,4)}`, `N={32,128}`, `batch={1,256,4096}` where each route can support it; route-specific larger sweeps are kept as smoke/probe data only | Done with coverage limits recorded |
| First compare matrix decomposition speed | Aligned batched Cholesky benchmark covers `DIM={6,12}`, `batch={1,256,4096}` | Done |
| Avoid end-to-end comparison | No end-to-end solver benchmark was added | Done |
| Structured block-tridiagonal factorization route | Explicit block-tridiagonal factorization benchmark exists with CPU baselines, GPU timing, and correctness error; dense full-KKT is retained only as a rejected anti-pattern probe | Done as route probe |
| Explore other route | Batched small dense factorization, scalar Riccati scan, block-LFT scan, batched scalar Riccati, and batched block LQR Riccati routes added | Done |

Working `Backend::GPU_MPX` / `Backend::GPU_PCR` integration is intentionally
not part of this exploratory microbenchmark branch. It remains gated below.

## Interpretation

### Structured Block-Tridiagonal Route

Strategy 1 means exploiting the OCP block structure instead of assembling the
whole KKT into a dense matrix. Dense full-KKT is therefore not a candidate
route. It remains only as a historical anti-pattern probe.

The block-tridiagonal benchmark keeps the explicit sparse/block structure and
solves a regularized normal-equation/Schur-complement view of the Newton
system. On the aligned grid it verifies correctness around `1e-17`. It only
beats the best CPU baseline for very large batch and small block size, for
example `block_dim=6, N=32, batch=4096` at `1.57x`; larger block/horizon rows
remain slower.

The separate single-horizon stress rows include `N=65536` for
`block_dim={6,12}`. They remain at roughly `0.01x` GPU-vs-CPU speedup with this
one-thread-per-system kernel, so large `N` alone does not rescue Strategy 1.

A serious Strategy 1 GPU route would need true block-sparse storage,
symbolic/numeric factorization reuse, and block-parallel tridiagonal
factorization or cyclic reduction. A one-thread-per-system block solver is not
enough.

### Prefix Scan / MPX-PCR Route

The scan route is mathematically relevant for parallel Riccati and recurrence
algorithms, but it still does not justify solver integration. In the aligned
batched re-run, `NX=4` scans show positive large-batch microkernel speedups,
but `batch=1` remains orders of magnitude slower. The `NX=8` batched
`scan_by_key` probe exceeded CUB shared-memory limits and is therefore a
coverage gap, not a positive result.

The scalar Riccati scan benchmark confirms the same conclusion in a more
Riccati-specific setting: the fractional-linear transform scan is correct, but
single-problem horizons below several thousand stages remain slower than CPU.

The block-LFT scan benchmark moves one step closer to block Riccati operator
composition. It still shows only marginal PCR crossover at very large `N`, and
it excludes operator assembly, feedback recovery, RHS propagation, transfer
cost, and solver residual checks. It therefore remains design evidence, not a
backend implementation.

The batched barrier-affine block Riccati benchmark is the closest current
artifact to a real Riccati direction workload. It executes Hessian,
feedforward, stage-varying synthetic barrier-derived packet, dynamics-defect
RHS, and mixed hard/L1/L2 recovery work and shows GPU wins only when there are
thousands of independent horizons. It reinforces that the credible near-term
GPU target is batched work, not a single-horizon backend.

### Batched Factorization Route

Batched small dense Cholesky is more promising for workloads with many
independent small systems. The simple one-thread-per-matrix kernel is strongest
for very small matrices at large batch. The cooperative one-block-per-matrix
kernel helps some larger-matrix cases, but it is not uniformly better and still
loses to the best CPU baseline for several large-batch shapes. A fixed-size
Eigen `LLT` baseline improves the smallest `DIM=4` CPU case but does not erase
the batched GPU signal. A basic cooperative thread sweep over `8/16/32/64`
threads helps selected `DIM=12/16` cases but still does not make the GPU path
uniformly better. This suggests GPU acceleration should first target specific
batched workloads instead of a generic single-problem backend:

- many MPC problems;
- many shooting guesses;
- sampling-based rollout batches;
- replay/corpus processing;
- batched local QP or collision-oracle subproblems.

It does not justify a single-problem GPU backend for ordinary NMPC horizons.

### Batched Short-Horizon Route

The batched scalar Riccati benchmark is the strongest signal on this branch.
For horizons `N=32..256`, a single problem is much slower on GPU, but batches of
hundreds already cross over and batches of thousands show large speedups. This
points to batched MPC, sampled MPC, multiple shooting guesses, replay/corpus
processing, and differentiable workloads as better GPU targets than replacing
one CPU Riccati solve.

The batched barrier-affine block Riccati benchmark strengthens this conclusion
with a multi-state, multi-control, feedforward recursion plus synthetic
constraint/barrier packet assembly, defect RHS propagation, and mixed
hard/L1/L2 recovery. On the aligned grid, batch `1` is orders of magnitude
slower on GPU, batch `256` is still slower than threaded CPU, and batch `4096`
crosses over at roughly `1.4x` to `2.25x` depending on dimension and horizon.

The generated packet upload benchmark adds the integration-cost side of the
story: if generated model evaluation and packet assembly stay on the CPU, the
host eval+pack and H2D upload cost can be milliseconds for large batched
packets. A persistent pinned-host/device-buffer staging variant removes
allocation and pageable-to-pinned copy from the timed path, but still pays
host-side generated-model evaluation plus H2D transfer each frame. A synthetic
device-side packet fill lower bound is much faster for large batches, but it
only measures packet-shaped writes and does not include generated-model math.
A hand-transcribed CUDA `CarModel` exact packet assembly kernel matches the
generated CPU packet to around `1e-14` max error and is much faster at large
batch. This is a positive signal for future MiniModel CUDA packet emission, but
it is still a single-model prototype. Device-resident Riccati speedups should
not be interpreted as end-to-end backend speedups until packet assembly and
transfer are fused or amortized.

### Neural-Network / Differentiable Workloads

If MiniSolver is used inside a neural-network training or inference loop, the
best-supported GPU direction is still batched structured solves rather than a
single-horizon `Backend::GPU_MPX` or `Backend::GPU_PCR` replacement.

The useful workload shape is:

- many independent horizons from a training batch;
- many sampled controls or policy rollouts;
- many replay/corpus cases;
- many implicit-differentiation linear solves with shared structure.

The recommended MiniSolver layer contract is therefore:

1. keep the ordinary forward solve on the normal SQP/IPM/Riccati path;
2. mark trainable problem inputs explicitly, for example through future
   `diff_parameter` metadata in MiniModel;
3. compute gradients with an explicit implicit-differentiation pass over the
   converged KKT/Riccati system;
4. accelerate the forward/backward pair with batched device-resident generated
   packets and batched structured Riccati/KKT kernels when the training batch is
   large enough.

For that use case, the current batched Riccati results are directly relevant:
the GPU only becomes competitive once there is enough independent work to fill
the device. This also matches the likely differentiable-solver contract: the
forward solve should remain the normal MiniSolver path, and any extra cost
should come from the explicit implicit-differentiation pass, not from changing
ordinary solves into a GPU-only mode.

PDLP/HPR-LP-style first-order LP routes are interesting for very large sparse LP
problems, but they are not the first fit for MiniSolver's small dense structured
Riccati direction solves. Using them here would require reformulating the local
QP/KKT work into a generic LP or first-order fixed-point problem, losing much of
the NMPC block structure that MiniSolver already exploits. They should be
revisited only if a future differentiable workload is dominated by large sparse
linear or conic subproblems rather than batched Riccati/KKT solves.

Do not treat PDLP/HPR-LP as a matrix-factorization replacement for this branch.
Their useful GPU property is avoiding factorization through large sparse
matrix-vector iterations; MiniSolver's useful property is the opposite: small
structured Newton systems whose factorization and implicit backward pass can
reuse the OCP/Riccati structure.

## Backend Integration Gate

Do not enable `Backend::GPU_MPX` or `Backend::GPU_PCR` until all of these are
true:

1. A GPU kernel path matches CPU Riccati/KKT directions on deterministic test
   problems.
2. The benchmark covers the intended workload shape, not only synthetic large
   batches.
3. Runtime is measured with accuracy, residuals, and failure modes, not only
   kernel time.
4. Host/device transfer and assembly cost are included or explicitly fused away.
5. A CPU SIMD/threaded baseline has been measured.
6. The target use case is explicit: single NMPC instance, batched NMPC, sampled
   control, replay corpus, or differentiable/batched training.

Until this gate is met, GPU support should remain an exploratory benchmark
area, not a solver backend claim.

## Recommended Next Steps

1. If GPU work continues, prototype generated CUDA packet emission in MiniModel
   for one small generated model before touching `RiccatiSolver`.
2. If a real workload appears, benchmark batched MPC or sampled-control
   workloads instead of single-horizon solve time.
3. Keep `src/cuda/gpu_ops.cu` as unsupported until the integration gate passes.
