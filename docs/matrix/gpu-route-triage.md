# GPU Route Triage

This branch investigates whether MPX/PCR and related GPU routes are worth
integrating into MiniSolver. It is intentionally scoped to standalone
microbenchmarks. `Backend::GPU_MPX` and `Backend::GPU_PCR` remain unsupported.

## Current Artifacts

| Artifact | Purpose | Evidence |
| --- | --- | --- |
| `tools/parallel_scan_gpu_bench.cu` | MPX/PCR-style affine prefix scan benchmark | Correctness error around `1e-15`; normal NMPC-scale horizons are slower on GPU |
| `docs/matrix/gpu-prefix-scan-microbench.md` | Scan benchmark contract and RTX 5080 results | Records reproduction commands and scale crossover observations |
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
| `tools/cuda_generated_packet_upload_bench.cu` | Generated-model packet eval/pack, persistent pinned staging, synthetic device packet fill, and H2D upload benchmark | Pinned H2D approaches `50 GB/s` on the largest buffers; synthetic device fill reaches roughly `850 GB/s` but does not include generated-model math |
| `docs/matrix/gpu-generated-packet-upload-microbench.md` | Generated packet upload benchmark contract and RTX 5080 results | Records why a GPU backend must fuse or amortize packet assembly and transfer |

The branch deliberately does not modify:

- `include/minisolver/algorithms/riccati_solver.h`
- `include/minisolver/solver/solver.h`
- `src/cuda/gpu_ops.cu`
- `SolverConfig`
- `Backend::GPU_*` behavior

## Prompt-To-Artifact Audit

Original request:

```text
新建一个分支，实现mpx和pcr，可以参考原来的gpu分支。
确认在不同规模问题上的加速效果，
建议第一步值对比矩阵分解的速度，而不是端到端对比，
这样工作量太大，也不好对比。
除了mpx和pcr，也可以探索其他路线。
```

| Requirement | Current evidence | Status |
| --- | --- | --- |
| New branch | `feat/gpu-mpx-pcr-microbench` pushed to remote | Done |
| Implement MPX/PCR | Standalone MPX-like Thrust scan and PCR-like Hillis-Steele scan exist in `parallel_scan_gpu_bench.cu`; scalar Riccati scan variant exists in `cuda_scalar_riccati_scan_bench.cu` | Partial, benchmark-only |
| Reference old GPU branch | Old branch was inspected; direct cherry-pick was rejected because it is highly stale | Done as design input |
| Confirm speedup at different scales | Scan benchmark covers `NX={2,4,8,12}`, `N={64..65536}`; scalar Riccati scan covers `N={64..65536}` | Done for scan microbenchmarks |
| First compare matrix decomposition speed | Batched Cholesky benchmark covers `DIM={4,8,12,16}`, `batch={1..65536}` | Done |
| Avoid end-to-end comparison | No end-to-end solver benchmark was added | Done |
| Explore other route | Batched small dense factorization, scalar Riccati scan, block-LFT scan, batched scalar Riccati, and batched block LQR Riccati routes added | Done |
| Working solver GPU backend | `Backend::GPU_MPX/GPU_PCR` still explicitly unsupported | Not done |

## Interpretation

### Prefix Scan / MPX-PCR Route

The scan route is mathematically relevant for parallel Riccati and recurrence
algorithms, but the current single-problem benchmark does not justify solver
integration. For normal NMPC horizon lengths, launch overhead and generic scan
cost dominate. Positive signals only appear for very large `N` or future batched
multi-problem workloads.

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
hard/L1/L2 recovery. For batch `1` and `256`, GPU is much slower than CPU; for
batch `4096` and `65536`, GPU starts to beat the threaded CPU baseline.

The generated packet upload benchmark adds the integration-cost side of the
story: if generated model evaluation and packet assembly stay on the CPU, the
host eval+pack and H2D upload cost can be milliseconds for large batched
packets. A persistent pinned-host/device-buffer staging variant removes
allocation and pageable-to-pinned copy from the timed path, but still pays
host-side generated-model evaluation plus H2D transfer each frame. A synthetic
device-side packet fill lower bound is much faster for large batches, but it
only measures packet-shaped writes and does not include generated-model math.
This means device-resident Riccati speedups should not be interpreted as
end-to-end backend speedups until packet assembly and transfer are fused or
amortized.

### Neural-Network / Differentiable Workloads

If MiniSolver is used inside a neural-network training or inference loop, the
best-supported GPU direction is still batched structured solves rather than a
single-horizon `Backend::GPU_MPX` or `Backend::GPU_PCR` replacement.

The useful workload shape is:

- many independent horizons from a training batch;
- many sampled controls or policy rollouts;
- many replay/corpus cases;
- many implicit-differentiation linear solves with shared structure.

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

1. If GPU work continues, prototype real device-side generated-model evaluation
   for one small generated model before touching `RiccatiSolver`.
2. If a real workload appears, benchmark batched MPC or sampled-control
   workloads instead of single-horizon solve time.
3. Keep `src/cuda/gpu_ops.cu` as unsupported until the integration gate passes.
