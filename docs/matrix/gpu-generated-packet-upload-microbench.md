# CUDA Generated Packet Upload Microbenchmark

Last updated: 2026-05-18

> Cross-route note: the aligned route re-run is recorded in
> `docs/matrix/gpu-aligned-route-microbench.md`. Treat this file as a
> route-specific probe note; do not use older heterogeneous rows here for
> cross-route speed conclusions.

This note records a standalone benchmark for the cost of taking packets from
generated MiniSolver models on the CPU and uploading them to a CUDA device. It
does not run a GPU solver backend. Its purpose is to quantify the integration
cost that a future GPU Riccati backend would need to hide or fuse away.

## Benchmark Contract

The benchmark uses two generated models already present in the repository:

- `examples/01_car_tutorial/generated/car_model.h`
- `examples/02_advanced_bicycle/generated/bicycleextmodel.h`

For each stage and batch sample, the host code:

1. seeds a `KnotPoint`;
2. calls `Model::compute_exact(..., RK4_EXPLICIT, 0.05)`;
3. packs the Riccati-relevant packet fields:
   `A/B/C/D/Q/R/H/q/r/g/s/lam/soft_s/f_resid`;
4. uploads the contiguous packet buffer to the GPU with pageable and pinned
   host memory variants.
5. measures a persistent staging variant that reuses pinned host and device
   buffers, packs directly into pinned memory, and uploads with `cudaMemcpyAsync`
   followed by stream synchronization.
6. measures a lower-bound device-side packet fill kernel with the same output
   buffer size. This synthetic kernel does not evaluate the generated model; it
   only estimates the cost of writing packet-shaped data on the GPU.
7. measures a hand-transcribed CUDA exact packet assembly kernel for `CarModel`
   and compares its packet output against the generated CPU `CarModel`.

The benchmark reports host eval+pack time, H2D copy time, persistent
pack-into-pinned plus async-H2D staging time, and a synthetic device packet-fill
lower bound. The `CarModel` exact CUDA packet kernel is included as a
single-model device-codegen prototype, not as a generic MiniModel CUDA backend.
The benchmark intentionally does not include a GPU Riccati solve, host/device
round trip, solver residuals, line search, SOC, restoration, or postsolve.

## Reproduction

```bash
cmake -S . -B .build_cuda_bench \
  -DMINISOLVER_BUILD_CUDA_BENCHMARKS=ON \
  -DMINISOLVER_BUILD_TESTS=OFF \
  -DMINISOLVER_BUILD_EXAMPLES=OFF \
  -DMINISOLVER_BUILD_TOOLS=OFF \
  -DMINISOLVER_FETCH_DEPS=OFF \
  -DMINISOLVER_CUDA_ARCHITECTURES=native

cmake --build .build_cuda_bench --target cuda_generated_packet_upload_bench -j
.build_cuda_bench/cuda_generated_packet_upload_bench
```

Run environment:

```text
GPU: NVIDIA GeForce RTX 5080
CUDA compiler: NVIDIA 13.1.115
CUDA architecture: native, resolved to sm_120
```

## 2026-05-18 Result Summary

Key timing observations:

| Model | NX | NU | NC | Horizon N | Batch | Packet entries | MiB | Host eval+pack us | Pinned H2D us | Pinned H2D GB/s | Persistent pinned eval+H2D us | GPU synthetic fill us | GPU synthetic fill GB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CarModel | 4 | 2 | 5 | 50 | 1 | 112 | 0.04 | 21.63 | 31.05 | 1.44 | 37.22 | 6.37 | 7.04 |
| CarModel | 4 | 2 | 5 | 50 | 256 | 112 | 10.94 | 1467.59 | 374.65 | 30.61 | 1961.75 | 17.36 | 660.71 |
| CarModel | 4 | 2 | 5 | 50 | 4096 | 112 | 175.00 | 24481.22 | 3622.91 | 50.65 | 28639.60 | 213.40 | 859.91 |
| CarModel | 4 | 2 | 5 | 100 | 1024 | 112 | 87.50 | 12402.77 | 1836.98 | 49.95 | 14504.53 | 108.42 | 846.26 |
| BicycleExtModel | 6 | 2 | 10 | 50 | 1 | 234 | 0.09 | 13.71 | 31.44 | 2.98 | 43.33 | 8.36 | 11.20 |
| BicycleExtModel | 6 | 2 | 10 | 50 | 256 | 234 | 22.85 | 3416.01 | 607.08 | 39.47 | 3538.26 | 31.53 | 759.86 |
| BicycleExtModel | 6 | 2 | 10 | 50 | 4096 | 234 | 365.62 | 59202.46 | 7545.96 | 50.81 | 66118.97 | 439.60 | 872.12 |
| BicycleExtModel | 6 | 2 | 10 | 100 | 1024 | 234 | 182.81 | 29898.76 | 3829.97 | 50.05 | 32939.98 | 219.93 | 871.62 |

For the largest contiguous buffers, pinned H2D bandwidth approaches
`50 GB/s` on this machine. Small buffers are dominated by fixed overhead.
Packing directly into persistent pinned memory avoids the extra pageable-to-
pinned staging copy, but the per-frame cost is still approximately host
generated-model evaluation plus H2D transfer. It does not remove the integration
cost unless generated packet assembly is fused onto the device or overlapped
with other independent work.

The synthetic device-fill lower bound is much faster than the CPU eval+pack
plus H2D path for large batches, reaching roughly `850 GB/s` effective packet
write bandwidth. This does not prove generated-model evaluation is cheap on the
GPU, but it shows that packet-shaped device writes are not the blocking cost if
the generated model evaluation can be moved or fused onto the device.

The hand-transcribed `CarModel` exact CUDA packet assembly benchmark produced
packets matching the generated CPU `CarModel` to roughly `1e-14` max error:

| Horizon N | Batch | MiB | CPU eval+pack us | CUDA exact packet us | CUDA exact GB/s | Max packet error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 1 | 0.04 | 13.20 | 17.13 | 2.62 | 8.88e-16 |
| 50 | 256 | 10.94 | 1475.97 | 50.28 | 228.11 | 3.55e-15 |
| 50 | 4096 | 175.00 | 25239.77 | 1488.33 | 123.29 | 3.55e-15 |
| 100 | 1024 | 87.50 | 12616.19 | 263.52 | 348.17 | 7.11e-15 |

This is the first positive signal for device-side generated-model packet
assembly: if generated model code can be emitted for CUDA and run over enough
independent horizons, packet assembly can be substantially faster than the
current CPU eval+pack path. It is still a single hand-transcribed model kernel,
not a general MiniModel CUDA codegen path.

## Backend Implication

The device-resident batched Riccati benchmarks show useful GPU speedups for
large batches. This benchmark shows the integration cost that sits before those
kernels if model evaluation and packet assembly stay on the CPU.

For large batched generated-model packets, H2D upload alone is often measured in
milliseconds, and host eval+pack is also nontrivial. Therefore a future GPU
backend should not be designed as:

```text
CPU evaluate all stages -> upload packets -> GPU Riccati -> download result
```

unless the workload is sufficiently batched and the transfer cost is amortized.
The better route is to fuse or batch more of the pipeline:

- batched generated model evaluation;
- persistent device-resident packet buffers;
- generated packet assembly directly on device, ideally emitted by MiniModel;
- persistent pinned or asynchronous staging;
- GPU Riccati only for workloads with enough independent horizons;
- differentiable or sampled-control workloads where many solves share a batch.

This reinforces the current triage conclusion: keep `Backend::GPU_MPX` and
`Backend::GPU_PCR` unsupported for the normal single-horizon solver path until
the full packet assembly and transfer story is measured in an end-to-end
prototype.
