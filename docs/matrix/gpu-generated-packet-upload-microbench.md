# CUDA Generated Packet Upload Microbenchmark

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

The benchmark reports host eval+pack time, H2D copy time, and persistent
pack-into-pinned plus async-H2D staging time. It intentionally does not include a
GPU Riccati solve, host/device round trip, solver residuals, line search, SOC,
restoration, or postsolve.

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

| Model | NX | NU | NC | Horizon N | Batch | Packet entries | MiB | Host eval+pack us | Pinned H2D us | Pinned H2D GB/s | Persistent pinned eval+H2D us |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CarModel | 4 | 2 | 5 | 50 | 1 | 112 | 0.04 | 22.97 | 31.10 | 1.44 | 37.55 |
| CarModel | 4 | 2 | 5 | 50 | 256 | 112 | 10.94 | 1503.40 | 335.65 | 34.17 | 1931.62 |
| CarModel | 4 | 2 | 5 | 50 | 4096 | 112 | 175.00 | 24393.33 | 3632.93 | 50.51 | 28814.13 |
| CarModel | 4 | 2 | 5 | 100 | 1024 | 112 | 87.50 | 12508.82 | 1848.59 | 49.63 | 14501.32 |
| BicycleExtModel | 6 | 2 | 10 | 50 | 1 | 234 | 0.09 | 13.39 | 31.64 | 2.96 | 42.85 |
| BicycleExtModel | 6 | 2 | 10 | 50 | 256 | 234 | 22.85 | 3440.65 | 581.26 | 41.22 | 3499.71 |
| BicycleExtModel | 6 | 2 | 10 | 50 | 4096 | 234 | 365.62 | 55910.40 | 7576.84 | 50.60 | 65058.73 |
| BicycleExtModel | 6 | 2 | 10 | 100 | 1024 | 234 | 182.81 | 28318.13 | 3816.41 | 50.23 | 32171.65 |

For the largest contiguous buffers, pinned H2D bandwidth approaches
`50 GB/s` on this machine. Small buffers are dominated by fixed overhead.
Packing directly into persistent pinned memory avoids the extra pageable-to-
pinned staging copy, but the per-frame cost is still approximately host
generated-model evaluation plus H2D transfer. It does not remove the integration
cost unless generated packet assembly is fused onto the device or overlapped
with other independent work.

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
- persistent pinned or asynchronous staging;
- GPU Riccati only for workloads with enough independent horizons;
- differentiable or sampled-control workloads where many solves share a batch.

This reinforces the current triage conclusion: keep `Backend::GPU_MPX` and
`Backend::GPU_PCR` unsupported for the normal single-horizon solver path until
the full packet assembly and transfer story is measured in an end-to-end
prototype.
