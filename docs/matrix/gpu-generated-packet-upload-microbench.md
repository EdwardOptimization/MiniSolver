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

The benchmark reports host eval+pack time and H2D copy time. It intentionally
does not include a GPU Riccati solve, host/device round trip, solver residuals,
line search, SOC, restoration, or postsolve.

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

| Model | NX | NU | NC | Horizon N | Batch | Packet entries | MiB | Host eval+pack us | Pinned H2D us | Pinned H2D GB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CarModel | 4 | 2 | 5 | 50 | 1 | 112 | 0.04 | 22.71 | 31.67 | 1.41 |
| CarModel | 4 | 2 | 5 | 50 | 256 | 112 | 10.94 | 1542.70 | 372.53 | 30.79 |
| CarModel | 4 | 2 | 5 | 50 | 4096 | 112 | 175.00 | 25772.15 | 3638.23 | 50.44 |
| CarModel | 4 | 2 | 5 | 100 | 1024 | 112 | 87.50 | 13278.92 | 1842.37 | 49.80 |
| BicycleExtModel | 6 | 2 | 10 | 50 | 1 | 234 | 0.09 | 17.05 | 31.82 | 2.94 |
| BicycleExtModel | 6 | 2 | 10 | 50 | 256 | 234 | 22.85 | 3655.08 | 610.29 | 39.26 |
| BicycleExtModel | 6 | 2 | 10 | 50 | 4096 | 234 | 365.62 | 62769.02 | 7564.75 | 50.68 |
| BicycleExtModel | 6 | 2 | 10 | 100 | 1024 | 234 | 182.81 | 31638.55 | 3819.32 | 50.19 |

For the largest contiguous buffers, pinned H2D bandwidth approaches
`50 GB/s` on this machine. Small buffers are dominated by fixed overhead.

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
- pinned or asynchronous staging;
- GPU Riccati only for workloads with enough independent horizons;
- differentiable or sampled-control workloads where many solves share a batch.

This reinforces the current triage conclusion: keep `Backend::GPU_MPX` and
`Backend::GPU_PCR` unsupported for the normal single-horizon solver path until
the full packet assembly and transfer story is measured in an end-to-end
prototype.
