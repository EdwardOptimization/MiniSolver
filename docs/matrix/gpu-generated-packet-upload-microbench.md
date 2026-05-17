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
6. measures a lower-bound device-side packet fill kernel with the same output
   buffer size. This synthetic kernel does not evaluate the generated model; it
   only estimates the cost of writing packet-shaped data on the GPU.

The benchmark reports host eval+pack time, H2D copy time, persistent
pack-into-pinned plus async-H2D staging time, and a synthetic device packet-fill
lower bound. It intentionally does not include a GPU Riccati solve,
host/device round trip, solver residuals, line search, SOC, restoration, or
postsolve.

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
| CarModel | 4 | 2 | 5 | 50 | 1 | 112 | 0.04 | 21.17 | 31.79 | 1.41 | 36.07 | 5.87 | 7.64 |
| CarModel | 4 | 2 | 5 | 50 | 256 | 112 | 10.94 | 1461.01 | 400.09 | 28.67 | 1962.77 | 17.01 | 674.19 |
| CarModel | 4 | 2 | 5 | 50 | 4096 | 112 | 175.00 | 24606.66 | 3642.01 | 50.38 | 28768.58 | 212.87 | 862.03 |
| CarModel | 4 | 2 | 5 | 100 | 1024 | 112 | 87.50 | 12625.78 | 1836.50 | 49.96 | 14636.71 | 108.20 | 847.98 |
| BicycleExtModel | 6 | 2 | 10 | 50 | 1 | 234 | 0.09 | 13.52 | 31.54 | 2.97 | 43.72 | 6.39 | 14.64 |
| BicycleExtModel | 6 | 2 | 10 | 50 | 256 | 234 | 22.85 | 3197.37 | 586.83 | 40.83 | 3466.89 | 31.15 | 769.30 |
| BicycleExtModel | 6 | 2 | 10 | 50 | 4096 | 234 | 365.62 | 62415.40 | 7575.93 | 50.61 | 70324.94 | 438.28 | 874.75 |
| BicycleExtModel | 6 | 2 | 10 | 100 | 1024 | 234 | 182.81 | 31769.19 | 3808.49 | 50.33 | 35247.50 | 220.31 | 870.12 |

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
- generated packet assembly directly on device;
- persistent pinned or asynchronous staging;
- GPU Riccati only for workloads with enough independent horizons;
- differentiable or sampled-control workloads where many solves share a batch.

This reinforces the current triage conclusion: keep `Backend::GPU_MPX` and
`Backend::GPU_PCR` unsupported for the normal single-horizon solver path until
the full packet assembly and transfer story is measured in an end-to-end
prototype.
