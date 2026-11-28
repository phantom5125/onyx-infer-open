# Native inference benchmarks

The measurements below distinguish operator probes from measurements of the full H3 pipeline. Benchmark records include the hardware, run conditions, and output hashes. [Kernel experiment archive](../benchmarks/kernel-experiments/README.md)

## Single-image performance

The teapot benchmarks compared builds `35bc6f9` and `e0b4244` on an M5 Max with 36 GB of unified memory. Each build used its own launcher with the same W4 package, six-step PDD schedule, seed 42, and `max-quality` profile. Measurements alternated between builds after one warmup per build and resolution. The `e0b4244` measurements hit the DiT modulation cache.

| Resolution | Runs per build | Wall time, `35bc6f9` → `e0b4244` | Peak process-tree RSS | Record |
| --- | ---: | ---: | ---: | --- |
| 1024×1024 | 3 | 33.54 → 19.58 s (median) | 1.80 → 1.71 GiB | [1024×1024](../benchmarks/results/campaign-20261123/against-main-1024.json) |
| 4096×4096 | 1 | 323.04 → 227.93 s | 2.66 → 1.91 GiB | [4096×4096](../benchmarks/results/campaign-20261123/against-main-4096.json) |

The newer build overlaps more weight reads with GPU work, gathers AdaLN modulation on the GPU, batches VAE tiles, and caches DiT modulation tables by default. At 4096×4096 the VAE stage took 99.64 s before batching and 28.95 s with batches of 25 tiles. The batched decoder changed 36 of 3,145,728 RGB channel values at 1024×1024 and 653 of 50,331,648 at 4096×4096, each by one 8-bit level. [Generated-image evidence](../benchmarks/results/campaign-20261123/generated-image-evidence.json)

The optional `m5-fast` profile ran in 20.98 s versus 22.71 s for `max-quality` in a separate three-run comparison on the same 1024×1024 request. Peak RSS increased from 1.67 to 3.31 GiB and output image correlation was 0.987. Speed depends on the workload. [Q4 kernel comparison](../benchmarks/results/campaign-20261123/round4-nax-q4-kernel.json)

The [experiment record](../benchmarks/kernel-experiments/local-single-image-20261123.md) documents individual changes and rejected variants. The [benchmark results](../benchmarks/results/campaign-20261123/) include the interleaved samples, comparisons with the earlier streaming baseline, and lettering cases.

## Production and isolated measurements

| Configuration | DiT stage | End-to-end per image | Observation |
| --- | ---: | ---: | --- |
| MPSGraph, streamed weights | 26.6 s | 36.8 s | Production default |
| MPSGraph with LRU | 26.6 s | — | Streaming already overlaps I/O |
| BF16 SIMD with LRU | 37.3 s | About 50 s | Experimental path |
| SIMD, streamed weights | About 41 s | — | Experimental path |

One isolated case measured 26.5 ms for tile16, 15.7 ms for tile32, 5.01 ms for SIMD (11.8 TFLOPS), and 5.26 ms for Steel. Across the measured DiT Q4 shapes, the SIMD candidate was about 5.3× faster than tile16 and 4% faster than Steel. Cached production MPSGraph fuses dequantization and matmul and uses shape-specialized graphs.

### Release 0.3.0 retest

One matching 1024×1024 teapot request per profile used the same W4 pack and seed 42 on M5 Max with 36 GB. The default profile took 44.1 s overall and 33.8 s in the DiT stage. The BF16 SIMD profile took 52.1 s overall and 41.8 s in the DiT stage. The default PNG matched the earlier result byte for byte. SSIM between profiles was 0.9580.

### Reuse DiT timestep modulation tables

The six PDD timesteps and the AdaLN projections depend on the model and schedule. Native DiT computes the 50 block tables and the final-layer table before denoising. With `--modulation-cache <directory>`, it can store their BF16 bytes in a 55.5 MiB entry and reuse them across requests. The cache binds the model file identity, schedule, native build, GPU, and macOS version. Cache writes are atomic and entries carry a payload checksum.

On the 36 GiB M5 Max, the fixed 1024×1024 teapot request gave these medians in three rotated blocks after warmup. Cold entries started in empty cache directories while the OS file cache remained warm.

| Modulation cache | End to end | Native DiT | DiT weight bytes read |
| --- | ---: | ---: | ---: |
| Off | 43.14 s | 32.93 s | 85,656,876,608 |
| First use | 42.96 s | 32.78 s | 85,656,876,608 |
| Hit | 38.21 s | 28.11 s | 59,514,797,120 |

All nine measured requests and both warmups produced the same PNG SHA-256. Peak sampled process-tree RSS was 1.78–1.80 GiB across the three measured configurations. System swap usage was unchanged at sample boundaries.

For a 512×768 native stage check, the uncached, cached, and rebuilt latents had the same byte hash. A payload bit flip forced a rebuild and the following request hit the repaired entry. A 4096×4096 request reused the 1024×1024 modulation entry and matched the PNG from the current uncached baseline byte for byte. [Samples and build identities](../benchmarks/results/m5-max-36gb-dit-modulation-cache-v1.json)

### Cache the repeated DiT weights

These measurements use serial DiT weight reads. The 10 GiB budget (`--dit-cache-mib 10240`) keeps repeatedly read packed weights in memory across sampling steps. The DiT stage reports the applied budget and observed reads.

With the same 1024×1024 request, same native binary, and a warm modulation table, the median of three paired end-to-end runs fell from **30.69 s to 24.96 s**. Native DiT fell from 20.11 s to 16.76 s. The cached runs recorded 2,015 weight hits and read 11,329,849,920 bytes per run, compared with 59,514,797,120 bytes with weight caching off. Peak sampled process-tree RSS rose from 1.79 GiB to 7.09 GiB. All PNGs matched byte for byte and system swap usage was unchanged at sample boundaries.

The 4096×4096 stress run produced the same PNG hash with this budget. It increased system swap by about 853 MiB and took 318.46 s overall, including 105.26 s in VAE. Keep the default weight budget for 16 MP on this machine. [Paired measurements and stress run](../benchmarks/results/m5-max-36gb-dit-weight-residency-v1.json)

## DiT weight prefetch on M5 Max

Profiling the `f8f73f8` streaming baseline on the 36 GiB M5 Max attributed the 33.14 s DiT stage to 14.20 s of weight reads and 12.98 s of Q4 linear operations. The six steps read 85,656,876,608 bytes. Modulated RMSNorm, fused attention, gated activation, and gated residuals accounted for another 3.20 s. The existing one-block prefetch was opt-in and restarted at each sampling-step boundary.

The candidate enables prefetch by default with a zero weight-cache budget and keeps the one-block pipeline active across step boundaries. Three samples per mode used the same binary and request in off/on/on/off/off/on order, with profiling disabled.

| Prefetch | End-to-end samples | Median total | Median DiT | Peak process-tree RSS |
| --- | --- | ---: | ---: | ---: |
| Disabled | 42.34 / 42.24 / 42.16 s | 42.24 s | 32.00 s | 1.794 GiB |
| Enabled | 32.81 / 32.81 / 33.11 s | **32.81 s** | **22.95 s** | 1.794 GiB |

The median fell 22.3% end to end and 28.3% in DiT. All six PNGs had SHA-256 `4160adb7ac0aa777d6cfe8bafaa9284dd0202198a3ea289c377edf8e93b577d1`. Prefetch waits were 17–23 ms per request. A paired 4096×4096 run with the same binary also produced byte-identical PNGs (SHA-256 `0b9761b2416b57639d9fe387a528136c072b867fa54f394399a651be36557e3f`); those single runs are correctness and memory evidence, not a 4K performance claim. Use `--no-dit-prefetch` to isolate serial DiT reads. Use `--no-prefetch-weights` for serial reads in both the text and DiT stages. [Samples, binary hash, and reproduction commands](../benchmarks/results/m5-max-36gb-dit-prefetch-default-v1.json)

## DiT weight prefetch on MacBook Air M4

The 1024×1024 teapot retest used an M4 with 10 GPU cores, 24 GiB of unified memory, macOS 26.2, and battery power. Each configuration had one warmup and three measured runs in rotating order. All runs used the same W4 pack, `max-quality` profile, seed 42, and six PDD evaluations.

| Native binary and setting | End-to-end samples | Median total | Median DiT | Peak process-tree RSS |
| --- | --- | ---: | ---: | ---: |
| 0.3.1 release | 174.98 / 192.19 / 173.26 s | 174.98 s | 143.48 s | 1.79 GiB |
| Prefetch branch, disabled | 179.54 / 196.47 / 171.72 s | 179.54 s | 148.80 s | 1.79 GiB |
| Prefetch branch, enabled | 177.23 / 184.61 / 159.54 s | 177.23 s | 145.17 s | 1.79 GiB |

Enabling prefetch lowered the median by 1.3% on the same binary. The median with prefetch disabled was 2.6% higher than the release. The sample ranges overlap across all three configurations, so this run cannot establish either a repeatable speedup or a default-path regression. An earlier two-sample comparison per mode on AC power measured a 6.5% lower median with prefetch. The two comparisons used different power conditions and sample counts. The separate M5 Max comparison above showed a clear improvement on that device.

All teapot PNG hashes matched. Earlier paired Moka pot and jet engine outputs also matched byte for byte. The Moka pot title remained legible, and the dense Chinese labels retained the same spelling errors in both outputs. The prefetch path reorders weight reads for one block ahead and uses the same numerical path. The memory column reports sampled process RSS. Metal allocations are recorded separately at DiT step boundaries. [Per-run timings, hashes, and build details](../benchmarks/results/m4-air-24gb-dit-prefetch-v1.json)

To measure prefetch with the local W4 package, use the following command. Pass `--no-dit-prefetch` for the control, and alternate the modes for repeated measurements.

```bash
uv run python benchmarks/run_native.py --case benchmarks/requests/teapot-1024.json \
  --label m4-prefetch --target config/targets/m4-air-24gb.json --dit-prefetch
```

## Correctness and profiling

MLX reference latents correlated 0.9962 with the native MPSGraph path. MPSGraph and native BF16 SIMD latents correlated 0.9973. Image SSIM against the MLX reference was 0.9812 for native MPSGraph and 0.9553 for native BF16 SIMD. The streamed MLX reference took 225.4 s for this comparison and native MPSGraph took 36.8 s.

- A probe initially interpreted BF16 output as FP16. Correcting the readback recovered agreement with CPU reference, including a Steel runtime check on the `m=69` edge tile.
- On the tested Metal 4 driver, a third encoder that consumed a buffer written by the immediately preceding encoder was silently skipped. A separate output conversion command buffer worked around the issue. Direct BF16 removed the conversion passes.
- FP16 residuals exceeded 65,504 near block 14 and produced NaNs. BF16 kept the range needed by the VAE.
- The isolated MPSGraph result included graph compilation. Production reused compiled graphs and measured roughly 0.4–2 ms for QKV calls that the probe reported at 106 ms.
- An apparent 377 non-finite baseline latent values came from reading FP32 storage as `uint16`. The FP32 values were finite.

`ONYX_SIMD_SKIP_ODD` alternated paths within one run. The measured average was 0.347 s per block on MPSGraph and 0.825 s on SIMD, a 2.38× gap. Cross-check equations agreed within 0.8 s. Thermal drift was +2.6% and +6.8% respectively, leaving a gap in in-context efficiency. Legacy GPU stage counters asserted on the reference device, its runtime did not expose the MTL4 counter-heap selector, and Instruments tracing required Xcode.


## Indexed RMSNorm for text-only AdaLN

Set `ONYX_DIT_INDEXED_NORM=1` for the experimental pure-text path. It indexes shift and scale inside the RMSNorm graphs, then gathers only the two gate tensors for the original MPSGraph residual operations. The native DiT JSON reports the effective `norm_mode` and `gate_mode`.

Three interleaved teapot pairs per resolution used the H3 W4 pack, M5 Max 36 GiB, six PDD steps, seed 42, `max-quality`, warm modulation cache, default prefetch, and VAE batch 25. Both resolutions matched latent and PNG hashes.

| Resolution | Native DiT | End to end | DiT peak physical footprint | DiT max RSS |
| --- | ---: | ---: | ---: | ---: |
| 1024×1024 | 14.39 → 14.31 s | 18.12 → 18.03 s | 6.44 → 6.40 GiB | 1.70 → 1.70 GiB |
| 4096×4096 | 194.52 → 196.21 s | 213.35 → 215.31 s | 12.31 → 11.67 GiB | 1.69 → 1.72 GiB |

The 4K DiT peak physical footprint fell in each pair by 0.43–0.70 GiB. The direction of 4K DiT timing varied across pairs. The observed physical footprint reduction determines the promotion. [Paired samples and verification](../benchmarks/results/m5-max-36gb-dit-indexed-rmsnorm-v1.json)

## Rejected experiments

| Experiment | Result |
| --- | --- |
| BK=64 tile | About 10% slower. Extra scalar writes and threadgroup storage outweighed fewer barriers. |
| Scaled FP16 input and output | Correct and 3.4 s slower. Conversion passes and allocations outweighed the FP16 MMA benefit. |
| LRU for MPSGraph | No measured improvement in the compute-bound path. |
| Buffer allocation cache | The 672 measured allocations consumed 17 ms, below the dominant cost. |

## Reproduce

The native sources include `q4_gemm.metal`, `q4_gemm_simd.metal`, and `q4_gemm_steel.metal`. The Steel and cache probes exercise real packed weights. `tools/profile_quality_gate.py` generates candidate images and reports SSIM.

```bash
python -m onyx quantize --source artifacts/models/h3-image-source --component transformer --all-linear --output "$output/transformer.onxpack"
python -m onyx quantize --source artifacts/models/h3-image-source --component text_encoder --all-linear --output "$output/text_encoder.onxpack"
python -m onyx package-w4 --source artifacts/models/h3-image-source --original-manifest artifacts/models/h3-w4-meta/manifest.pb --output "$output"

python3 tools/profile_quality_gate.py --candidate speed-focus
```

Diagnostic environment variables include `ONYX_SIMD_SKIP_ODD=1` for mixed-path measurements, `ONYX_SIMD_NOWAIT=1` for deferred waits, `ONYX_GPU_COUNTERS=1` for counter probes on eligible devices, `ONYX_PREFETCH_THREADS=<n>` for the number of background readers, `ONYX_PREFETCH_TRACE=1` to list each weight that a stage still reads synchronously, and `ONYX_VAE_BATCH=<n>` for the number of VAE tiles per decoder batch, where 1 decodes one tile at a time.
