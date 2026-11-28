# Task Contract: In-place Metal gated residual on current main

- Task name `adaln-metal-gate-v1`, issue #9.
- Baseline `a82f0cf907648e9dabf1457cf35efbbfccd28f4d`, M5 Max 36 GiB, Release, W4, max-quality, zero DiT cache with default prefetch, warm modulation cache, batch-25 VAE, six PDD steps, teapot seed 42. Compare default and candidates in one build with an explicit experimental switch. The recent stock 4K anchor on `d2ec390` was 178.756 s DiT and 196.779 s end to end (issue #11). The older 4096 prototype measured 6.4780 vs 1.4916 ms per gate and 217.08 vs 210.27 s for one fixed-input DiT pair (`8e22daf`, `benchmarks/kernel-experiments/onyx-adaln-metal-v1.md`). Those numbers use different baselines.
- Inputs are contiguous BF16 residual and update matrices [tokens, 5376], a BF16 [12, 5376] or [18, 5376] modulation-table view with a possibly nonzero aligned offset, one int32 modality per token, and gate slot 2 or 5. The mutated residual belongs to the DiT block. Preserve the table and update. Read modality 0 for target images, 1 for text, and 2 for reference images.
- Output is the original residual buffer containing BF16(x + BF16(update * gate)), with explicit BF16 ties-to-even at both steps. Require bitwise equality with the current MPSGraph gate for both slots, mixed modality indices, singleton and ragged row counts, and an aligned nonzero table offset. Require byte-identical fixed-input DiT latents and native PNGs at 1024 and 4096. Preserve the full CTest and Python suites.
- Keep RMSNorm graph math, Q4 packing, profiles, text, VAE and sampling unchanged. Start with one Metal dispatch per gate, with zero additional per-token buffers. Keep the explicit default/experimental comparison. If a candidate removes unused materialized gates, preserve all normalization parts byte for byte. Multiple-reference inputs must respect actual per-token modality indices.
- Validation uses a Release build, full CTest, pytest and schema compatibility. Evaluate direct runtime calls at 1093 and 16453 rows on synthetic deterministic BF16 data. Warm and rotate three ABBA/BAAB/ABBA timing blocks, comparing paired medians. Profile and time in separate runs. Verify fixed-input DiT and PNG SHA-256 through the real W4 runtime at both sizes. Record DiT and wall time, lifetime physical footprint, process-tree RSS, swap and contention. Use clean interleaved 4K runs. If an arm drifts over 5%, repeat the comparison.
- Promotion requires a strict, repeatable gate latency improvement for the integration, with later variants judged against the incumbent path they change. The final candidate must also reduce 4K DiT latency in a clean interleaved test, pass correctness and full tests, and avoid a 1024 regression above the observed noise. Preserve unmodified defaults if the final candidate is not promoted.

# Draft

There are 600 gated residual calls over 50 blocks and six steps. At 16453 rows each MPSGraph residual output covers 176902656 bytes. This allocation size describes one operation, and the peak depends on overlapping live buffers. On current main the ordinary path materializes all six parts with one GPU gather, then uses parts 2 and 5 in the two MPSGraph gates. The optional reference-indexed path already keeps modulation compact. The historical prototype inferred modalities from a text prefix, which cannot represent reference tokens on current main.

Start with the historical four-BF16 vector layout adapted to current per-token int32 indices and table views. Then test omitting the two expanded gate outputs from the ordinary gather with unchanged normalization, and test kernel scheduling or memory traffic against the fastest verified path. The four-, eight- and sixteen-value thread layouts were close in the historical isolated benchmark. Prefer removing expanded-buffer traffic before revisiting width. Stop after at most eight episodes or two consecutive episodes without a strict improvement.

# E1. In-place BF16 residual with token modality indices (2026-11-26)

Promoted for further evaluation. Current-main Metal uses the real per-token modality index and preserves the BF16 multiplication and addition boundaries. At 16453 rows three warmed blocks measured 5.8333 ms for MPSGraph and 1.4900 ms for Metal, with less than 5% block-median spread in both arms. The first real 1024 replay exposed the two-modality table layout. The corrected path accepts both 12- and 18-row tables and the two layouts have separate CTest coverage. The corrected 1024 default and Metal arms matched both the latent and PNG hashes from clean main. The profiled run attributed 572.78 ms to 600 MPSGraph gates and 160.27 ms to the 600 Metal gates. Profile timing is diagnostic.

# E2. Gather only the four normalization tensors (2026-11-26)

Promoted for further evaluation. With the in-place gate reading the compact table, the two expanded gates are dead outputs. The existing graph can omit them and preserve exact bytes for all four norm inputs. At 16453 rows the six- and four-output gather medians were 24.1094 ms and 16.6991 ms with less than 5% block-median spread. Each omitted output is 176902656 bytes at this shape. The 1024 profiled launcher replay matched the E1 and clean-main output hashes. CTest passed 23/23, including the offset table and mixed-tag tests. The first 1093-row gather batch had over 5% spread and was excluded from speed claims.

# E3. Reuse four expanded inputs through a native gather (2026-11-26)

Promoted for further evaluation. One Metal dispatch copies the exact BF16 payloads into the four buffers reused by the per-step normalization calls. Refill from a second table verifies that reuse updates all four parts. At 16453 rows, the four-output graph and the native gather measured 15.7470 ms and 2.0711 ms. At 1093 rows they measured 1.1026 ms and 0.2489 ms. All block-median spreads were below 5%. The 33942-row reference shape, singleton and ragged mixed modalities passed. CTest passed 23/23. The real 1024 replay matched clean-main latent and PNG hashes. The 1024 profile attributed 502.68 ms to the four-output MPSGraph gather and 72.52 ms to the native gather, across 300 calls.

The remaining expanded norm outputs are large even when cheap to refill. E4 tests the already-shipped indexed MPSGraph RMSNorm composition on the ordinary path with the same Metal gate. The normalization primitives, epsilon, and BF16 acceptance gate remain the same.

# E4. Indexed RMSNorm with the in-place Metal gate (2026-11-26)

Promoted for final replay. The existing indexed norm queries the BF16 shift and scale in its MPSGraph and uses the same normalization primitives and epsilon. The Metal gate reads the compact table directly. This eliminates the standalone gather and its six dense outputs, including the native gather in E3. At 16453 rows the two norm calls and their input selection measured 13.2528 ms with the native gather and 11.9754 ms indexed. At 1093 rows the re-warmed ABBA measured 1.2426 ms and 1.1201 ms. Both accepted batches had under 5% spread. Both norm outputs and both gate outputs matched byte for byte. The real 1024 latent and PNG were identical to clean main. Diagnostic 1024 DiT times were 13.440 s default, 12.457 s E3, and 12.455 s E4. The final patch retains only the indexed path and the original explicit-rounding Metal gate.

# E5. Gate threadgroup geometry (2026-11-26)

Rejected. The six alternative threadgroup shapes were bitwise correct. The initially promising 256x1 and 64x2 groups did not reproduce a strict 4K improvement on the second comparison. The 4K repeat medians were 2.0602 to 2.0717 ms for 256x1 and 2.0970 to 2.0959 ms for 64x2, with mixed paired results. Small-shape tuning batches exceeded the 5% drift limit. The incumbent stays at 128x1.

# E6. Direct BF16 vector arithmetic (2026-11-26)

Rejected. Replacing explicit FP32 arithmetic and BF16 conversion points with BF16 vector operators, with contraction disabled, diverged after repeated updates at both workload sizes. The first mismatch was cell 3 (expected 0x4209, received 0x420a). No timing from the failing path qualifies. The explicit BF16 conversion after the product and after the sum is retained. E5 and E6 complete the two-episode stall rule.

# Final promotion gate (2026-11-26)

Promoted as an opt-in candidate through `ONYX_DIT_METAL_GATE=1`. The final same-binary build on `a82f0cf` passed three clean interleaved pairs per resolution.

- 4096×4096 measured 181.39 → 171.67 s DiT (5.4%), 199.44 → 189.61 s wall (4.9%), 12.31 → 11.32 GiB DiT peak physical footprint.
- 1024×1024 measured 13.23 → 12.39 s DiT (6.4%), 16.80 → 15.96 s wall (5.0%), 6.43 → 6.37 GiB DiT peak physical footprint.
- The three clean 4096×4096 DiT pairs saved 9.84 s, 10.04 s, 9.72 s. The pairs with overlapping native stages were excluded and rerun. Every observed latent and PNG matched at its resolution.
- A native 1024×1024 edit against the existing indexed path matched both output hashes with one real reference image. The 33942-row mixed-modality probe also matched.
- CTest passed 23/23, pytest passed 78/78, and schema v1 compatibility passed.

# Reproduce

Run the gate probe on the built native engine.

```sh
.build/onyx-dit-gate-probe 1093
.build/onyx-dit-gate-probe 16453
.build/onyx-dit-gate-probe 33942 --verify-only --references
```

Compare both modes of the same Release binary with the native benchmark runner. Repeat with `teapot-1024.json` for the smaller case.

```sh
uv run python benchmarks/ab_compare.py --label adaln-gate-4096 \
  --case benchmarks/requests/teapot-4096.json --rounds 3 --warmup 1 \
  --arm 'base .build/onyx-native' --arm 'metal .build/onyx-native' \
  --env base=ONYX_DIT_METAL_GATE=0 --env metal=ONYX_DIT_METAL_GATE=1
```

[Measured runs and hashes](../benchmarks/results/m5-max-36gb-adaln-metal-gate-v1.json)

## Metal compiler compatibility (2026-11-27)

The macOS 26 ARM64 runner's MPSGraph output matched a once-rounded product-plus-residual at both failing probe cells. An independent CPU check found that the original Metal kernel also produced the once-rounded bytes locally, where it matched the incumbent across the campaign. The original Metal source expressed a BF16 intermediate, so the compiled behavior differed across the two environments. The final kernel now expresses FP32 fused multiply-add followed by BF16 output conversion. The probe checks both gate slots against the independent CPU FMA reference and both MPSGraph paths on every runner. This check supersedes the earlier interpretation of the two conversion points in the Metal source.
