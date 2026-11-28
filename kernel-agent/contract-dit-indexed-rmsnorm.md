# Task Contract: Text-only indexed DiT RMSNorm

- Task `dit-indexed-rmsnorm-v1`, issue #14. Pin `a82f0cf`, M5 Max 36 GiB, Release, H3 W4, `max-quality`, six PDD steps, teapot seed 42, default prefetch, zero DiT cache, and a warm modulation cache. The #9 campaign measured 13.23 s at 1024² and 181.39 s at 4096² for stock DiT at this revision. Its candidate also changed the gates. This experiment measures the normalization path while retaining the gathered MPSGraph gates.
- Inputs are BF16 activations `[tokens, 5376]`, existing norm weights, a BF16 table view `[12, 5376]` with a timestep offset, and one int32 modality per token (target image 0, text 1). Outputs preserve the existing BF16 normalization, modulation, and gated residual operation order.
- In one GraphSession compare steps 0, 1, and 0 against independently allocated contiguous tables, varying the modality split while holding the token count fixed. Check exact BF16 bytes for both norms and both gathered gates. Verify the effect of the nonzero offset. Compare the gathered gate outputs bitwise and retain the original gate math.
- Compare the six-output gather against a distinct graph emitting only slots 2 and 5, with the same downstream gates. At 4K the removed outputs account for 672 MiB for image tokens alone. Measure peak physical footprint independently of output allocation size.
- Run full CTest and the contract suite. Measure operator latency in three warmed ABBA/BAAB/ABBA blocks at 1093 and 16453 rows. Verify effective norm and gate modes, including the reference flag on a pure-text request as a negative control.
- Run three interleaved, warmed 1024² and 4096² native pairs, at batch one image per run with identical VAE settings within each pair. Require identical latent and PNG SHA-256, and record DiT latency, whole-request time, DiT peak physical footprint, and RSS. Trace allocations in a separate run if the peak is flat.
- Retain the experimental path only if the probe and end-to-end correctness pass and 4K latency or footprint improves repeatably without a 1024² regression. Leave production defaults for the maintainer’s review.

## E1. Two-output gate gather with indexed RMSNorm (2026-11-27)

Promoted as an opt-in candidate. Both RMSNorm outputs, gathered gate tensors, and both residual outputs matched the independently allocated contiguous tables after step 0 → 1 → 0 with changing text/image splits. The existing reference-edit switch left both modes gathered in the pure-text negative control. In the main comparison the candidate reported indexed norms and gathered gates. The native latent and PNG hashes matched at both resolutions.

The isolated operator comparisons measured 1093 rows: 3.37 → 2.76 ms. 16453 rows: 45.42 → 31.45 ms. Each accepted operator batch was within 5% block-median spread. The 4K DiT peak physical footprint fell in each pair by 0.43–0.70 GiB. The direction of 4K DiT timing varied across pairs. The observed physical footprint reduction determines the promotion.

| Resolution | Native DiT | End to end | DiT peak physical footprint | DiT max RSS |
| --- | ---: | ---: | ---: | ---: |
| 1024×1024 | 14.39 → 14.31 s | 18.12 → 18.03 s | 6.44 → 6.40 GiB | 1.70 → 1.70 GiB |
| 4096×4096 | 194.52 → 196.21 s | 213.35 → 215.31 s | 12.31 → 11.67 GiB | 1.69 → 1.72 GiB |

Run `.build/onyx-dit-modulation-probe 16453 --text-only` for the 4K-shaped operator comparison and set `ONYX_DIT_INDEXED_NORM=1` for the native candidate. [Raw samples, hashes, and build identity](../benchmarks/results/m5-max-36gb-dit-indexed-rmsnorm-v1.json)
