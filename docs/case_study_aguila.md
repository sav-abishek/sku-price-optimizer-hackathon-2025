# Case Study - Aguila 330 RB

To ground the optimisation in business terms, we analyse the highest-volume ABI SKU in the portfolio: **Aguila 330 ml returnable bottle** (`key = AGUILA|RB|330`).

## Baseline Snapshot

| KPI | Value | Source |
| --- | --- | --- |
| Volume | **6,256,602 HL** | `sellin_summary['volume_hl']` |
| Net Revenue | **COP 2.52e12** | `sellin_summary['NR']` |
| MACO | **COP 1.81e12** | `sellin_summary['MACOstd']` |
| Segment | Core | Price architecture metadata |
| Own elasticity | **-0.10** | `extract_own_elasticities` |

A 1% list-price increase therefore implies a 0.10% volume decline (about 6,256 HL) before accounting for cross effects.

## Cannibalisation Matrix (ABI SKUs)

The Round 1 elasticity matrix exposes the strongest ABI substitutes. Applying a 1% price lift to the 330 RB pack frees demand that flows into adjacent sizes and packaging.

| Substitute SKU | Elasticity \( \epsilon_{ij} \) | Base volume (HL) | Volume gain for +1% price on 330 RB (HL) |
| --- | --- | --- | --- |
| Aguila 330 ml NRB | 0.0285 | 6,260,480 | **+1,783** |
| Aguila 750 ml RB | 0.0281 | 148,238 | +41.5 |
| Aguila 250 ml RB | 0.0300 | 962,312 | +28.9 |
| Aguila 1 L RB | 0.0275 | 739,177 | +20.4 |

Interpretation:

- **Package switching** - The strongest substitution comes from the same liquid in non-returnable glass. The 330 NRB pack absorbs roughly 28% of the volume loss, reflecting on-trade consumers trading to a one-way pack.
- **Size laddering** - Larger returnable packs (750 ml, 1 L) pick up the remainder, suggesting group consumption occasions tolerate upsizing with minimal friction.
- **Small formats** - The 250 ml RB pack captures incremental volume in convenience channels.

The optimiser uses these elasticities when re-scoring portfolio KPIs, so MACO and volume are rebalanced according to the sourcing matrix above.

## Competition and Industry Volume

The unmasked elasticity template bundled with `Data Files Round 2/` does **not** list competitor-to-ABI elasticities. As a result, competitor volume remains static and the industry share guardrail becomes the primary control point for this SKU. During the live demo we will:

1. Call out the missing competitor entries (visible in the exported elasticity CSV).
2. Show how the market-share constraint prevents the optimiser from unrealistically inflating ABI share despite the missing cross-elasticities.
3. Outline next steps (see [GenAI & Roadmap](genai_future.md#roadmap-if-we-had-more-time)) to estimate competitor responses from the industry model once richer data is unlocked.

## Key Takeaways for Commercial Teams

- **Price leverage is limited** - With an own elasticity of -0.10, even small price moves erode high volumes; guardrails and rounding quickly force a baseline fallback.
- **Focus on architecture** - Gains come from nudging consumers between pack types and sizes. The Streamlit price architecture chart visualises these substitutions in real time.
- **Mind the data gap** - Competitor sourcing is conservatively handled today; integrating competitor elasticities is the top post-hackathon priority.
