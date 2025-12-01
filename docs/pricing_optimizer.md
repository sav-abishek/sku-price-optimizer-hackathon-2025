# Pricing Optimizer

Round 2 extends the elasticity work into an actionable pricing engine. This page covers the data preparation, optimisation objective, constraint handling, and the current limitations that will guide the live demo narrative.

## Data Pipeline

1. **Sell-in aggregation** - `build_sellin_summary` (`src/optimizer.py:323`) rolls `Sellin.xlsx` to SKU level, computes portfolio KPIs (`NR`, `MACOstd`, `volume_hl`), derives guard-rail attributes (segment, size group), and backfills PTC when missing via $\text{PTC} = \frac{\text{NR}}{\text{volume\_units}}$.
2. **Price list merge** - PTC, PTR, markup, and pack size metadata are injected from `Price_List.xlsx`.
3. **Sell-out bridge** - Fuzzy matching (`map_sellout_to_sellin` at `src/optimizer.py:387`) links unmasked sell-out SKUs to aggregated sell-in keys for market-share computations.
4. **Elasticity reuse** - The Round 1 elasticity matrix is re-hydrated via `sku_mapping` so every ABI key inherits its own elasticity plus mapped cross elasticities.
5. **Baselines** - `PricingOptimizer._prepare` stores base volume, MACO, NR, and market share before any perturbation. These figures seed the guard-rail checks.

## Objective Function

We maximise total MACO across the ABI portfolio subject to the user-specified portfolio PINC:

$$
\max_{\Delta p} \quad \text{MACO}_{\text{ABI}}(\Delta p)
$$

subject to:

| Constraint | Implementation |
| --- | --- |
| PINC target $ \sum_i w_i \cdot \Delta p_i = \text{PINC}_{\text{target}} $ | Weighted average via `_portfolio_pinc` |
| Price bounds $ \Delta p_i \in [\ell_i, u_i] $ | Derived from floor/ceiling deltas |
| Volume guard rails $ -1\% \le \frac{V_{\text{new}}}{V_{\text{base}}} - 1 \le 5\% $ | Penalties in `_constraint_penalty` |
| MACO non-decreasing $ \text{MACO}_{\text{new}} \ge \text{MACO}_{\text{base}} $ | Penalty term |
| Industry volume $ \text{Industry}_{\text{new}} \ge 99\% \times \text{Industry}_{\text{base}} $ | Penalty term |
| Share guard rail $ \text{Share}_{\text{new}} \ge \text{Share}_{\text{base}} - 0.5\% $ | Penalty term |
| Rounding (50-unit) | Rounded post-search, re-checked |

MACO, NR, and volume are computed in `_financial_projection` (`src/optimizer.py:595`) using the elasticity matrix to project new units:

$$
V_{i}^{\text{new}} = V_i^{\text{base}} \cdot \left( 1 + \epsilon_{ii} \Delta p_i + \sum_{j \in \text{ABI}} \epsilon_{ij} \Delta p_j \right)_+
$$

where $ \epsilon_{ii} $ are the own elasticities and $ \epsilon_{ij} $ the mapped cross elasticities. New MACO follows ABI's definition: $\text{MACO} = \text{NR} - \text{VILC}$ with VAT and markup adjustments applied per SKU.

## Search Strategy

The optimiser uses a MILP solver aligned to the plan (integer 50‑unit price steps, linearised guardrails), with a deterministic heuristic fallback. Typical runs complete in under 15 seconds on a laptop:

1. **MILP (preferred)** - Integer step variables enforce 50‑unit price steps; PINC equality and volume/industry/share guardrails are linear constraints; the objective maximises an approximate linear MACO.
2. **Heuristic (fallback)** - Uniform PINC allocation clipped to bounds; portfolio balancing; binary scaling against penalties; post‑search rounding in 50‑unit steps; feasibility check with baseline fallback.

The Streamlit app calls `run_optimizer`, receives the summary/portfolio tables, and exposes the constraint diagnostics so the user understands when and why the solution reverts.

### Illustrative Objective Curve

Using the default target PINC (3%) and ABI guardrails, scaling the initial price vector demonstrates how penalties accumulate:

| Scale factor $s$ | Portfolio PINC | MACO (COP) | ABI volume (HL) | Penalty |
| --- | --- | --- | --- | --- |
| 0.00 | 0.000 | 7.92e12 | 3.25e7 | 0 |
| 0.25 | 0.007 | not available (rounding violates MACO/industry checks) | 3.41e7 | 0 |
| 0.50 | 0.015 | not available (penalty triggered) | 3.65e7 | 2.36e6 |
| 0.75 | 0.023 | not available (penalty triggered) | 3.88e7 | 4.75e6 |
| 1.00 | 0.030 | not available (penalty triggered) | 4.12e7 | 7.14e6 |

The sharp rise in penalty reflects volume and MACO guardrails being breached once price lifts exceed roughly 0.75 of the initial vector. The Streamlit UI surfaces the same diagnostics, helping business users tune guardrails before re-running the optimiser.

## How Elasticity Feeds the Optimiser

- **Own elasticity** drives each SKU's direct volume response (`self.own_lookup`).
- **Cross elasticity (ABI)** channels cannibalisation gains into substitute SKUs via `self.cross_lookup`.
- **Competitor elasticity** - the unmasked template does not include competitor-to-ABI pairings, so competitor volume stays flat in Round 2. The market-share guardrail compensates by capping ABI uplift; documenting this gap is part of the post-hackathon plan.

## Handling Soft Constraints

Soft guardrails (segment hierarchy and size architecture) are computed in `evaluate_constraints` and displayed in the UI:

- **Segment NR/HL order** - Value < Core < Core+ < Premium < Super Premium.
- **Size NR/HL order** - Small > Regular > Large.

We currently monitor these ratios and flag violations rather than enforcing them algorithmically. In practice, price rounding plus the MACO guardrail preserves the expected ordering for the supplied datasets; extending the optimiser to inject explicit penalties is on the roadmap (see [GenAI & Roadmap](genai_future.md#roadmap-if-we-had-more-time)).

## Known Limitations

- No competitor elasticities in the unmasked template, so industry volume and share constraints become conservative.
- Guard-rail penalties rely on deterministic heuristics; plugging in SLSQP or proximal gradient descent would support finer adjustments once richer elasticities are available.
- Price rounding happens post-search; in edge cases this creates feasibility gaps that force a baseline fallback.

These trade-offs are transparent in the Streamlit UI and will be highlighted live so judges understand how to interpret the optimisation output.

## Price Steps and Guardrails

- Rounded prices obey the required step size:
  $$
  \text{PTC}_{\text{new}} = \operatorname{round}_{50}\!\left(\text{PTC}_{\text{base}} (1 + \Delta p)\right).
  $$
- After rounding we clip to $ \text{PTC}_{\text{base}} + [\text{floor}, \text{ceiling}] $ to guarantee compliance with SKU-level bounds.
