# GenAI & Roadmap

This page consolidates the remaining judging questions: how generative AI supported the build, the cross-cutting business assumptions, and what we would tackle next with more time or data.

## How We Leveraged GenAI

- **Code scaffolding** - GPT-5/Codex pair-programming sessions helped draft the initial Streamlit layout and FastAPI boilerplate, reducing boilerplate time without embedding any runtime dependency.
- **Documentation support** - Large-language models summarised the 70 page hackathon brief into actionable checklists (`instructions.txt`, MkDocs structure).
- **UI copy and theming** - Prompted GenAI for tone-aligned microcopy and colour palettes, iterating manually before committing.

No GenAI model runs inside the product; all usage was offline and human-in-the-loop to preserve reproducibility.

## Business Hypotheses & Assumptions

1. **Elasticity stationarity** - Short-term price moves behave like the historical log-log elasticities; structural breaks (COVID, excise shocks) are absent in the masked data.
2. **Distribution as proxy for availability** - Numeric and weighted distribution variables capture shelf presence well enough to stand in for explicit OOS flags.
3. **Uniform guardrails** - The provided price floor/ceiling and PINC bounds represent current management guidance; we do not dynamically infer tighter limits per segment.
4. **Industry model exogenous** - Industry volume caps are enforced via constraints rather than re-estimating a full macro model inside the optimiser.
5. **Competitor elasticity gap** - Until unmasked competitor mappings are released, we rely on portfolio share guards to avoid over-claiming gains.

These assumptions are reiterated during the demo so commercial stakeholders appreciate the guardrails around the optimiser's recommendations.

## Roadmap (If We Had More Time/Data)

1. **Competitor response modelling** - Estimate competitor cross elasticities by combining unmasked industry data with instrumental-variable regressions; feed them into `competitor_effect`.
2. **True constrained optimiser** - Swap the heuristic scaler for SLSQP or sequential convex programming, letting us honour rounding and guardrails simultaneously.
3. **Promotion-aware demand** - Integrate promo depth, frequency, and feature/display flags to separate price and promo effects in the elasticity estimation.
4. **Scenario memory in Streamlit** - Allow users to bookmark and compare multiple optimisation runs, exporting a change log for commercial sign-off.
5. **Automated validation harness** - Extend `tests/` with unit tests for elasticity generation and regression tests comparing optimiser outputs against golden CSVs.

These enhancements, coupled with cleaner competitor data, would let us maximise MACO without the conservative fallbacks visible in the current release.
