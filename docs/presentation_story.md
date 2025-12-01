# Presentation Narrative

"_Tell us the story in ten minutes._" That was the jury brief. This section doubles as our speaking script so we can walk judges from business context to optimisation output without losing the thread.

## 1. Setting the Stage - Business Assumptions

1. **Stable short-run behaviour** - Price response in the coming quarter behaves like the elasticities we measured in the masked Round 1 data; no extraordinary shocks (COVID, tax spikes) are modelled.
2. **Distribution approximates availability** - Numeric and weighted distribution are our best proxies for whether a shopper can find the SKU, because explicit OOS flags are missing.
3. **Guardrails are non-negotiable** - The PINC window, SKU price floors/ceilings, and share limits in the brief reflect management policy, so we treat them as hard constraints.
4. **Industry guidance is exogenous** - We accept the 1% industry volume floor rather than rebuilding the macro model inside the optimiser.
5. **Competitor behaviour is latent** - Competitor elasticities are absent from the unmasked template, so competitor volume stays flat and the market-share constraint keeps ABI share honest.

## 2. Rebuilding Elasticity - Why This Model?

### 2.1 Why log transforms?

- Constant-elasticity demand theory suits the problem framing.
- Log transformations linearise the multiplicative relationship, so the coefficient on $\log(\text{price})$ is the elasticity we need.
- Log scaling also tames heteroskedasticity between SKUs that sell 100 HL versus 10,000 HL.

### 2.2 Why hierarchical layers?

- Some SKUs have rich history; others are sparse.
- We train closed-form ridge regressions at global, brand, and SKU levels (`train_hierarchical_models` in `src/compute_elasticity.py`).
- At prediction time we use the most granular model available; if a SKU lacks depth we fall back to brand and then global. This preserves SKU nuance without sacrificing coverage.

### 2.3 Why a closed-form solution?

- The ridge equation $ \beta = (X^\top X + \alpha I)^{-1} X^\top y $ is analytical and deterministic.
- No numerical solver means retraining is fast and repeatable—handy during live demos and code reviews.

### 2.4 Why ridge regularisation?

- Price, numeric distribution, and weighted distribution are correlated; ridge stabilises the coefficients.
- It guards against overfitting on SKUs with short time series and ensures every SKU yields a negative own-price elasticity.
- A common $\alpha = 1.0$ across hierarchy layers keeps the modelling playbook simple.

## 3. Meeting the Elasticity Rulebook

1. **Own elasticities stay negative** - Any positive coefficient is clipped to at most $-0.1$ (`extract_own_elasticities`).
2. **Cross elasticities stay positive** - Template pairs use $ \epsilon_{ij} = \min\left(\max\left(0.3 |\epsilon_{ii}| \cdot \text{similarity}_{ij}, 0.01\right), 2.0\right) $ so cannibalisation is always positive.
3. **Coverage is preserved** - Every ABI SKU retains the templated ABI partners; fuzzy similarity backfills when templates are sparse.

The regenerated matrix matches our Round 1 submission byte-for-byte, satisfying the hackathon scoring criteria.

## 4. Turning Elasticity into Action - The Optimiser

### 4.1 What optimiser did we build?

A deterministic heuristic wrapped around the elasticity-aware KPI engine (`PricingOptimizer` in `src/optimizer.py`). Think of it as a safeguarded line search—transparent, fast, and reproducible.

### 4.2 How does it work?

1. Seed every SKU with the user PINC target, clipped to its floor/ceiling.
2. Adjust only the free SKUs so the weighted PINC matches the target.
3. Scale the price-change vector by a factor $s \in [0, 1]$ with binary search until all hard constraints pass.
4. Round prices to the 50-unit step, re-clip to bounds, and re-check constraints.
5. If rounding breaks a guardrail, revert to baseline and surface the reason in the metadata and UI.

### 4.3 How does it use Round 1 elasticities?

- Own elasticities drive $ V_i^{\text{new}} = V_i^{\text{base}} (1 + \epsilon_{ii} \Delta p_i) $.
- Cross elasticities redistribute volume across ABI substitutes so cannibalisation flows through to MACO, NR, and share.
- Competitor volume stays flat because competitor elasticities are not available; the market-share constraint prevents ABI overreach.

### 4.4 Why this optimiser?

- **Transparency** - every adjustment is explainable to business stakeholders.
- **Speed** - a full run completes in under 15 seconds on a laptop.
- **Determinism** - identical inputs yield identical outputs, easing QA and handover.

### 4.5 Hard vs. soft constraints

- **Hard**: MACO non-negative delta, volume window ($-1\%$ to $+5\%$), industry floor, share floor, portfolio PINC, SKU price bounds, and 50-unit steps. Violation triggers a baseline fallback.
- **Soft**: Segment and size NR/HL hierarchies are monitored via diagnostics (`evaluate_constraints`); users can review them in the Streamlit dashboard.

### 4.6 When no feasible optimisation exists

If every scaled and rounded solution violates a hard constraint, we return the baseline scenario and flag “reverted to baseline – guardrail binding” in both API and UI responses.

## 5. End-to-End Journey (for the deck)

1. Rehydrate Round 1 models to reproduce forecasts and elasticities.
2. Map masked to unmasked SKUs using `slugify` and fuzzy token matching.
3. Build the elasticity matrix with negative own, positive cross, and required coverage.
4. Aggregate Round 2 Sell-in data to compute MACO, NR, volume, and share baselines.
5. Run the guarded optimiser against user PINC and guardrails.
6. Present KPI deltas, price architecture, and exports in the Streamlit UI.

## 6. Price Steps and Guardrails

- Prices are rounded after feasibility scaling using
  $$
  \text{PTC}_{\text{new}} = \operatorname{round}_{50}\!\left(\text{PTC}_{\text{base}} (1 + \Delta p)\right).
  $$
- We clip to $ \text{PTC}_{\text{base}} + [\text{floor}, \text{ceiling}] $ before and after rounding so no SKU leaves the approved range.

## 7. Taming Messy Data

- **Slugify everything** - remove accents and punctuation so “BOTELLA NO RETORNABLE” matches “NRB”.
- **Fuzzy joins** - `map_sellout_to_sellin` combines slugified brand tokens, pack types, and size buckets to reconcile Sell-out, Sell-in, and price list SKUs.
- **Defensive defaults** - missing capacity or markup values fall back to sensible estimates to keep every SKU in the optimiser.

## 8. If We Had More Time

1. Estimate competitor elasticities from the industry datasets to model share transfers explicitly.
2. Layer a formal constrained optimiser (such as SLSQP) on top of our heuristic for finer tuning.
3. Make the demand model promotion-aware by incorporating depth, feature, and display signals.
4. Add scenario memory in Streamlit so users can bookmark and compare optimisation runs.
5. Build regression tests with golden optimiser outputs to secure future refactors.

That roadmap forms our closing slide when judges ask how we would productionise the prototype after the hackathon.
