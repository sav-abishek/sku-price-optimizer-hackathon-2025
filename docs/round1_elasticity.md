# Round 1 Elasticity Model

This section documents how we regenerate the Round 1 submission: the demand model, elasticity extraction, business hypotheses, and safeguards that keep price and volume economically consistent.

## Data and Feature Engineering

- **Inputs** - Masked sell-out data from `Data Files/Sellout_Train.csv` and `Data Files/Sellout_Test_Predict.csv`.
- **Derived keys** - \(\text{sku\_str} = \operatorname{CONCAT}(\text{brand}, \text{sub\_brand}, \text{package}, \text{package\_type}, \text{capacity\_number})\) (see `add_derived_columns` in `src/compute_elasticity.py:259`).
- **Feature frame** - Log transforms and seasonality:
  - \(\text{log\_price} = \log(\text{avg\_price\_per\_liter})\)
  - \(\text{log\_nd} = \log(1 + \text{numeric\_distribution\_stores\_handling})\)
  - \(\text{log\_wd} = \log(1 + \text{weighted\_distribution\_tdp\_reach})\)
  - \(\text{log\_inv} = \log(1 + \text{inventory\_hectoliters})\)
  - \(\text{month\_sin} = \sin\left(\frac{2\pi m}{12}\right), \quad \text{month\_cos} = \cos\left(\frac{2\pi m}{12}\right)\)

Missing values in the scoring frame are filled with training medians prior to inference.

## Hierarchical Ridge Regression

The demand model is a closed-form ridge regression trained at three levels (global -> brand -> SKU), reusing the most granular fit available for each record (`predict_with_hierarchy` in `src/compute_elasticity.py:422`).

### Final Equation

For any SKU \(i\) and calendar period \(t\), the log-volume forecast is:

$$
\begin{aligned}
\ln \hat{V}_{i,t}
&= \beta_{0}^{(h)} + \beta_{p}^{(h)} \ln P_{i,t} + \beta_{nd}^{(h)} \ln(1 + ND_{i,t}) \\
&\quad + \beta_{wd}^{(h)} \ln(1 + WD_{i,t}) + \beta_{inv}^{(h)} \ln(1 + INV_{i,t}) \\
&\quad + \beta_{s}^{(h)} \sin\left(\frac{2\pi m_t}{12}\right)
  + \beta_{c}^{(h)} \cos\left(\frac{2\pi m_t}{12}\right)
\end{aligned}
$$

where:

- \(h \in \{\text{SKU}, \text{Brand}, \text{Global}\}\) is the most granular hierarchy with the minimum observations (`min_obs_sku = 6`, `min_obs_brand = 20`).
- \(P\) is average price per litre, \(ND\) numeric distribution, \(WD\) weighted distribution, \(INV\) inventory, and \(m_t\) calendar month.
- `RidgeClosedForm` (`src/compute_elasticity.py:328`) solves \( \beta = (X^\top X + \alpha I)^{-1} X^\top y \) with \(\alpha = 1.0\) and the intercept excluded from regularisation.

Predicted hectolitres are recovered via \(\hat{V}_{i,t} = \exp(\ln \hat{V}_{i,t})\).

### Packages and Algorithms

| Layer | Package / implementation | Notes |
| --- | --- | --- |
| Feature engineering | `pandas`, `numpy` | Deterministic log/seasonal transforms |
| Regression | Custom `RidgeClosedForm` | Closed-form ridge with intercept guard, no scikit dependency |
| Hierarchy orchestration | Native Python | Fallback order SKU -> brand -> global |
| Metrics | `numpy` | wMAPE, Pearson \(r\) for validation splits |

## Own and Cross Elasticities

1. **Own-price elasticity** - Extracted directly from the SKU-level ridge coefficient on `log_price` (`extract_own_elasticities` at `src/compute_elasticity.py:466`). Bounds enforced: \(-5.0 \le \epsilon_{ii} \le -0.1\); sign forced negative.
2. **Cross-price elasticity** - Template-driven matrix generated via `build_elasticity_matrix` (`src/compute_elasticity.py:526`):
   - Similarity score between SKU descriptors guides magnitude (brand/sub-brand/package/size token overlap).
   - Cross elasticity \( \epsilon_{ij} = \min\left(\max\left(0.3 \cdot |\epsilon_{ii}| \cdot \text{similarity}_{ij}, 0.01\right), 2.0\right) \).
   - Manufacturer label retained to distinguish ABI versus competitor pairings.

The resulting matrix satisfies the round rulebook: every ABI SKU carries its own elasticity plus 2-3 ABI cross links; where the template offers competitor rows we propagate them unchanged.

## Ensuring Price-Volume Coherence

- **Feature choice** - Prices enter exclusively through `log_price`, creating the expected log-log elasticity relationship.
- **Sign enforcement** - Any SKU-level ridge regression returning a non-negative price coefficient is clipped back to \(-0.1\).
- **Fallback elasticity** - If a SKU lacks sufficient history, we use the median own elasticity (`__FALLBACK__`) to keep optimisation grounded.
- **Hierarchical inference** - By preferring SKU-level coefficients and only backing off to brand/global fits when required, we reduce leakage across heterogeneous SKUs.

## Business Hypotheses and Assumptions

1. **Stationary price-response** - Historical log-log elasticities sufficiently describe near-term reactions; no structural breaks are modelled.
2. **Distribution drives availability** - Numeric and weighted distribution proxies account for shelf presence without explicit promo flags.
3. **Template coverage suffices** - The supplied masked template lists the most relevant cannibalisation partners; similarity scoring fills residual gaps.
4. **Seasonality approximates a monthly cycle** - Sin/Cos harmonics capture predictable monthality without overfitting daily shocks.

These assumptions were validated against the masked leaderboard scores in Round 1; the regenerated predictions match the submitted CSVs byte-for-byte.

## What Could Be Better With More Time?

- Introduce promotion and pricing event features (for example depth and frequency) once unmasked flags become available.
- Replace heuristic cross elasticities with estimated cross-SKU regression using Bayesian shrinkage.
- Enrich seasonality with holiday and temperature signals to improve shoulder periods.
- Automate template refresh directly from unmasked Sell-in data to avoid manual partner curation.
