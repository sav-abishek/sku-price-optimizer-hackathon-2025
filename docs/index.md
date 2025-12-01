# Hackathon 2025 RGM Pricing Optimizer

End-to-end dossier covering the Round 1 elasticity reproduction and the Round 2 pricing optimizer we are submitting to the Hackathon 2025 jury panel.

## TLDR

- **Ready to run locally** - one `pip install -r requirements.txt`, one entry point (`streamlit run streamlit_app.py`). Streamlit, FastAPI, and the optimizer share the same virtual environment.
- **Round 1 reproduced** - hierarchical ridge models regenerate volume forecasts and the own/cross elasticity matrix delivered in Round 1 (`src/compute_elasticity.py`).
- **Round 2 optimized** - guarded heuristic optimizer (`src/optimizer.py`) balances MACO uplift, PINC, and share while respecting price guardrails and 50-unit rounding.
- **Documentation-driven** - MkDocs mirrors the judging rubric: methodology, engineering hygiene, testing, and roadmap.

## How to Use This Site

| Goal | Where to start |
| --- | --- |
| Revisit the modelling stack and final equations | [Round 1 Elasticity Model](round1_elasticity.md) |
| Inspect the pricing engine, constraints, and objective function | [Pricing Optimizer](pricing_optimizer.md) |
| Walk through a SKU-level cannibalisation story | [Case Study - Aguila 330 RB](case_study_aguila.md) |
| Validate code quality, testing, and release hygiene | [Engineering Quality](engineering.md) |
| Capture GenAI usage, key assumptions, and roadmap | [GenAI & Roadmap](genai_future.md) |

## Evaluation Readiness

- **Local setup & execution (25%)** - Detailed instructions live in `README.md`; troubleshooting mirrors are in [Engineering Quality](engineering.md#local-execution-checklist). No hidden environment variables, no manual data munging.
- **Code quality (15%)** - Deterministic formatting (ruff profile), layered modules, defensive error handling (FastAPI exception guard, Streamlit input sanitisation). Highlights summarised in [Engineering Quality](engineering.md#code-quality).
- **Testing implementation (15%)** - Pytest suite covers `/health` and `/optimize` smoke cases; elasticity utilities are validated via deterministic fixtures (see [Engineering Quality](engineering.md#testing-strategy)).
- **Documentation quality (15%)** - This MkDocs site anchors the live walk-through; each judging question is answered explicitly in [Round 1 Elasticity Model](round1_elasticity.md), [Pricing Optimizer](pricing_optimizer.md), and [GenAI & Roadmap](genai_future.md).

## Repository Guide

```
Data Files/              # Masked Round 1 CSVs (elasticity reproduction)
Data Files Round 2/      # Unmasked Sell-in/Sell-out workbooks
docs/                    # MkDocs content (this site)
src/
  compute_elasticity.py  # Hierarchical ridge + elasticity builder
  optimizer.py           # Pricing heuristics, constraints, exports
  api.py                 # FastAPI wrapper for batch execution
streamlit_app.py         # Front-end for pricing scenarios
tests/                   # Pytest smoke suite
outputs/                 # Regenerated Round 1 assets + optimizer runs
```

Use the navigation bar or the quick links above to dive into modelling, optimisation, and engineering details before the live presentation.
