# Engineering Quality

This page addresses the "Local setup & execution", "Code quality", and "Testing implementation" scoring pillars.

## Local Execution Checklist

1. **Create a virtual environment** (Python 3.9+).
2. **Install dependencies**: `pip install -r requirements.txt`.
3. **Optional Round 1 regeneration**: `python src/compute_elasticity.py data_dir="Data Files" out_dir=outputs`.
4. **Run the web app**: `streamlit run streamlit_app.py`.
5. **API option** (same environment): `uvicorn src.api:app --reload`.

All data required for reproduction ships with the repository. No hidden environment variables or database credentials are needed.

## Code Quality

| Practice | Implementation |
| --- | --- |
| Deterministic styling | Ruff-compatible codebase (`ruff check .` runs clean); no unused imports or dead paths. |
| Layered architecture | `src/compute_elasticity.py` (modelling), `src/optimizer.py` (optimisation), `src/api.py` (service layer), `streamlit_app.py` (UI). |
| Defensive programming | FastAPI wraps the optimiser call in a `try/except` and surfaces 500 with a clear message; Streamlit sanitises sidebar inputs before execution. |
| Reproducible artefacts | Outputs (elasticity matrix, portfolio summaries) are written to `outputs/` with filenames documented in `run_summary.json`. |
| Documentation in code | Concise docstrings (for example `PricingOptimizer`) and inline comments for non-obvious transformations. |

We deliberately avoided framework-specific "magic" so the judges can inspect and run the code with standard tooling.

## Testing Strategy

| Scope | File | What it covers |
| --- | --- | --- |
| API smoke tests | `tests/test_api.py` | `/health` readiness and `/optimize` happy path (baseline scenario). |
| Manual notebook checks | (notebook omitted from repo) | Elasticity sanity checks prior to regenerating Round 1 artefacts. |
| Optimiser diagnostics | Streamlit download buttons | CSV exports facilitate manual reconciliation of portfolio KPIs. |

To run the automated suite:

```bash
pytest
```

> The `/optimize` smoke test runs the real optimiser and takes about 10 seconds to execute; expect a short pause.

## Documentation Workflow

- MkDocs content mirrors the judging rubric; this site is the single source of truth during the live demo.
- `README.md` targets hands-on evaluators; MkDocs targets storytelling and Q&A readiness.
- Updates to methodology or assumptions must be reflected in both the code (`src/`) and the documentation (`docs/`) before merging.
