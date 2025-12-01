# Source Code Layout

- `compute_elasticity.py` — regenerates the ridge models and elasticity artefacts submitted in Round 1  
- `optimizer.py` — loads Round 2 Sell-in/Sell-out workbooks, reuses the fixed elasticity matrix, and applies the pricing heuristic  
- `api.py` — placeholder for the upcoming FastAPI service that will expose `/optimize` and `/health`
- `streamlit_app.py` — Streamlit UI located at the repository root; it imports `optimizer` to render the dashboards

The package relies on `pandas`, `numpy`, and `scipy`. Install dependencies with:

```bash
pip install -r ../requirements.txt
```
