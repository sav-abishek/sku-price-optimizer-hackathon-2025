# Hackathon 2025 RGM Pricing Optimizer

End-to-end workspace for the Hackathon 2025 Round 2 submission. The repository reproduces the Round 1 demand models, computes refreshed elasticities from the Sell-in/Sell-out workbooks, and surfaces both a Streamlit UI and a FastAPI service for price-scenario analysis.

## Preview
![Hackathon Website Preview](https://github.com/user-attachments/assets/f0994990-5c7a-4648-9d4c-f23c62629160)
![Hackathon Journey 2](https://github.com/user-attachments/assets/f804ece7-7979-456c-beb8-b2c0edd852ba)
![Hackathon Journey 1](https://github.com/user-attachments/assets/50ac9521-208e-4538-80b9-fb4bfcc835ad)


## Setup Instructions

### Prerequisites

```
- Python 3.9+
- pip (bundled with Python)
- Optional: MkDocs Material (`pip install mkdocs-material`) for documentation previews
```

### Installation

```bash
# Clone the repository
git https://github.com/ab-inbev-analytics/analytics-hackathon-round2
cd analytics-hackathon-round2

# Install Python dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# (Optional) Recreate Round 1 deliverables from the masked inputs
python src/compute_elasticity.py data_dir="Data Files" out_dir=outputs

# Launch the Streamlit web app
streamlit run streamlit_app.py
```

Streamlit opens at `http://localhost:8501`. Use the sidebar controls to adjust the PINC target, price guardrails, and step size, then click **Run Optimization** to refresh the scenario.

### Running Tests

```bash
pytest
```

The tests exercise the `/health` and `/optimize` API endpoints using the live optimizer.

## API Endpoints

Start the FastAPI service:

```bash
uvicorn src.api:app --reload
```

- **GET `/health`** – readiness probe returning `{"status": "ok"}`  
- **POST `/optimize`** – run the pricing optimizer and return KPI summaries and tables. Example body:
  ```json
  {
    "pinc_target": 0.02,
    "price_floor_delta": -300,
    "price_ceiling_delta": 500,
    "price_step": 50
  }
  ```

## Project Structure

```
├── Data Files/                 # Round 1 (masked) CSV inputs
├── Data Files Round 2/         # Round 2 Sell-out/Sell-in Excel workbooks
├── docs/                       # MkDocs content (index, requirements, design notes)
├── models/                     # Notes for reproducing Round 1 models
├── outputs/                    # Generated predictions & elasticity artefacts
├── outputs/round2/             # Optimizer exports (prices, portfolio, mapping)
├── scripts/                    # One-off utilities (SKU mapping, template generator)
├── src/
│   ├── compute_elasticity.py   # Hierarchical ridge pipeline for Round 1
│   ├── optimizer.py            # Pricing optimizer and heuristics
│   ├── api.py                  # FastAPI routes for `/health` and `/optimize`
│   └── README.md               # Source layout overview
├── streamlit_app.py            # Streamlit UI entry point
├── tests/                      # Pytest suite for API smoke tests
├── mkdocs.yml                  # Documentation site configuration
└── requirements.txt            # Python dependencies
```

## Documentation

Serve the MkDocs site locally during presentations or dry runs:

```bash
mkdocs serve
# open http://127.0.0.1:8000
```

The site renders `docs/index.md`, the functional requirements, and design notes.

## Troubleshooting

- **Missing data files** – ensure `Data Files/` (masked Round 1 assets) and `Data Files Round 2/` (unmasked Excel workbooks) contain the official inputs with the expected filenames.  
- **Streamlit cannot import `optimizer`** – confirm `streamlit_app.py` remains at the repository root so the dynamic path adjustment continues to reference `src/`.  
- **Optimizer reverts to baseline** – the banner “reverted to baseline (no price changes)” indicates the rounded solution violated a guard rail; relax the price bounds or volume/share constraints and rerun.

---
