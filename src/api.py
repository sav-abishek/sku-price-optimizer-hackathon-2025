from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from optimizer import run_optimizer


class OptimizeRequest(BaseModel):
    pinc_target: float = Field(0.02, ge=0.0, le=0.06)
    price_floor_delta: float = Field(-300.0)
    price_ceiling_delta: float = Field(500.0)
    price_step: float = Field(50.0, gt=0)


app = FastAPI(
    title="Hackathon 2025 RGM Optimizer API",
    description="FastAPI wrapper around the Round 2 pricing optimizer.",
    version="1.0.0",
)


@app.get("/health")
def healthcheck():
    return {"status": "ok"}


@app.post("/optimize")
def optimize(req: OptimizeRequest):
    try:
        result = run_optimizer(
            pinc_target=req.pinc_target,
            price_floor_delta=req.price_floor_delta,
            price_ceiling_delta=req.price_ceiling_delta,
            price_step=req.price_step,
        )
    except Exception as exc:  # pragma: no cover - surface unexpected issues
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    summary = result["summary"].copy()
    summary["volume_delta"] = summary["volume_new"] - summary["volume_base"]
    summary["nr_delta"] = summary["nr_new"] - summary["nr_base"]
    summary["maco_delta"] = summary["maco_new"] - summary["maco_base"]

    portfolio = result["portfolio"].copy()
    if {"base", "new"}.issubset(portfolio.columns):
        portfolio["delta"] = portfolio["new"] - portfolio["base"]

    response = {
        "metadata": result["metadata"],
        "portfolio": portfolio.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
        "architecture": result["architecture"].to_dict(orient="records"),
        "unmapped": result["unmapped"].to_dict(orient="records"),
    }
    return response


__all__ = ["app"]
