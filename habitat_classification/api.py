"""
FastAPI endpoint for habitat classification predictions.

Usage:
    python api.py

The server will start on http://0.0.0.0:4321
"""

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from model import predict
from utils import decode_patch

HOST = "0.0.0.0"
PORT = 4321


class PredictRequest(BaseModel):
    patch: str  # base64-encoded float32 bytes for array (15,35,35)


class PredictResponse(BaseModel):
    prediction: int  # Class index 0-70


app = FastAPI(
    title="Habitat Classification API",
    description="Classify Icelandic satellite image patches into 71 habitat types",
    version="1.0.0",
)


@app.get("/")
def index():
    return {"status": "running", "message": "Habitat Classification API"}


@app.get("/api")
def api_info():
    return {
        "service": "habitat-classification",
        "version": "1.0.0",
        "endpoints": {
            "/": "Health check",
            "/api": "API information",
            "/predict": "POST - Classify a patch",
        },
    }



from fastapi import HTTPException

@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    try:
        patch = decode_patch(request.patch)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid patch encoding: {e}")

    prediction = predict(patch)
    return PredictResponse(prediction=int(prediction))



if __name__ == "__main__":
    print(f"Starting server on http://{HOST}:{PORT}", flush=True)
    uvicorn.run(app, host=HOST, port=PORT, reload=False, log_level="info", access_log=False)
