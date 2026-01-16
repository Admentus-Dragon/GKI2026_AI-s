import numpy as np
import base64
import requests

URL = "https://jace-unengraved-unresiliently.ngrok-free.dev/predict"
API_KEY = "GKI2026-habitat-9f3a8c1e2b7d4ebb"

patch = np.random.randn(15,35,35).astype(np.float32)
payload = {
    "patch": base64.b64encode(patch.tobytes()).decode("utf-8")
}

headers = {
    "X-API-Key": API_KEY
}

r = requests.post(URL, json=payload, headers=headers, timeout=10)
print(r.status_code, r.text)
