"""RoadScan web app — FastAPI backend + dashboard frontend.
Serves the working dashboard, runs detection, geotags, and stores reports.

Run:    uvicorn server:app --host 0.0.0.0 --port 7860
Deploy: Hugging Face Spaces (Docker SDK) — see DEPLOY_HF_SPACES.md
"""
import io
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, Body
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pathlib import Path

import core

ROOT = Path(__file__).parent
app = FastAPI(title="RoadScan")
core.seed_if_empty()      # populate 5 example reports on first run
app.mount("/uploads", StaticFiles(directory=str(core.UPLOADS)), name="uploads")


@app.get("/", response_class=HTMLResponse)
def index():
    return (ROOT / "dashboard.html").read_text()


@app.get("/api/reports")
def reports():
    return JSONResponse({"reports": core.recent(50), "stats": core.stats()})


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...),
                  lat: str = Form(None), lng: str = Form(None)):
    data = await file.read()
    if not data:
        return JSONResponse({"error": "empty file"}, status_code=400)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "could not read image"}, status_code=400)
    try:
        pil = Image.open(io.BytesIO(data))          # keep EXIF for GPS
    except Exception:
        pil = None
    result = core.analyze_bgr(img, pil_for_exif=pil, lat_in=lat, lng_in=lng)
    return JSONResponse(result)


def _pdf_response(pdf: bytes, filename: str):
    return Response(content=pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/api/report/{rid}/pdf")
def report_pdf(rid: str):
    return _pdf_response(core.build_pdf([rid]), f"RoadScan_{rid}.pdf")


@app.post("/api/report/pdf")
def reports_pdf(ids: list[str] = Body(..., embed=True)):
    fn = f"RoadScan_{ids[0]}.pdf" if len(ids) == 1 else f"RoadScan_report_{len(ids)}_items.pdf"
    return _pdf_response(core.build_pdf(ids), fn)


@app.get("/health")
def health():
    return {"ok": True}
