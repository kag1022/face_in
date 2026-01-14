import time

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .analyzer import FaceAnalyzer
from .schemas import (
    AnalysisResponse,
    DiagnosisInfo,
    Scores,
    ScoreItem,
    Landmarks,
    DiagnosisMeta,
)

app = FastAPI(
    title="Face Diagnosis API",
    description="Product-Ready API for Face Diagnosis",
    version="2.0.0",
)


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Face Diagnosis API is running."}


# CORS Setup
origins = [
    "http://localhost:3000",
    "http://localhost:8000",
    "*",  # Relaxed for dev, tighten in production if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Analyzer (Singleton-like)
# Ensure tasks file is present or fallback to mock is handled inside class
analyzer = FaceAnalyzer(model_path="./face_landmarker.task")


def get_verdict(key: str, z_score: float) -> str:
    """
    Generate readable verdict based on z-score.
    Thresholds: +/- 0.5 (Standard deviation units)
    """
    if abs(z_score) <= 0.5:
        return "Standard"

    if key == "aspect_ratio":
        # Positive Z for aspect -> Long face -> "Long"
        return "Long" if z_score > 0 else "Short"  # or Round
    elif key == "eye_ratio":
        # Positive Z for eye -> Far -> "Far"
        return "Far" if z_score > 0 else "Close"
    elif key == "lower_ratio":
        # Positive Z -> Long chin -> "Long"
        return "Long" if z_score > 0 else "Short"

    return "High" if z_score > 0 else "Low"


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_face(file: UploadFile = File(...)):
    start_time = time.time()

    # 1. Validation
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=400, detail="Invalid file type. Only JPEG/PNG/WEBP allowed."
        )

    # 2. In-memory Read
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to decode image data.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")

    h, w = image.shape[:2]

    # 3. Analyze
    try:
        # Assuming gender is hardcoded or could be parameter. Default 'female'.
        result = analyzer.analyze_from_array(image, gender="female")
    except Exception as e:
        print(f"Analyzer Error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    # 4. Map to Schema
    details = result.get("details", {})
    geo_metrics = details.get("geometric_metrics", {})
    z_scores = details.get("z_scores", {})
    landmarks_data = result.get("landmarks", {})

    # Construct Scores
    scores_obj = Scores(
        aspect_ratio=ScoreItem(
            value=geo_metrics.get("aspect_ratio", 0),
            z_score=z_scores.get("aspect", 0),
            verdict=get_verdict("aspect_ratio", z_scores.get("aspect", 0)),
        ),
        eye_ratio=ScoreItem(
            value=geo_metrics.get("eye_ratio", 0),
            z_score=z_scores.get("eye", 0),
            verdict=get_verdict("eye_ratio", z_scores.get("eye", 0)),
        ),
        lower_ratio=ScoreItem(
            value=geo_metrics.get("lower_ratio", 0),
            z_score=z_scores.get("lower", 0),
            verdict=get_verdict("lower_ratio", z_scores.get("lower", 0)),
        ),
    )

    process_time = time.time() - start_time

    return AnalysisResponse(
        diagnosis=DiagnosisInfo(
            type=result.get("type", "Unknown"),
            generation_score=details.get("generation_score", 0),
            shape_score=details.get("shape_score", 0),
        ),
        scores=scores_obj,
        landmarks=Landmarks(
            face_box=landmarks_data.get("face_box", [0, 0, 0, 0]),
            chin_line=landmarks_data.get("chin_line", []),
            left_eye_line=landmarks_data.get("left_eye_line", []),
            right_eye_line=landmarks_data.get("right_eye_line", []),
            all_points=landmarks_data.get("all_points", []),
        ),
        meta=DiagnosisMeta(
            process_time=round(process_time, 3), image_width=w, image_height=h
        ),
    )
