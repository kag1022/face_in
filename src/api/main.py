from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import cv2
from typing import Dict, Any

from ..analyzer import FaceAnalyzer
from .schemas import AnalysisResponse, GeometricScores, ComparisonData

app = FastAPI(
    title="Face Diagnosis API",
    description="API for Explainable Face Diagnosis Logic",
    version="1.0.0",
)

# Initialize Analyzer (Lazy loading inside)
analyzer = FaceAnalyzer()

# Constants for Average Data (Japanese Average Mock Data)
AVERAGE_SCORES = {
    "aspect_ratio_score": 0.5,  # 0.5 is considered standard oval
    "eye_spacing_score": 0.5,  # 0.5 is standard spacing
    "gravity_score": 0.5,  # 0.5 is standard positioning
}


def calculate_comparison(scores: Dict[str, float]) -> ComparisonData:
    """Calculate difference from average scores."""
    return ComparisonData(
        diff_aspect_ratio=round(
            scores["aspect_ratio_score"] - AVERAGE_SCORES["aspect_ratio_score"], 2
        ),
        diff_eye_spacing=round(
            scores["eye_spacing_score"] - AVERAGE_SCORES["eye_spacing_score"], 2
        ),
        diff_gravity=round(
            scores["gravity_score"] - AVERAGE_SCORES["gravity_score"], 2
        ),
    )


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_face(file: UploadFile = File(...)):
    # 1. Validation
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400, detail="Invalid file type. Only JPEG/PNG allowed."
        )

    # 2. Read Image (In-memory)
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Failed to decode image.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")

    # 3. Analyze
    try:
        result = analyzer.analyze_from_array(image)
    except Exception as e:
        # Unexpected analysis error
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    if "error" in result:
        # Logic specific errors (No face, etc)
        raise HTTPException(status_code=422, detail=result["error"])

    # Extract details
    details = result.get("details", {})
    geo_scores_dict = details.get("geometric_scores", {})
    imp_scores = details.get("impression_scores", {})
    matrix_coords = details.get("matrix_coordinates", {})

    # 4. Calculate Comparison
    # Ensure scores are present and float (handle potential missing keys safely if needed)
    # Using Pydantic validation here implicitly but let's be explicit manually if specific keys are expected
    comparison = calculate_comparison(geo_scores_dict)

    # 5. Build Response
    return AnalysisResponse(
        status="success",
        face_type=result.get("type", "Unknown"),
        description=result.get("description", "No description available."),
        geometric_scores=GeometricScores(**geo_scores_dict),
        impression_scores=imp_scores,
        matrix_coordinates=matrix_coords,
        comparison=comparison,
    )


# Root endpoint for health check
@app.get("/")
def health_check():
    return {"status": "ok", "message": "Face Diagnosis API is running."}
