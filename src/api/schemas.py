from pydantic import BaseModel, Field
from typing import Dict, Optional


class GeometricScores(BaseModel):
    aspect_ratio_score: float = Field(
        ..., description="0.0(Round/Short) - 1.0(Long/Oval)"
    )
    eye_spacing_score: float = Field(
        ..., description="0.0(Far/Centrifugal) - 1.0(Close/Centripetal)"
    )
    gravity_score: float = Field(..., description="0.0(Lower/Cute) - 1.0(Upper/Mature)")


class ComparisonData(BaseModel):
    diff_aspect_ratio: float = Field(
        ..., description="Difference from average aspect ratio"
    )
    diff_eye_spacing: float = Field(
        ..., description="Difference from average eye spacing"
    )
    diff_gravity: float = Field(
        ..., description="Difference from average gravity score"
    )


class AnalysisResponse(BaseModel):
    status: str = Field("success", description="Response status")
    face_type: str = Field(
        ..., description="Diagnosed face type (e.g., 'Fresh', 'Cute')"
    )
    description: str = Field(..., description="Detailed diagnosis text")

    geometric_scores: GeometricScores
    impression_scores: Dict[str, float] = Field(
        ..., description="Impression scores from CLIP (e.g. {'Cute_vs_Cool': 0.7})"
    )

    matrix_coordinates: Dict[str, float] = Field(
        ..., description="Coordinates for matrix plotting {'x': float, 'y': float}"
    )
    comparison: ComparisonData
