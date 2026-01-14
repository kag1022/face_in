from pydantic import BaseModel
from typing import List


class DiagnosisMeta(BaseModel):
    process_time: float
    image_width: int
    image_height: int


class DiagnosisInfo(BaseModel):
    type: str  # e.g. "Elegant"
    generation_score: float  # X-axis
    shape_score: float  # Y-axis


class ScoreItem(BaseModel):
    value: float
    z_score: float
    verdict: str


class Scores(BaseModel):
    aspect_ratio: ScoreItem
    eye_ratio: ScoreItem
    lower_ratio: ScoreItem
    # impression_softness: ScoreItem # Optional if needed, but user requirement showed "aspect_ratio", "eye_ratio"


class Landmarks(BaseModel):
    face_box: List[int]  # [x, y, w, h]
    chin_line: List[List[int]]  # [[x,y], [x,y]]
    left_eye_line: List[List[int]]
    right_eye_line: List[List[int]]
    all_points: List[List[int]]


class AnalysisResponse(BaseModel):
    diagnosis: DiagnosisInfo
    scores: Scores
    landmarks: Landmarks
    meta: DiagnosisMeta
