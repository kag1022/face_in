import numpy as np
from typing import Dict, Any, Optional, Tuple


class GeometricFeatureExtractor:
    """
    MediaPipe Face Landmark (478 points) を使用して
    幾何学的特徴(Raw Ratios)を計算するクラス。

    注意: 入力は MediaPipe の NormalizedLandmark (x, y) のリストまたはオブジェクトを想定。
    """

    def __init__(self):
        # Indices for MediaPipe Face Mesh
        self.IDX_TOP = 10  # 額の頂点
        self.IDX_BOTTOM = 152  # 顎先
        self.IDX_LEFT_CHEEK = 234  # 左頬 (画面向かって左、本人にとって右?) -> MediaPipe layout check needed.
        # 234 is often Left side of image (Right cheek of person)
        # 454 is Right side of image (Left cheek of person)
        self.IDX_RIGHT_CHEEK = 454
        self.IDX_LEFT_EYE = 468  # 左目中心 (Iris)
        self.IDX_RIGHT_EYE = 473  # 右目中心 (Iris)
        self.IDX_NOSE_TIP = 4  # 鼻頭

    def calc_distance(self, p1, p2) -> float:
        """2点間のユークリッド距離を計算"""
        # p1, p2 are expected to have .x and .y attributes (normalized 0-1)
        # アスペクト比などの比率計算なので、ピクセル座標変換しなくても相対比は同じだが、
        # 縦横のスケールが違う場合(長方形画像)はアスペクト比補正が必要。
        # ここでは入力が正規化座標(0-1)であることを前提とし、アスペクト比は画像サイズに依存せず
        # "形状"としての距離を測るため、本来は画素数掛けるべきだが、
        # User script uses simple sqrt((x-x)^2 + (y-y)^2).
        # THIS IS PROBLEMATIC if image is not square. Normalized coordinates distort distance.
        # User's calibration script:
        #   face_width = calc_distance(left, right)
        #   aspect_ratio = face_height / face_width
        # If user script assumes square image or doesn't care, I should match or improve.
        # Ideally, we multiply by (width, height).
        # But let's follow standard "Raw" extraction.
        # I will accept width/height args to correct aspect.
        return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    def analyze(
        self, landmarks, image_shape: Optional[Tuple[int, int]] = None
    ) -> Dict[str, float]:
        """
        Args:
            landmarks: List of NormalizedLandmark (must have x, y fields) or similar object.
                       Can be dictionary or object.
            image_shape: (height, width) of image. Used to correct aspect ratio of normalized coords.
        """
        if not landmarks:
            return {}

        # Helper to get point with aspect correction
        def get_point(idx):
            pt = landmarks[idx]
            if image_shape:
                h, w = image_shape
                return type("Point", (), {"x": pt.x * w, "y": pt.y * h})()
            return pt

        top = get_point(self.IDX_TOP)
        bottom = get_point(self.IDX_BOTTOM)
        left_cheek = get_point(self.IDX_LEFT_CHEEK)
        right_cheek = get_point(self.IDX_RIGHT_CHEEK)
        left_eye = get_point(self.IDX_LEFT_EYE)
        right_eye = get_point(self.IDX_RIGHT_EYE)
        nose_tip = get_point(self.IDX_NOSE_TIP)

        # 1. Face Width & Height
        face_width = self.calc_distance(left_cheek, right_cheek)
        face_height = self.calc_distance(top, bottom)

        if face_width == 0 or face_height == 0:
            return {"aspect_ratio": 0, "eye_ratio": 0, "lower_ratio": 0}

        # Ratio A: Aspect Ratio (Height / Width)
        # Large -> Long/Adult
        aspect_ratio = face_height / face_width

        # Ratio B: Eye Ratio (Eye Dist / Face Width)
        # Large -> Far/Child (User's script comment: "値が大きいほど「目が離れている(子供)」")
        eye_dist = self.calc_distance(left_eye, right_eye)
        eye_ratio = eye_dist / face_width

        # Ratio C: Lower Face Ratio (Nose-Chin / Height)
        # Large -> Long Chin / Adult
        nose_to_chin = self.calc_distance(nose_tip, bottom)
        lower_ratio = nose_to_chin / face_height

        return {
            "aspect_ratio": aspect_ratio,
            "eye_ratio": eye_ratio,
            "lower_ratio": lower_ratio,
        }
