import os
import cv2
import numpy as np
from typing import Dict, Any

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    print("Warning: mediapipe not found. Using mock data.")

from .geometric import GeometricFeatureExtractor
from .impression import ImpressionScorer
from .classifier import DiagnosisClassifier


class FaceAnalyzer:
    """
    顔診断システムのメインFacadeクラス。
    MediaPipe + CLIP + Z-Score Logic
    """

    def __init__(
        self, use_mock: bool = False, model_path: str = "./face_landmarker.task"
    ):
        self.use_mock = use_mock or (not MP_AVAILABLE)

        self.geometric = GeometricFeatureExtractor()
        self.impression = ImpressionScorer()
        self.classifier = DiagnosisClassifier()

        self.detector = None

        if not self.use_mock:
            if os.path.exists(model_path):
                try:
                    base_options = python.BaseOptions(model_asset_path=model_path)
                    options = vision.FaceLandmarkerOptions(
                        base_options=base_options,
                        output_face_blendshapes=False,
                        output_facial_transformation_matrixes=False,
                        num_faces=1,
                    )
                    self.detector = vision.FaceLandmarker.create_from_options(options)
                    print(f"MediaPipe FaceLandmarker loaded from {model_path}")
                except Exception as e:
                    print(f"Failed to load MediaPipe model: {e}. Switching to mock.")
                    self.use_mock = True
            else:
                print(f"Model file {model_path} not found. Switching to mock.")
                self.use_mock = True

    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        if not os.path.exists(image_path):
            return {"error": "File not found"}

        if cv2 is None:
            # Fallback if cv2 is missing (very unlikely if mediapipe present)
            img = np.zeros((640, 640, 3), dtype=np.uint8)
        else:
            img = cv2.imread(image_path)

        if img is None:
            return {"error": "Failed to load image"}

        return self.analyze_from_array(img)

    def analyze_from_array(
        self, img_bgr: np.ndarray, gender: str = "female"
    ) -> Dict[str, Any]:
        """
        img_bgr: OpenCV BGR image
        """
        h, w = img_bgr.shape[:2]

        # 1. Detect Landmarks (MediaPipe)
        landmarks = self._get_landmarks(img_bgr)
        if landmarks is None:
            return {"error": "No face detected"}

        # 2. Geometric Analysis (Raw Ratios)
        # Pass image shape (h, w) for aspect correction in geometric calc
        geo_metrics = self.geometric.analyze(landmarks, image_shape=(h, w))

        # 3. Impression Analysis (CLIP)
        text_pairs = [("Cute", "Cool"), ("Soft", "Sharp")]
        imp_scores = self.impression.score(img_bgr, text_pairs)

        # 4. Classification (Z-Score)
        result = self.classifier.classify(geo_metrics, imp_scores, gender=gender)

        # Serialize landmarks for API
        serialized_landmarks = self._serialize_landmarks(landmarks, (w, h))

        return {
            "status": "success",
            "type": result.face_type,
            "description": result.description,
            "details": {
                "geometric_metrics": geo_metrics,  # Raw Ratios
                "impression_scores": imp_scores,  # CLIP probs
                "z_scores": result.details["z_scores"],
                "matrix_coordinates": result.coordinates,
                "generation_score": result.scores["generation_z"],
                "shape_score": result.scores["shape_score"],
            },
            "landmarks": serialized_landmarks,
        }

    def _get_landmarks(self, img_bgr: np.ndarray):
        if self.use_mock or self.detector is None:
            return self._mock_landmarks()

        # MediaPipe expects RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        detection_result = self.detector.detect(mp_image)

        if not detection_result.face_landmarks:
            return None

        # Return the first face's landmarks (list of NormalizedLandmark)
        return detection_result.face_landmarks[0]

    def _mock_landmarks(self):
        """Create a mock object with x, y attributes simulating normalized landmarks"""

        class Point:
            def __init__(self, x, y):
                self.x, self.y = x, y

        # Mocking generic positions
        # Indices: 10(Top), 152(Bot), 234(L), 454(R), 468(LE), 473(RE), 4(Nose)
        # Create a list where indices enable access
        # Max index is ~478.
        lms = [Point(0, 0)] * 500

        # Manually set key points to reasonable values
        lms[10] = Point(0.5, 0.2)  # Top
        lms[152] = Point(0.5, 0.8)  # Bottom
        lms[234] = Point(0.3, 0.5)  # Left Cheek
        lms[454] = Point(0.7, 0.5)  # Right Cheek
        lms[468] = Point(0.4, 0.45)  # Left Eye
        lms[473] = Point(0.6, 0.45)  # Right Eye
        lms[4] = Point(0.5, 0.6)  # Nose Tip

        return lms

    def _serialize_landmarks(self, landmarks, image_shape):
        """
        Convert MediaPipe landmarks to JSON-friendly format.
        Args:
            landmarks: list of NormalizedLandmark objects (x, y)
            image_shape: tuple (width, height) used for calculating pixel coordinates

        Returns:
            Dict containing key lines/points.
        """
        w, h = image_shape

        def to_px(pt):
            return [int(pt.x * w), int(pt.y * h)]

        # Extract all points as flat list for "face_box" calculation or general use
        all_points = [to_px(pt) for pt in landmarks]

        # Calculate bounding box [x, y, w, h] from all points
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        if not xs:
            return {}

        FACE_BOX = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]

        # Essential Lines for Visualization
        # Chin Line: 234 -> ... -> 152 -> ... -> 454 (MediaPipe indices)
        # Simplified chin line for visualization (just using key points)
        # 234(L), 176, 148, 152(Bot), 377, 400, 454(R) are some jawline indices
        # Let's just return key points for simple drawing or the full set if needed.
        # User requested: "chin_line", "eye_line".

        # Eye Lines: Left 33-133, Right 362-263 (approx) or just the 468/473 centers
        # We'll just provide the key contour points if we wanted to be perfect,
        # but for now let's provided indices-based construction if possible.
        # Since 'landmarks' is just a list, we can return the whole list or specific subsets.
        # Let's return the key semantic lines.

        # Indices for Jawline (approximate path)
        jaw_indices = [
            234,
            93,
            132,
            58,
            172,
            136,
            150,
            149,
            176,
            148,
            152,
            377,
            400,
            378,
            379,
            365,
            397,
            288,
            361,
            323,
            454,
        ]
        # Indices for Left Eye
        left_eye_indices = [33, 160, 158, 133, 153, 144]  # loop
        # Indices for Right Eye
        right_eye_indices = [362, 385, 387, 263, 373, 380]  # loop

        def get_line(indices):
            line = []
            for idx in indices:
                if idx < len(landmarks):
                    line.append(to_px(landmarks[idx]))
            return line

        return {
            "face_box": FACE_BOX,
            "chin_line": get_line(jaw_indices),
            "left_eye_line": get_line(left_eye_indices),
            "right_eye_line": get_line(right_eye_indices),
            "all_points": all_points,  # Full 478 points if frontend wants to draw mesh
        }
