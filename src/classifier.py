from typing import Dict, Any, Optional
from dataclasses import dataclass
# math unused


@dataclass
class DiagnosisResult:
    face_type: str
    description: str
    scores: Dict[str, float]
    coordinates: Dict[str, float]
    details: Dict[str, Any]  # Detailed Z-scores


class DiagnosisClassifier:
    """
    統計的基準値(Z-score)を用いた顔診断クラス。
    """

    # 日本人女性/男性の基準値データ (ユーザー提供)
    STANDARDS = {
        "female": {
            "aspect_ratio": {"mean": 1.188, "std": 0.038},  # 顔の縦横比
            "eye_ratio": {"mean": 0.451, "std": 0.012},  # 目の間隔
            "lower_ratio": {"mean": 0.470, "std": 0.014},  # 下顔面比率
        },
        "male": {
            "aspect_ratio": {"mean": 1.208, "std": 0.030},
            "eye_ratio": {"mean": 0.445, "std": 0.010},
            "lower_ratio": {"mean": 0.493, "std": 0.011},
        },
    }

    def __init__(self):
        # 4タイプの定義（Y軸はImpression/Shapeで決定する想定）
        # X軸(Generation)はZスコアで決定
        pass

    def calculate_z_score(self, val: float, mean: float, std: float) -> float:
        if std == 0:
            return 0.0
        return (val - mean) / std

    def classify(
        self,
        geometric_metrics: Dict[str, float],
        impression_scores: Dict[str, float],
        gender: str = "female",
    ) -> DiagnosisResult:
        """
        Args:
            geometric_metrics: Raw measured ratios {'aspect_ratio', 'eye_ratio', 'lower_ratio'}
            impression_scores: CLIP scores {'Cute_vs_Cool', 'Soft_vs_Sharp'}
            gender: 'female' or 'male'
        """
        if gender not in self.STANDARDS:
            gender = "female"  # Default

        stds = self.STANDARDS[gender]

        # 1. Calculate Z-scores
        z_aspect = self.calculate_z_score(
            geometric_metrics.get("aspect_ratio", stds["aspect_ratio"]["mean"]),
            stds["aspect_ratio"]["mean"],
            stds["aspect_ratio"]["std"],
        )

        z_eye = self.calculate_z_score(
            geometric_metrics.get("eye_ratio", stds["eye_ratio"]["mean"]),
            stds["eye_ratio"]["mean"],
            stds["eye_ratio"]["std"],
        )

        z_lower = self.calculate_z_score(
            geometric_metrics.get("lower_ratio", stds["lower_ratio"]["mean"]),
            stds["lower_ratio"]["mean"],
            stds["lower_ratio"]["std"],
        )

        # 2. X-axis: Generation Score (Adult vs Child)
        # "アスペクト比(70%) + 下顔面(30%)"
        # Positive Z -> Adult (面長, 顎長い)
        # Negative Z -> Child (丸顔, 顎短い)
        generation_score_z = (z_aspect * 0.7) + (z_lower * 0.3)

        # Determine Generation Rank
        gen_rank = "Average"
        if generation_score_z >= 1.0:
            gen_rank = "Strong Adult"
        elif generation_score_z >= 0.5:
            gen_rank = "Adult"
        elif generation_score_z <= -1.0:
            gen_rank = "Strong Child"
        elif generation_score_z <= -0.5:
            gen_rank = "Child"

        # 3. Y-axis: Shape Score (Curve vs Straight)
        # User requirement: "CLIPモデルで判定した「直線/曲線」スコアと組み合わせて"
        # CLIP: Soft_vs_Sharp (val is Sharp prob? Check impression.py implementation)
        # In implementation: Soft_vs_Sharp -> Value is Sharp(1.0). Soft(0.0).
        # We need -1.0(Straight/Sharp) to +1.0(Curve/Soft) OR vice versa.
        # Let's align with user matrix. Usually:
        # X: Child -> Adult
        # Y: Curve(Soft) -> Straight(Hard) OR Curve -> Straight.
        # User prompt example: "Adult x Curve = Feminine"
        # Let's define Y as Curve Score.

        imp_sharp_prob = impression_scores.get("Soft_vs_Sharp", 0.5)
        # imp_sharp_prob: 1.0 = Sharp(Straight), 0.0 = Soft(Curve)

        # We also have eye_ratio Z-score.
        # Large eye_ratio (Positive Z) -> Far eyes (Usually associated with Child/Curve?)
        # Small eye_ratio (Negative Z) -> Close eyes (Adult/Straight?)
        # User table:
        # +1.0 Z (Adult) -> Close eyes (求心)
        # -1.0 Z (Child) -> Far eyes (遠心)
        # Wait, the user table for "eye" says:
        # +1.0 Z (Adult-ish context? No, just Z score) -> "目が寄っている（求心顔）"
        # -1.0 -> "離れ目（遠心顔）"
        # BUT wait.
        # eye_ratio = dist / width.
        # Far eyes -> dist is LARGE -> Ratio is LARGE -> Z is POSITIVE.
        # Close eyes -> dist is SMALL -> Ratio is SMALL -> Z is NEGATIVE.
        # Contrast with user table:
        # User Table: "Z score +1.0 ... 目が寄っている（求心顔）"
        # This implies the user *inverted* the Z-sense for eyes, OR my eye_ratio definition is opposite,
        # OR the user meant "Score" not raw Z of ratio.
        # Let's normalize consistent with "Adultness" or "Sharpness".
        # Usually Centrifugal(Far) = Child/Soft. Centripetal(Close) = Adult/Sharp.
        # If Ratio is High (Far), Z is High.
        # User says Z>1.0 is "Centripetal (Close)". This contradicts Ratio=High (Far).
        # Hypothesis: User's Z-score for eyes is (Mean - Val) / Std? Or user considers "Eye Score" where High = Close.
        # Let's look at the "判定式 (Diagnosis Logic)" section again.
        # It says "アスペクト比のスコアと下顔面のスコア...". It doesn't explicitly use Eye Z for Generation.
        # But later "Adult x Curve ...".

        # Let's stick to the physical meaning first.
        # Eye Ratio High -> Far -> Soft/Child-like.
        # Eye Ratio Low -> Close -> Sharp/Adult-like.

        # Let's calculate Y (Shape) score.
        # Shape = (Impression_Softness + Eye_Centrifugal?)
        # Let's rely heavily on CLIP for Shape as before, but maybe modulate with Eye Z.
        # Sharpness (1.0) ~ Straight. Softness (0.0) ~ Curve.

        # Let's define a normalized coordinate system for output:
        # X: -1 (Child) to +1 (Adult) based on generation_score_z (clamped/scaled)
        # Y: -1 (Curve/Soft) to +1 (Straight/Sharp)

        # Map Z (-3 to +3) to (-1 to +1) roughly for coord
        x_coord = max(-1.0, min(1.0, generation_score_z / 2.0))

        # Y axis:
        # CLIP Sharpness (0.0=Soft, 1.0=Sharp).
        # Map to -1.0(Soft) to +1.0(Sharp)
        y_coord_clip = (imp_sharp_prob - 0.5) * 2.0

        y_coord = y_coord_clip  # Use CLIP primarily for shape/impression

        # Determine Final Type
        # Quadrants:
        # X<0, Y<0: Child x Soft = Cute
        # X<0, Y>0: Child x Sharp = Fresh
        # X>0, Y<0: Adult x Soft = Feminine/Elegant
        # X>0, Y>0: Adult x Sharp = Cool

        face_type = "Unknown"
        if x_coord < 0:
            if y_coord < 0:
                face_type = "Cute (子供×曲線)"
            else:
                face_type = "Fresh (子供×直線)"
        else:
            if y_coord < 0:
                face_type = "Feminine (大人×曲線)"
            else:
                face_type = "Cool (大人×直線)"

        return DiagnosisResult(
            face_type=face_type,
            description=f"Generation Rank: {gen_rank}",
            scores={
                "generation_z": generation_score_z,
                "aspect_z": z_aspect,
                "eye_z": z_eye,
                "lower_z": z_lower,
                "shape_score": y_coord,
            },
            coordinates={"x": x_coord, "y": y_coord},
            details={
                "gen_rank": gen_rank,
                "z_scores": {"aspect": z_aspect, "eye": z_eye, "lower": z_lower},
            },
        )
