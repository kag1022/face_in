import argparse
import os
import sys

# srcモジュールが見つかるようにパスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), "."))

from src.analyzer import FaceAnalyzer


def main():
    parser = argparse.ArgumentParser(description="Explainable Face Diagnosis System")
    parser.add_argument("image_path", nargs="?", help="Path to the face image file")
    parser.add_argument(
        "--gender",
        choices=["male", "female"],
        default="female",
        help="Gender for standardization (default: female)",
    )
    parser.add_argument("--mock", action="store_true", help="Force usage of mock mode")

    args = parser.parse_args()

    use_mock = args.mock
    image_path = args.image_path

    if not image_path:
        print("Usage: python main.py <image_path> [--gender female] [--mock]")
        image_path = "dummy_face.jpg"
        if not os.path.exists(image_path):
            import cv2
            import numpy as np

            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            cv2.imwrite(image_path, dummy_img)
            print(f"Created {image_path}")
        use_mock = True

    print(f"Initializing Face Analyzer (Mock: {use_mock})...")
    # Note: model_path defaults to ./face_landmarker.task, user must provide it for real run
    analyzer = FaceAnalyzer(use_mock=use_mock)

    print(f"Analyzing {image_path} as {args.gender}...")
    # Fix: analyzer.analyze_image currently doesn't accept gender.
    # analyzer.analyze_from_array DOES accept gender.
    # analyzer.analyze_image just calls analyze_from_array with default.
    # We should update analyze_image to accept gender OR call analyze_from_array manually here.
    # To avoid changing analyzer.py again, let's load image here.

    import cv2

    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image.")
        return

    result = analyzer.analyze_from_array(img, gender=args.gender)

    # 結果の表示
    print("\n" + "=" * 40)
    print("      SCIENTIFIC DIAGNOSIS RESULT      ")
    print("=" * 40)

    if result.get("error"):
        print(f"Error: {result['error']}")
    else:
        print(f"Face Type   : {result['type']}")
        print(f"Description : {result['description']}")
        print("-" * 40)

        details = result["details"]
        metrics = details["geometric_metrics"]
        z_scores = details["z_scores"]

        print("[Geometric Metrics & Z-Scores]")
        print(f"  Gender Standard: {args.gender}")
        print(
            f"  - Aspect Ratio : {metrics['aspect_ratio']:.3f} (Z: {z_scores['aspect']:+.2f})"
        )
        print(
            f"  - Eye Ratio    : {metrics['eye_ratio']:.3f} (Z: {z_scores['eye']:+.2f})"
        )
        print(
            f"  - Lower Ratio  : {metrics['lower_ratio']:.3f} (Z: {z_scores['lower']:+.2f})"
        )
        print(f"  => Generation Score (Z): {details['generation_score']:+.2f}")

        print("\n[Impression Scores (CLIP)]")
        for k, v in details["impression_scores"].items():
            print(f"  - {k:<20} : {v:.2f}")

        coords = details["matrix_coordinates"]
        print("-" * 40)
        print(
            f"Matrix Position: X={coords['x']:.2f} (Generation), Y={coords['y']:.2f} (Shape)"
        )


if __name__ == "__main__":
    main()
