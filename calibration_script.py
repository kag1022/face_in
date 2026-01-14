import cv2
import mediapipe as mp
import numpy as np
import glob
import os
import csv
from tqdm import tqdm

# MediaPipe Tasks API設定
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- 設定 ---
DATASET_ROOT = "./dataset"  # 画像フォルダのルート
OUTPUT_CSV = "face_standards.csv"

# MediaPipe Tasks API設定


# モデルファイルのパス (同じディレクトリにある前提)
MODEL_PATH = "./face_landmarker.task"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1,
    min_face_detection_confidence=0.5,
)
detector = vision.FaceLandmarker.create_from_options(options)


def calc_distance(p1, p2):
    """2点間のユークリッド距離を計算"""
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def analyze_face(image_path):
    """1枚の画像から顔の特徴比率を計算する"""
    image = cv2.imread(image_path)
    if image is None:
        return None

    # MediaPipe Image形式に変換
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    )

    # 推論実行
    detection_result = detector.detect(mp_image)

    if not detection_result.face_landmarks:
        return None

    # 最初の顔のランドマークを取得
    landmarks = detection_result.face_landmarks[0]

    # --- 重要なランドマークID (MediaPipe Face Mesh) ---
    # 輪郭: 上=10, 下=152, 左=234, 右=454
    # 目: 左目中心=468, 右目中心=473
    # 眉間: 9
    # 鼻下: 2

    top = landmarks[10]
    bottom = landmarks[152]
    left_cheek = landmarks[234]
    right_cheek = landmarks[454]
    left_eye = landmarks[468]
    right_eye = landmarks[473]
    nose_tip = landmarks[4]  # 鼻頭

    # 1. 顔の幅と高さ (ピクセル単位ではなく相対座標での距離)
    face_width = calc_distance(left_cheek, right_cheek)
    face_height = calc_distance(top, bottom)

    # 【指標A】アスペクト比 (縦/横)
    # 値が大きいほど「面長(大人)」、小さいほど「丸顔(子供)」
    aspect_ratio = face_height / face_width if face_width > 0 else 0

    # 【指標B】目の位置のバランス (求心・遠心)
    # (両目の距離) / (顔の幅)
    # 値が大きいほど「目が離れている(子供)」、小さいほど「寄っている(大人)」
    eye_dist = calc_distance(left_eye, right_eye)
    eye_ratio = eye_dist / face_width if face_width > 0 else 0

    # 【指標C】顔の下半分の長さ (中顔面〜下顔面)
    # (鼻下〜顎) / (顔の高さ)
    # 値が小さいほど「顎が短い(子供/キュート)」、大きいほど「顎が長い(大人/クール)」
    # ※厳密には眉間〜鼻下などの比率も重要ですが、簡易指標として採用
    nose_to_chin = calc_distance(nose_tip, bottom)
    lower_face_ratio = nose_to_chin / face_height if face_height > 0 else 0

    return {
        "aspect_ratio": aspect_ratio,
        "eye_ratio": eye_ratio,
        "lower_face_ratio": lower_face_ratio,
    }


def main():
    print(f"[{DATASET_ROOT}] 内の画像を解析中...")

    # 男女別にリストを用意
    categories = ["male", "female"]
    stats_results = []

    for gender in categories:
        image_paths = glob.glob(os.path.join(DATASET_ROOT, gender, "*.*"))
        valid_data = []

        print(f"\nProcessing {gender} ({len(image_paths)} images)...")

        for path in tqdm(image_paths):
            # 拡張子チェック
            if not path.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue

            data = analyze_face(path)
            if data:
                valid_data.append(data)

        if not valid_data:
            print(f"Warning: No valid faces found for {gender}.")
            continue

        # 統計計算
        aspects = [d["aspect_ratio"] for d in valid_data]
        eyes = [d["eye_ratio"] for d in valid_data]
        lowers = [d["lower_face_ratio"] for d in valid_data]

        stats = {
            "gender": gender,
            "count": len(valid_data),
            # 平均値 (Mean)
            "mean_aspect_ratio": np.mean(aspects),
            "mean_eye_ratio": np.mean(eyes),
            "mean_lower_face_ratio": np.mean(lowers),
            # 標準偏差 (Std Dev) - ばらつき具合
            "std_aspect_ratio": np.std(aspects),
            "std_eye_ratio": np.std(eyes),
            "std_lower_face_ratio": np.std(lowers),
        }
        stats_results.append(stats)

    # 結果の表示とCSV保存
    print("\n" + "=" * 50)
    print("ANALYSIS RESULT (基準値)")
    print("=" * 50)

    # CSV書き込み
    with open(OUTPUT_CSV, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=stats_results[0].keys())
        writer.writeheader()

        for res in stats_results:
            writer.writerow(res)
            print(f"\n--- {res['gender'].upper()} ---")
            print(f"サンプル数: {res['count']}")
            print(
                f"縦横比(大人度) : 平均 {res['mean_aspect_ratio']:.4f} (±{res['std_aspect_ratio']:.4f})"
            )
            print(
                f"目の距離(遠心度): 平均 {res['mean_eye_ratio']:.4f} (±{res['std_eye_ratio']:.4f})"
            )
            print(
                f"下顔面比(顎長さ): 平均 {res['mean_lower_face_ratio']:.4f} (±{res['std_lower_face_ratio']:.4f})"
            )

    print("\n" + "=" * 50)
    print(f"結果を {OUTPUT_CSV} に保存しました。")
    print("この数値をアプリの判定ロジックに使用してください。")


if __name__ == "__main__":
    main()
