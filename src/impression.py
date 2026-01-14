import numpy as np
from typing import List, Tuple, Dict
import os

# Optional imports for Mock mode compatibility
try:
    import torch
    import clip
    from PIL import Image
    import cv2

    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    print("Warning: torch/clip/cv2/PIL not found. Running in potential mock mode.")


class ImpressionScorer:
    """
    OpenAI CLIPモデルを使用して、顔画像が特定のテキスト（形容詞など）に
    どれくらい近いかをスコアリングするクラス。

    説明可能性の観点から、「確率（％）」で出力することを重視する。
    """

    def __init__(self, device: str = "cpu", model_name: str = "ViT-B/32"):
        self.device = device
        self.model_name = model_name
        self.model = None
        self.preprocess = None

        # 遅延ロード: 初回利用時にモデルを読み込む
        self._is_loaded = False

    def _load_model(self):
        if not self._is_loaded:
            if not IMPORTS_AVAILABLE:
                self.model = "mock"
                self._is_loaded = True
                return

            print(f"Loading CLIP model: {self.model_name} on {self.device}...")
            # 注意: 実際の実行環境にclipがインストールされている必要がある
            try:
                self.model, self.preprocess = clip.load(
                    self.model_name, device=self.device
                )
                self._is_loaded = True
            except (ImportError, NameError) as e:
                print(f"Warning: clip module load failed ({e}). Using mock scores.")
                self.model = "mock"

    def score(
        self, image_np: np.ndarray, text_pairs: List[Tuple[str, str]]
    ) -> Dict[str, float]:
        """
        画像に対して、対となるテキストペア（例: ["Cute", "Cool"]）のどちらに近いかを判定する。

        Args:
            image_np: OpenCV形式 (BGR) のnumpy画像配列
            text_pairs: 比較したい単語ペアのリスト。
                        例: [("Cute", "Cool"), ("Feminine", "Masculine")]

        Returns:
            Dict[str, float]: Keyは "Cute_vs_Cool" のような形式、
                              Valueは後者("Cool")の確率スコア (0.0〜1.0)
                              0.0に近いほど前者、1.0に近いほど後者。
        """
        self._load_model()

        if self.model == "mock":
            return self._mock_score(text_pairs)

        # 画像の前処理
        # OpenCV (BGR) -> PIL (RGB)
        try:
            image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)

            image_input = self.preprocess(image_pil).unsqueeze(0).to(self.device)

            results = {}

            with torch.no_grad():
                for text1, text2 in text_pairs:
                    # プロンプトエンジニアリング: 単語そのものではなく文章にする
                    prompts = [
                        f"A photo of a {text1} face",
                        f"A photo of a {text2} face",
                    ]
                    text_inputs = clip.tokenize(prompts).to(self.device)

                    # 特徴量抽出
                    image_features = self.model.encode_image(image_input)
                    text_features = self.model.encode_text(text_inputs)

                    # 正規化
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)

                    # 類似度計算 (Cosine Similarity)
                    similarity = (100.0 * image_features @ text_features.T).softmax(
                        dim=-1
                    )

                    # values: [prob_text1, prob_text2]
                    probs = similarity[0].cpu().numpy()

                    # Key: "Text1_vs_Text2", Value: prob_text2 (0.0-1.0)
                    key = f"{text1}_vs_{text2}"
                    val = float(probs[1])
                    results[key] = val
        except Exception as e:
            print(f"Error during CLIP inference: {e}. Falling back to mock.")
            return self._mock_score(text_pairs)

        return results

    def _mock_score(self, text_pairs: List[Tuple[str, str]]) -> Dict[str, float]:
        """依存ライブラリがない場合のモック動作"""
        results = {}
        for t1, t2 in text_pairs:
            # ランダムにスコアを返す (0.4 - 0.6)
            results[f"{t1}_vs_{t2}"] = 0.5 + (np.random.rand() - 0.5) * 0.2
        return results
