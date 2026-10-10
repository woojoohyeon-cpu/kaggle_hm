"""
ローカル(Macのvenv環境)で実行するスクリプト。
Colabに巨大な train_features_*.parquet を丸ごと運ばずに済むよう、
multi_week_eval.load_train_categories() が必要とする category 対応表だけを
小さいpickleとして書き出す。

使い方(ローカルのリポジトリルートで):
    cd ~/Desktop/01_coding/Kaggle/H&M
    python Z_notebook/local_extract_categories.py

出力:
    02_output/models/train_categories_0906.pkl
    02_output/models/train_categories_0917.pkl
これと model_20260906_0924_ranker.pkl / model_20260917_2337_ranker.pkl の
計4ファイルだけをColabにアップロードすればよい(Google Driveに置いて
drive.mount()経由で読むのが手軽)。
"""

import os
import pickle
import sys

sys.path.append(os.path.abspath("Z_notebook"))

from common import FEATURES_DIR, MODELS_DIR  # noqa: E402
from multi_week_eval import load_model, load_train_categories  # noqa: E402

TARGETS = [
    ("0906", "model_20260906_0924_ranker.pkl", "train_features_20260906_0910.parquet"),
    ("0917", "model_20260917_2337_ranker.pkl", "train_features_20260917_2324.parquet"),
]

for label, model_file, train_feat_file in TARGETS:
    model_path = os.path.join(MODELS_DIR, model_file)
    train_feat_path = os.path.join(FEATURES_DIR, train_feat_file)

    if not os.path.exists(model_path):
        print(f"[skip] {model_path} が見つかりません")
        continue
    if not os.path.exists(train_feat_path):
        print(f"[skip] {train_feat_path} が見つかりません")
        continue

    payload = load_model(model_path)
    categories = load_train_categories(train_feat_path, payload["cat_cols"])

    out_path = os.path.join(MODELS_DIR, f"train_categories_{label}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(categories, f)
    print(f"[ok] {out_path} を書き出しました (cat_cols={payload['cat_cols']})")

print("\nColabには以下の4ファイルだけをアップロードすればOKです:")
for label, model_file, _ in TARGETS:
    print(f"  - 02_output/models/{model_file}")
    print(f"  - 02_output/models/train_categories_{label}.pkl")