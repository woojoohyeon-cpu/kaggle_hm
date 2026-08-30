# H&M Personalized Fashion Recommendations

## フォルダ構成

- `00_md/` — ドキュメント・メモ(コンペ概要、進め方など)
- `01_input/` — 生データ(articles.csv, customers.csv, transactions_train.csv, images/)。**Git管理外**
- `02_output/`
  - `submission/` — 提出用csv(タイムスタンプ+CVスコア付きで保存、上書きしない)
  - `models/` — 学習済みモデル(pkl等)
  - `features/` — 特徴量エンジニアリングの中間生成物
  - `logs/` — 学習ログ
- `Z_notebook/` — 役割ごとに1ファイルのNotebook(`01_eda.ipynb`, `02_baseline.ipynb`, `03_candidate_generation.ipynb`, `04_feature_engineering.ipynb`, `05_train_lgbm.ipynb` ...)。`common.py`は共通ユーティリティ
- `experiment_log.csv` — 実験記録(日時・特徴量・パラメータ・CV・Public LB・メモ)
- `.gitignore` — データ・モデル・提出物など大容量ファイルをGit管理外に

## 運用ルール

1. コード・Notebook・README・設定ファイルはGitで管理する。ファイル名に`v1`や日付を付けない。
2. Notebookは役割ごとに1ファイル。変更履歴はGitコミットで追う。
3. `submission_YYYYMMDD_HHMM_cv{score}.csv`、`model_YYYYMMDD_HHMM.pkl`のようにタイムスタンプ付きで保存し、上書きしない。
4. 実験内容は`experiment_log.csv`に都度追記する。
5. データセット・大容量ファイルはGit管理しない(`.gitignore`参照)。

## 命名例

- `02_output/submission/submission_20260725_1040_cv0.7231.csv`
- `02_output/models/model_20260725_1040.pkl`
- `02_output/features/feature_importance_20260725_1040.xlsx`
