# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

Kaggle「H&M Personalized Fashion Recommendations」コンペのリポジトリ。過去の購買履歴・顧客属性・商品メタデータ(テキスト・画像)から、各顧客に最も関連性の高い商品を12個推薦する。評価指標はMAP@12。

主なアプローチは、候補生成(candidate generation)→ ランキングモデルによる再ランキングの2段階構成。時系列データのため検証はランダム分割ではなく直近期間を切り出すバックテスト設計が必須。

## フォルダ構成

- `00_md/` — ドキュメント・メモ(コンペ概要、進め方など)
- `01_input/` — 生データ。**Git管理外**(`.gitignore`参照)
  - `articles.csv`(25列, 商品マスタ), `customers.csv`(7列, 顧客マスタ), `transactions_train.csv`(5列, 購買履歴 約3,100万行), `images/`(商品画像 約105,100枚, 1166×1750px)
- `02_output/`
  - `submission/` — 提出用csv
  - `models/` — 学習済みモデル
  - `features/` — 特徴量エンジニアリングの中間生成物
  - `logs/` — 学習ログ
- `Z_notebook/` — 役割ごとに1ファイルのNotebook(`01_eda.ipynb` 済み、以降 `02_feature_engineering.ipynb`, `03_train_lgbm.ipynb` ...)
- `experiment_log.csv` — 実験記録

## バージョン管理・成果物の運用ルール

これらのルールは厳守すること。新しいコードやNotebookを追加・変更する際は必ず従う。

- コード・Notebook・README・設定ファイルはGitで管理する。ファイル名に`v1`や日付を付けない(履歴はGitコミットで追う)。
- Notebookは役割ごとに1ファイル(`01_eda.ipynb`, `02_feature_engineering.ipynb`, `03_train_lgbm.ipynb` ...)。連番を崩さない。
- Notebookをコミットする前は、可能な限りセル出力をクリアしてからコミットする(diffノイズを避けるため)。
- 提出物・モデル・分析結果は上書きせず、タイムスタンプ(+スコア)を付けて保存する。
  - 例: `02_output/submission/submission_YYYYMMDD_HHMM_cv{score}.csv`
  - 例: `02_output/models/model_YYYYMMDD_HHMM.pkl`
  - 例: `02_output/features/feature_importance_YYYYMMDD_HHMM.xlsx`
- 実験(特徴量・パラメータ・CV・Public LB・気づき)は都度 `experiment_log.csv` に追記する。列: `run_id, timestamp, notebook, description, features, params, cv_score, public_lb, submission_file, model_file, notes`
- `01_input/`(生データ)や `02_output/` 配下の大容量成果物(submission csv, モデル, features, ログ)はGit管理しない。`.gitignore` に既定済み。新しい大容量出力先を追加した場合は `.gitignore` も更新する。
- `kaggle.json`(Kaggle API認証情報)はコミットしない。

## テーブルの結合関係

- `transactions_train.article_id` ↔ `articles.article_id`
- `transactions_train.customer_id` ↔ `customers.customer_id`

## データ取得

Kaggle APIトークン(`kaggle.json`)が必要。`kaggle competitions download -c h-and-m-personalized-fashion-recommendations` でダウンロードし、`01_input/` に解凍する。
