# パイプライン構築ログ / 最新版ステータス(2026-08-14 時点)

H&M Personalized Fashion Recommendations の実装記録。
`hypothesis_driven_design.md`(Phase 1〜2: EDA・仮説検証の詳細)と対になるドキュメントで、
候補生成 → モデル学習 → 提出までの実装内容・実施順・詰まったポイントをまとめている。

**新しいセッションでこの続きに着手する場合は、このファイルを最初に読むこと。**

---

## 0. 実施ロードマップ(何をどの順でやったか)

```mermaid
flowchart TD
    subgraph P12["Phase 1-2 : データ理解 → 初回提出"]
        direction LR
        A1["01_eda 07/25・08/06<br/>EDA + 仮説 H1〜H11 検証"]
        A2["02_baseline 08/07<br/>自己リピート → トレンド人気<br/>MAP@12 0.0248 / LB 0.02000"]
        A1 --> A2
    end

    subgraph P34["Phase 3-4 : 2段構成の初版(08/13 午前〜午後)"]
        direction LR
        B1["03 候補 A〜D を単体監査<br/>結合 Recall 上限 0.082"]
        B2["04 08/13 08:59<br/>4ソース × 学習2週"]
        B3["05 14:18 Ranker vs Classifier<br/>MAP@12 0.0322 / 0.0203"]
        B4["06 14:41 提出<br/>LB 0.02527 「+26%」"]
        B1 --> B2 --> B3 --> B4
    end

    subgraph P4b["Phase 4' : 候補追加 + 学習週の拡張(08/13 夕方〜夜)"]
        direction LR
        C1["16:05-16:50<br/>features.py / build_one_cutoff.py<br/>に切り出し・別プロセス化"]
        C2["04 再実行 17:49<br/>5ソース「+E 共起」× 学習4週"]
        C3["05 再学習 19:56<br/>MAP@12 0.0332 / Cls 0.0307"]
        C4["06 20:47 提出<br/>LB 0.02556"]
        C5["03 再訪 21:08-21:21<br/>F テキスト類似を追加監査<br/>6ソース結合 Recall 0.093"]
        C1 --> C2 --> C3 --> C4 --> C5
    end

    subgraph P5["Phase 5 : ハイパラ探索(08/14 午前)"]
        direction LR
        D1["07:19 training_data.py に<br/>OOM対策ロードを共通化"]
        D2["07 08:33 optuna 20試行<br/>MAP@12 0.0336"]
        D3["06 08:55 提出<br/>LB 0.02570 / Private 0.02572"]
        D1 --> D2 --> D3
    end

    subgraph NEXT["次の一手(未着手)"]
        direction LR
        N1["F を build_candidate_pool に統合<br/>→ 04→05→06 を通しで再実行"]
        N2["Ranker × Classifier<br/>アンサンブル"]
        N3["複数期間バックテストで<br/>安定性を確認"]
    end

    A2 --> B1
    B4 --> C1
    C5 --> D1
    D3 --> N1
```

補足: 03 は「候補ソースの価値検証専用」のため、本番パイプラインの経路には入っていない。
新しい候補ソースを思いついたとき(E, F を追加したときのように)に戻ってきて再実行する位置づけ。

---

## 1. 本番パイプラインのデータフロー(現在の構成)

```mermaid
flowchart TD
    RAW["01_input/<br/>transactions_train.csv<br/>articles.csv / customers.csv"]
    LITE["02_output/features/<br/>trans_lite.parquet<br/>customer_id を整数キー化"]
    CON["common.get_con<br/>duckdb: trans / articles / cust_map"]

    subgraph CAND["Stage 1 : 候補生成 candidates.py"]
        direction TB
        SA["A 自己リピート"]
        SB["B トレンド人気"]
        SC["C 同カテゴリ / 同色"]
        SD["D 協調フィルタリング ALS"]
        SE["E アイテム共起"]
        SF["F テキスト類似<br/>※ 未統合"]
        POOL["build_candidate_pool<br/>A〜E を統合 + ソースフラグ"]
        SA --> POOL
        SB --> POOL
        SC --> POOL
        SD --> POOL
        SE --> POOL
        SF -. 未接続 .-> POOL
    end

    FEAT["features.build_feature_table<br/>user / item / user-item / 時間 特徴量 + ラベル"]
    SUB1["build_one_cutoff.py<br/>cutoff ごとに別プロセス実行"]
    PARQ["02_output/features/<br/>train_features_*.parquet 学習4週<br/>val_features_*.parquet 検証1週"]
    LOAD["training_data.load_train_val<br/>ダウンキャスト + 負例サンプリング"]
    MODEL["05 / 07<br/>LGBMRanker LambdaRank<br/>group = customer_idx x cutoff"]
    PKL["02_output/models/<br/>model_*_ranker.pkl"]
    PRED["06_predict_submit.ipynb<br/>アクティブ約40万人はモデル<br/>非アクティブ約100万人はトレンド人気"]
    CSV["02_output/submission/<br/>submission_*.csv"]

    RAW --> LITE --> CON --> CAND
    POOL --> FEAT
    FEAT --> SUB1 --> PARQ --> LOAD --> MODEL --> PKL --> PRED --> CSV
    CON --> FEAT
```

---

## 2. 結果推移(詳細は `experiment_log.csv`)

| 段階 | 説明 | バックテストMAP@12 | Public LB | Private LB |
|---|---|---|---|---|
| 02_baseline | 自己リピート+トレンド人気のルールベースハイブリッド | 0.0248 | 0.02000 | 0.02013 |
| 05(初回) | LGBMRanker、候補4ソース・学習2週 | 0.0322 | 0.02527 | 0.02505 |
| 05(共起+多週) | 候補5ソース(+アイテム共起)・学習4週 | 0.0332 | 0.02556 | 0.02574 |
| 07(optuna) | 上記+optuna20試行でのハイパラ探索 | 0.0336 | 0.02570 | 0.02572 |

baseline比で Public LB +28.5%。02_baselineとLGBMRankerとの間の伸びが最も大きく(+26%)、
そこから先(候補ソース追加・ハイパラ調整)は一段小さい伸び(各+1%前後)という収穫逓減の傾向。

---

## 3. ノートブック/モジュールの役割

- `01_eda.ipynb` — EDA、H1〜H11の仮説検証(詳細は `hypothesis_driven_design.md`)
- `02_baseline.ipynb` — 自己リピート(H1)+トレンド人気(H10)のルールベースハイブリッド。初回提出用
- `03_candidate_generation.ipynb` — 候補ソースA〜Fそれぞれの単体Recall@12/MAP@12を監査。
  **本番パイプラインには使わず、新しい候補ソースを追加する前の価値検証専用**
- `04_feature_engineering.ipynb` — 5つのcutoff(学習4週+検証1週)分の「候補×特徴量×ラベル」テーブルを作り、
  `train_features_*.parquet` / `val_features_*.parquet` に保存。
  cutoffごとに `build_one_cutoff.py` を**サブプロセスとして**呼ぶ(理由は6節のOOM対策参照)
- `05_train_lgbm.ipynb` — LGBMRanker(LambdaRank)とLGBMClassifier(binary)を学習・比較し、
  真の実購入集合に対するMAP@12で評価。モデルをpickle保存し実験ログに記録
- `06_predict_submit.ipynb` — 学習済みモデルで全顧客分の予測を作り、提出csvを生成
  (直近56日非アクティブな顧客はトレンド人気でフォールバック)
- `07_optuna_tune.ipynb` — LGBMRankerのハイパーパラメータをoptunaで探索し、
  ベストパラメータで本番モデルを再学習

共有モジュール(`Z_notebook/` 直下、`sys.path.append(os.path.abspath("."))` でnotebookから読み込む想定):

- `common.py` — duckdb接続(`get_con`、threads=4 / memory_limit=12GB)、バックテスト分割(`backtest_split`,
  `label_window`, `active_customers`)、評価指標(`apk`/`mapk`/`recall_at_k`/`mean_recall_at_k`)、
  提出フォーマット(`format_prediction`)、実験ログ追記(`append_experiment_log`)、
  共通パス定数(`DATA_DIR`/`FEATURES_DIR`/`TRANS_LITE_PATH`/`SUBMISSION_DIR`/`MODELS_DIR`/`EXPERIMENT_LOG_PATH`)
- `candidates.py` — 6候補ソースの生成ロジック。全関数 `target_customers` 引数で対象顧客に絞り込み可能
  (絞り込まないと大規模データでOOMする、6節参照)
  - A `build_repeat_candidates` — 自己リピート(H1)。直近購入日 → 頻度の順
  - B `build_popularity_candidates` — トレンド人気(H10、顧客非依存の全体トップN、直近7日窓)
  - C `build_category_color_candidates` — 同product_code色違い(H1)+最頻section_name人気(H2c)で穴埋め
  - D `build_cf_candidates` — implicit ALS(factors=50, iterations=12, alpha=40, `filter_already_liked_items=False`)
  - E `build_cooccurrence_candidates` — 同一顧客・同一日(バスケット、H8)の同時購入からitem-to-item共起。
    Aの上位12件をシードにする(`build_item_cooccurrence_table` で `cooc_top` を作ってから集計)
  - F `build_content_similarity_candidates` — 商品テキストのTF-IDF類似度(`build_content_similarity.py` が
    事前計算した `content_similarity.parquet` を使用、カタログのみに依存しcutoff非依存)。Aの上位12件をシードにする
  - `build_candidate_pool` — **現在A〜Eの5ソースを統合**(Fは未統合、8節参照)
- `features.py` — `build_feature_table(con, cutoff, ...)`: 候補プール + user系/item系/user-item交差系/
  時間系特徴量 + ラベルを1cutoff分作る(04・06から共通利用)。対象顧客は
  「cutoff以前56日以内に購入あり」∪「ラベル期間に購入あり」
- `training_data.py` — 05/07で使う学習データロード。OOM対策込み(6節)。`load_train_val()` で
  (train_df, val_df, feature_cols)を返す。`group_sizes()` でLightGBMの `group` 引数も作る
- `build_one_cutoff.py` — `build_feature_table` を1cutoff分だけ実行しparquet保存するCLI。
  `python build_one_cutoff.py --name train1 --cutoff 2020-09-08 --out <path>`
- `build_content_similarity.py` — articles.csvの `prod_name`/`product_type_name`/`colour_group_name`/
  `detail_desc` からTF-IDF(max_features=20000)+コサイン類似度(sklearn NearestNeighbors)で
  `content_similarity.parquet` を作るCLI。カタログのみに依存するため一度作れば使い回せる

### 学習/検証の週構成(04)

| 用途 | cutoff | ラベル期間 |
|---|---|---|
| 検証(val) | 最終日-7日 | val cutoff の翌日 〜 最終日 |
| 学習(train1) | 最終日-14日 | train1 cutoff の翌日 〜 val cutoff |
| 学習(train2) | 最終日-21日 | train2 cutoff の翌日 〜 train1 cutoff |
| 学習(train3) | 最終日-28日 | train3 cutoff の翌日 〜 train2 cutoff |
| 学習(train4) | 最終日-35日 | train4 cutoff の翌日 〜 train3 cutoff |

valは05/07でモデル評価専用に使い学習には混ぜない。全cutoffが2020年8〜9月に収まっており、
COVID期(2020年3〜4月)のレジームシフトは含まれない。

### 特徴量(30列、`experiment_log.csv` の `features` 列と一致)

- 候補ソースフラグ: `from_repeat` / `from_trend` / `from_category_color` / `from_cf` / `from_cooccurrence` / `n_sources`
- カテゴリ(category dtype): `product_group_name` / `index_group_name` / `section_name`
- user系: `u_total_purchases`, `u_distinct_articles`, `u_distinct_product_codes`, `u_avg_price`, `u_price_cv`,
  `u_days_since_last_purchase`, `u_online_ratio`, `u_active_weeks`, `u_menswear_share`,
  `u_babychildren_share`, `u_ladieswear_share`(H2a/H2b/H2c, H5, H7)
- item系: `i_total_sales`, `i_avg_price`, `i_days_since_last_sale`, `i_sales_7d`, `i_sales_28d`(H10)
- user-item交差系: `ui_times_bought`, `ui_days_since_last_bought`, `ui_times_bought_product_code`,
  `ui_days_since_last_bought_product_code`, `ui_price_diff`(H1, H5/H7)
- 時間系: `target_week_start_dom`(給料日効果、H11)
- `product_code` は生成はされるが学習では未使用(読み込み時点で除外)

---

## 4. 候補ソース別Recall@12/MAP@12(直近7日ホールドアウト、03の最終結果)

| ソース | Recall@12 | MAP@12 |
|---|---|---|
| A: 自己リピート | 0.0405 | 0.0238 |
| B: トレンド人気 | 0.0255 | 0.0087 |
| C: 同カテゴリ/同色 | 0.0211 | 0.0082 |
| D: 協調フィルタリング(ALS) | 0.0168 | 0.0065 |
| E: アイテム共起 | 0.0172 | 0.0063 |
| F: テキストコンテンツ類似 | 0.0098 | 0.0033 |
| 参考: 02_baselineハイブリッド(A→B) | 0.0457 | 0.0248 |
| **6ソース単純結合プール(上限72件)** | **0.0933** | — |

現行モデル(候補5ソース、Fは未統合)のMAP@12は0.0336で、5ソース結合時点のRecall上限(0.088)にはまだ届いていない
(=候補側よりモデル側にまだ伸びしろがある可能性)。

---

## 5. 現在の最良モデル(2026-09-01時点)

- モデル: `02_output/models/model_20260901_0953_ranker.pkl`(LGBMRanker, デフォルトパラメータ、
  `{"model":..., "feature_cols":..., "cat_cols":...}` の辞書)
- 学習データ: `02_output/features/train_features_20260831_2226.parquet`
  (候補6ソース[from_content_similarity含む]×学習4週) / `val_features_20260831_2226.parquet`
- バックテストMAP@12: 0.0338 / Public LB: 0.02624 / Private LB: 0.02618
- 最新提出ファイル: `02_output/submission/submission_20260901_1019_cv0.0338.csv`
- 詳細は `experiment_log.csv` の `run_20260901_0953_ranker` 行

### optunaは現時点では不採用

`07_optuna_tune.ipynb`(20試行、単一val週での探索)で得たパラメータは
backtest MAP@12こそ0.0344に改善したが、実際のLBはPublic 0.02571 / Private 0.02611と悪化した
(`run_20260901_1232_ranker_optuna`)。単一val週(直近7日)へのoptuna過適合と見られる。
再挑戦する場合は複数week分のvalで評価するなど、単一週依存を減らす工夫を検討すること。

### 候補ソースG(from_price_band)も不採用

顧客の平均価格帯±標準偏差(H5)に収まる直近人気商品を候補にする`build_price_band_popularity_candidates`
(`candidates.py`)を追加し7ソース化した。単体Recall@12=0.0217/MAP@12=0.0076でD/E/Fより良く、
パイプライン全体でもbacktest MAP@12は0.0338→0.0343に改善(`run_20260901_1508_ranker`)。
しかし実際のLBはPublic 0.02624→0.02585, Private 0.02618→0.02608と悪化し、
optunaの時と同様に単一val週には合うが実際の翌週テストには合わない傾向が見られた。
関数自体はcandidates.py/features.py/training_data.pyに残しているが、
`build_candidate_pool`からは呼ばない(6ソースに戻す)。

### 06のモデル選択について(重要)

以前は `06_predict_submit.ipynb` の `ranker_path = sorted(glob.glob(...))[-1]` で
最新モデルファイルを自動選択していたが、上記のoptuna過適合の件で
「最新 = 最良」が成り立たないことが判明したため、**`ranker_path` は現在ベストモデルへの
固定パス(ハードコード)に変更済み**(`06_predict_submit.ipynb` 内にコメントで理由を明記)。
新しいrunが `experiment_log.csv` 上でこのベストを明確に上回ったことを確認してから、
手動で `ranker_path` と `BACKTEST_MAP12` を更新すること。

---

## 6. ハマったポイントと対策(重要、再度踏まないように)

このマシンは実メモリ16GBで、80M行規模のデータを扱うと素朴な実装ではほぼ確実にOOMする。
jetsamログ(`/Library/Logs/DiagnosticReports/JetsamEvent-*.ips`)で `Python` プロセスのkillを複数回確認した。

1. **train_features parquet(80M行)を `pd.read_parquet` で素朴に読むと即OOM**。
   文字列カテゴリ列がpandasの `object` dtype(行ごとにPythonの文字列オブジェクト)として展開され、
   読み込み時点だけでメモリを使い切る。
   → `pyarrow.parquet.read_table(..., columns=...).to_pandas(categories=[...])` で
   dictionary encodingのまま `category` dtypeにする。`training_data.py` に集約済み。
2. **train/valを両方読み込んでからダウンキャストすると、2つの巨大な生DataFrameが同時に存在する瞬間ができてOOM**
   (実測でメモリ使用量が約30GBに到達)。
   → trainを読み込み→ダウンキャスト→負例サンプリングまで完全に終わらせてから、valの生DataFrameを読み込む
   (`training_data.load_train_val` の実装順序そのもの)。
3. **candidates.pyのA/Cソースが元々target_customersで絞り込んでおらず、毎回全顧客(最大約137万人)分を
   計算していた**。これも大規模データでのOOMの一因(かつ単純に無駄な計算)。
   → `build_repeat_candidates`/`build_category_color_candidates` に `target_customers` 引数を追加し、
   `_register_target_customers` でduckdbに一時テーブル登録してWHERE句で絞り込む。
   `build_candidate_pool` はこれを渡すよう修正済み。**03のような「あえて全顧客で見たい」場面以外は
   必ずtarget_customersを渡すこと**(03でも `actual_map.keys()` を渡すよう修正済み)。
4. **04で5週分を1つのJupyterカーネルに溜め込むとOOM killされる**。
   → `build_one_cutoff.py` でcutoffごとに独立したOSプロセスとして実行し、
   各cutoff処理後にOSへメモリを完全に返却させる(04ノートブックは `subprocess.run` でこれを呼ぶ)。
5. **duckdbの `PRAGMA memory_limit` は低く設定しすぎると正常なクエリまで失敗する**。
   8GBに設定した状態で同カテゴリ/同色クエリが `OutOfMemoryException` になった実例あり。
   現在 `common.py` で12GBに設定している。サブプロセス(build_one_cutoff.py)は
   1プロセス=16GB丸ごと使えるので問題ないが、03/06のように単一プロセスで完結させるノートブックは
   この値の影響を受ける。
6. **陽性率が0.08%程度と極端に不均衡**なため、80M行のうち正解を1件も含まないグループ(顧客×cutoff)が
   96.8%を占める。これをグループ単位で15%だけ残す負例サンプリングでサイズを約14Mまで落としている
   (`training_data.py` の `neg_group_sample_frac` パラメータ)。**行単位で間引くとLightGBMの `group` 引数と
   対応が崩れるので、必ずグループ(顧客×cutoff)単位で残す/落とすを決めること**。
7. pyarrowで別々のparquetファイルを `categories=` 指定で読むと、ファイルごとに辞書順(category codes)が
   独立に決まる。06での予測時は学習時と同じcategoriesを使う必要があるため、
   train_features parquetから同じ手順でcategoriesを再現している(06のセル参照)。
8. `Z_notebook/.tmp/` にduckdbのディスクスピル用一時ファイルが溜まる(実行を重ねると数GB規模)。
   実行完了後に安全に削除できる(次回実行時に再作成される)。

---

## 7. コードとドキュメントの不整合(2026-08-14 のコード確認で検出、未修正)

いずれも結果には影響しないが、次に触るときに直しておくと迷いが減る。

1. `07_optuna_tune.ipynb` の実験ログ書き込みで `notes` が **「valは6候補ソース結合プール由来」** と
   ハードコードされているが、実際の `build_candidate_pool` は5ソース。
   `experiment_log.csv` の `run_20260814_0833_ranker_optuna` 行にもそのまま入っている(`features` 列の
   `from_*` フラグが5個であることが正)。
2. `05_train_lgbm.ipynb` は `training_data.py` を切り出す**前**に書かれたため、
   読み込み・ダウンキャスト・負例サンプリングのロジックをノートブック内に重複して持っている
   (07は `training_data.load_train_val` を使用)。05も差し替えれば単一実装になる。
3. `candidates.py` のモジュール冒頭docstringが
   「自己リピート/トレンド人気/同カテゴリ同色/協調フィルタリング」の4ソースのままで、E・Fが未記載。
4. `common.py` のdocstringに「候補生成・特徴量エンジニアリングそのもののロジックはここに置かず、
   各ノートブックに書く」とあるが、実際は `candidates.py` / `features.py` に切り出し済み。
5. `README.md` が 06/07 と `candidates.py`/`features.py`/`training_data.py` の追加を反映していない
   (`CLAUDE.md` 側は最新)。

---

## 8. 未着手・今後の選択肢

- **ソースF(テキストコンテンツ類似)は `build_candidate_pool` に未統合**。単体では全ソース中最弱だが、
  6ソース結合時のRecall上限は5ソース時点の0.088から0.093まで伸びた(+5.7%)。統合する場合は
  `candidates.py` の `build_candidate_pool` に `from_content_similarity` フラグを追加し、
  `features.py` のSELECT文(および `training_data.py` の `INT8_COLS`)にも追加した上で、
  04→05→06を通しで再実行する必要がある
  (04だけで候補5ソースの場合実測約45分、05の学習も含めると通しで1時間強)
- 画像embeddingは未着手。テキスト類似度(F)が最弱だったこと、105,100枚の高解像度画像への
  CNN推論はこのマシンだと相当重くなることから優先度を下げている
- 複数期間(季節性の異なる時期)でのバックテスト安定性は未検証。今回の学習/検証cutoffは
  全て2020年8月〜9月に集中しており、COVID期(2020年3〜4月)や他の季節は含まれていない
- CF(implicit ALS)は一貫して単体最弱グループ。`alpha`/`factors`/`iterations` のチューニングは未実施
- Classifierは負例サンプリング後のデータでMAP@12=0.0307までRankerに肉薄した(前回0.0203から大幅改善)。
  Rankerとのアンサンブルは試していない
- 05/07ともに **valでearly stoppingを掛けたうえで同じvalでMAP@12を測っている**ため、
  報告値は完全に独立した評価ではない(LBとの乖離幅を見るときはこの点に留意)

---

## 9. 環境・運用メモ

- Python: `/Users/woojoohyeon/Desktop/01_coding/.venv/bin/python3`(`.venv` は `01_coding` 直下、
  Kaggle全体で共有)
- このセッションで追加インストール: `implicit`(協調フィルタリング用)。`Kaggle/requirements.txt` に追記済み
- ノートブック実行は `jupyter nbconvert --to notebook --execute --inplace <name>.ipynb` で行っている
  (結果はセル出力にそのまま残る形。重い処理はバックグラウンド実行+完了待ちのパターンを多用した)
- Kaggle提出: `kaggle competitions submit -c h-and-m-personalized-fashion-recommendations -f <csv> -m "<message>"`、
  スコア確認: `kaggle competitions submissions -c h-and-m-personalized-fashion-recommendations`

## 10. Git状態(2026-08-14 時点)

直近のコミットは `762c2a8 Add hypothesis-driven EDA (H1-H11) and project CLAUDE.md` まで。
以下は**まだ未コミット**:

- 変更: `CLAUDE.md`, `README.md`, `experiment_log.csv`
- 新規: `00_md/pipeline_build_log.md`, `Z_notebook/02〜07_*.ipynb`,
  `Z_notebook/{common,candidates,features,training_data,build_one_cutoff,build_content_similarity}.py`,
  `Z_notebook/00_md/`

コミットするかはユーザーの指示を待つこと。コミット前は `CLAUDE.md` のルール通り
Notebookのセル出力をクリアすることを忘れずに。
