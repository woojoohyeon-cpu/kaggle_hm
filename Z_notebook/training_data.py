"""05_train_lgbm.ipynb で確立したOOM対策込みの学習データ読み込みロジックを共通化したモジュール。

80M行規模のtrain_features parquetを素朴にpandasで読むとメモリを使い切ってカーネルがOOM killされることを
確認した(jetsamログでPythonプロセスのkillを確認)。以下の対策をまとめて行う。

1. pyarrowの`categories`引数でカテゴリ列を直接`category`dtypeにする(pandasの素朴な読み込みだと
   行ごとにPython文字列オブジェクトが展開されて非常に重い)
2. train→val の順で完全に処理してから次に進む(2つの巨大な生DataFrameを同時にメモリへ載せない)
3. 0/1のソースフラグ・label・n_sourcesはint8、customer_idx/article_id/カウント系はint32、
   価格・比率系はfloat32、`cutoff`もcategoryへダウンキャストする
4. 陽性率が0.08%程度と極端に不均衡なため、正解を1件も含まないグループ(顧客×cutoff)を間引く
   (グループ単位で残す/落とすを決める。行単位で間引くとLightGBMのgroup引数と対応が崩れる)
"""

import glob
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from common import FEATURES_DIR

CAT_COLS = ["product_group_name", "index_group_name", "section_name"]
INT8_COLS = ["from_repeat", "from_trend", "from_category_color", "from_cf", "from_cooccurrence",
             "from_content_similarity", "n_sources", "label"]
INT32_COLS = ["customer_idx", "article_id", "u_total_purchases", "u_distinct_articles",
              "u_distinct_product_codes", "u_active_weeks", "u_purchases_7d", "u_purchases_28d",
              "u_purchases_91d", "i_total_sales",
              "i_days_since_last_sale", "i_sales_7d", "i_sales_28d", "i_sales_yoy_7d",
              "cat_sales_yoy_7d", "cat_sales_recent_28d",
              "ui_times_bought", "ui_times_bought_product_code", "target_week_start_dom"]
FLOAT32_COLS = ["u_avg_price", "u_price_cv", "u_online_ratio", "u_menswear_share",
                 "u_babychildren_share", "u_ladieswear_share", "i_avg_price", "ui_price_diff",
                 "u_days_since_last_purchase", "ui_days_since_last_bought",
                 "ui_days_since_last_bought_product_code", "cat_seasonal_lift"]
EXCLUDE_COLS = ["customer_idx", "article_id", "label", "cutoff"]


def latest_feature_paths():
    train_path = sorted(glob.glob(os.path.join(FEATURES_DIR, "train_features_*.parquet")))[-1]
    val_path = sorted(glob.glob(os.path.join(FEATURES_DIR, "val_features_*.parquet")))[-1]
    return train_path, val_path


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    df["cutoff"] = df["cutoff"].astype("category")
    return df


_ARROW_DOWNCAST = {}
for _c in INT8_COLS:
    _ARROW_DOWNCAST[_c] = pa.int8()
for _c in INT32_COLS:
    _ARROW_DOWNCAST[_c] = pa.int32()
for _c in FLOAT32_COLS:
    _ARROW_DOWNCAST[_c] = pa.float32()


def _load_one(path: str) -> pd.DataFrame:
    """pandasに変換する前にArrow側で数値列を狭い型にcastしてから読み込む。

    pyarrowのtable→pandas変換は、変換前のArrow型(parquetに書き出された生のint64/float64)の
    ままメモリに展開されてから初めてpandas側でdowncastできる。列数が増えるほどこの変換直前の
    ピークメモリが線形に増え、80M行超のtrain_featuresでOOM killされることを確認済み
    (00_md/pipeline_build_log.md 6節)。Arrow Tableの段階でint8/int32/float32にcastしてから
    to_pandas()する(=変換前から既に狭い型)ことで、このピークをpandas側dowcastより大きく削減する。
    """
    load_columns = [c for c in pq.ParquetFile(path).schema.names if c != "product_code"]
    table = pq.read_table(path, columns=load_columns)
    for name, arrow_type in _ARROW_DOWNCAST.items():
        if name in table.column_names:
            idx = table.column_names.index(name)
            table = table.set_column(idx, name, table.column(name).cast(arrow_type))
    df = table.to_pandas(categories=CAT_COLS)
    del table
    return _downcast(df)


def load_train_val(neg_group_sample_frac: float = 0.15, random_state: int = 42, verbose: bool = True):
    """(train_df, val_df, feature_cols) を返す。train_dfは負例サンプリング・カテゴリ整合まで完了済み。"""
    import gc

    train_path, val_path = latest_feature_paths()
    if verbose:
        print(f"train: {train_path}")
        print(f"val:   {val_path}")

    # --- train: 読み込み→ダウンキャスト→負例サンプリングまで完結させる ---
    train_df = _load_one(train_path)
    if verbose:
        print(f"train ダウンキャスト後: {train_df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

    group_has_positive = train_df.groupby(["cutoff", "customer_idx"], observed=True)["label"].transform("max")
    positive_rows = train_df[group_has_positive == 1]
    negative_only_rows = train_df[group_has_positive == 0]

    neg_group_keys = negative_only_rows[["cutoff", "customer_idx"]].drop_duplicates()
    sampled_neg_keys = neg_group_keys.sample(frac=neg_group_sample_frac, random_state=random_state)
    sampled_negative_rows = negative_only_rows.merge(sampled_neg_keys, on=["cutoff", "customer_idx"])

    train_df = pd.concat([positive_rows, sampled_negative_rows], ignore_index=True)
    del positive_rows, negative_only_rows, sampled_negative_rows, group_has_positive
    gc.collect()
    if verbose:
        print(f"サンプリング後のtrain_df: {train_df.shape}, memory: {train_df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

    # --- val: trainの生データが片付いた後に読み込む ---
    val_df = _load_one(val_path)
    if verbose:
        print(f"val: {val_df.shape}, memory: {val_df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

    feature_cols = [c for c in train_df.columns if c not in EXCLUDE_COLS]

    train_df = train_df.sort_values(["cutoff", "customer_idx"]).reset_index(drop=True)
    val_df = val_df.sort_values(["customer_idx"]).reset_index(drop=True)

    for c in CAT_COLS:
        train_df[c] = train_df[c].astype("category")
        val_df[c] = pd.Categorical(val_df[c], categories=train_df[c].cat.categories)

    return train_df, val_df, feature_cols


def group_sizes(df: pd.DataFrame, group_cols) -> "pd.Series":
    return df.groupby(group_cols, sort=False, observed=True).size().to_numpy()
