"""既存モデルを複数の未学習週で評価する軽量eval harness(再学習なし)。

単一val週(データ最終日の直近7日)だけを見る現状のbacktestは、optuna再チューニングや
候補ソースG(price_band)追加のように「backtestは改善するがLBは悪化する」というケースを
一度も検出できなかった(00_md/pipeline_build_log.md 5節)。これはval週1本のサンプルサイズの
小ささ(MAP@12は顧客ごとに正解1〜2件のみのためノイズが大きい)が主因とみられる。

このモジュールは既存の学習済みモデルpickleを、学習に使われていない複数の過去週に対して
スコアリングし、MAP@12の週ごとのばらつきを見る。改善がどの週でも一貫して見られるなら
本物の改善、週によってバラつく/1週だけ良いなら過適合を疑う、という判断に使う。
"""

import os
import pickle

import pandas as pd

from common import label_window
from features import build_feature_table

NULLABLE_INT_COLS = [
    "u_days_since_last_purchase",
    "ui_days_since_last_bought",
    "ui_days_since_last_bought_product_code",
]


def holdout_cutoffs(con, n_weeks: int = 4, min_gap_days: int = 42, step_days: int = 7) -> list:
    """データ最終日から min_gap_days 日以上遡った週次cutoffを n_weeks 個返す。

    既存モデルは全て「データ最終日 - 35日」(train4のcutoff)以降をラベル期間として学習に
    使っているため、min_gap_days=42以上離せばどの既存モデルの学習にも含まれない
    (cutoff+7日のラベル期間末がdata最終日-35日以前に収まる)。
    """
    max_dat = con.execute("SELECT MAX(t_dat) FROM trans").fetchone()[0]
    return [
        (max_dat - pd.Timedelta(days=min_gap_days + step_days * i)).strftime("%Y-%m-%d")
        for i in range(n_weeks)
    ]


def load_model(model_path: str) -> dict:
    with open(model_path, "rb") as f:
        return pickle.load(f)


def build_pred_map_from_scores(df: pd.DataFrame, scores, k: int = 12) -> dict:
    tmp = df[["customer_idx", "article_id"]].copy()
    tmp["score"] = scores
    top = tmp.sort_values("score", ascending=False).groupby("customer_idx").head(k)
    return top.groupby("customer_idx")["article_id"].apply(list).to_dict()


def score_model_at_cutoff(
    con,
    payload: dict,
    cutoff: str,
    train_categories: dict,
    k_per_source=None,
    label_days: int = 7,
    trailing_days: int = 56,
):
    """1つのcutoffで候補+特徴量テーブルを作り、モデルでMAP@12を測る。

    train_categories: {cat_col: pandas Index/CategoricalDtype.categories} で、そのモデルの
    学習時と同じcategory符号化(コード対応)を再現するために使う(06_predict_submit.ipynbと同じ手順、
    00_md/pipeline_build_log.md 6節9項参照)。
    """
    feature_cols = payload["feature_cols"]
    cat_cols = payload["cat_cols"]

    df = build_feature_table(
        con, cutoff, label_days=label_days, trailing_days=trailing_days,
        k_per_source=k_per_source, with_label=True,
    )
    actual_map = label_window(con, cutoff, label_days)

    for c in NULLABLE_INT_COLS:
        if c in df.columns:
            df[c] = df[c].astype("float32")
    for c in cat_cols:
        df[c] = pd.Categorical(df[c], categories=train_categories[c])

    X = df[feature_cols]
    scores = payload["model"].predict(X)
    pred_map = build_pred_map_from_scores(df, scores, k=12)

    from common import mapk
    return mapk(actual_map, pred_map, k=12), len(actual_map)


def load_train_categories(train_features_path: str, cat_cols: list) -> dict:
    import pyarrow.parquet as pq

    t = pq.read_table(train_features_path, columns=cat_cols).to_pandas(categories=cat_cols)
    return {c: t[c].cat.categories for c in cat_cols}


def score_df_with_model(df: pd.DataFrame, actual_map: dict, payload: dict, train_categories: dict):
    """既に構築済みの候補×特徴量テーブルdfに対して、1モデル分のMAP@12を測る。

    同じcutoff・同じ候補ソース構成(k_per_source)を共有するモデル同士(例: デフォルトパラメータ vs
    optunaパラメータ。候補プールは同一で学習ハイパラだけが違う)は、候補生成・特徴量計算という
    重い処理を1回で済ませてスコアリングだけ複数回行える(group_score_at_cutoff参照)。
    dfそのものは書き換えない(呼び出し元で複数モデルに使い回すため)。
    """
    feature_cols = payload["feature_cols"]
    cat_cols = payload["cat_cols"]

    X = pd.DataFrame({c: df[c] for c in feature_cols})
    for c in NULLABLE_INT_COLS:
        if c in X.columns:
            X[c] = X[c].astype("float32")
    for c in cat_cols:
        X[c] = pd.Categorical(df[c], categories=train_categories[c])

    scores = payload["model"].predict(X)
    pred_map = build_pred_map_from_scores(df, scores, k=12)

    from common import mapk
    return mapk(actual_map, pred_map, k=12)


def group_score_at_cutoff(
    con,
    cutoff: str,
    models: list,
    k_per_source=None,
    label_days: int = 7,
    trailing_days: int = 56,
) -> dict:
    """同じ候補ソース構成(k_per_source)を共有するモデル群を、1回の候補+特徴量構築で評価する。

    models: [(label, payload, train_categories), ...] のリスト。
    各モデルのfeature_colsがdfに存在する列のみを参照する前提(候補ソース構成が一致していること)。
    戻り値: {label: map12}
    """
    df = build_feature_table(
        con, cutoff, label_days=label_days, trailing_days=trailing_days,
        k_per_source=k_per_source, with_label=True,
    )
    actual_map = label_window(con, cutoff, label_days)

    results = {}
    for label, payload, train_categories in models:
        results[label] = score_df_with_model(df, actual_map, payload, train_categories)
        print(f"  [{label}] cutoff={cutoff} MAP@12={results[label]:.4f} (n_customers={len(actual_map):,})")
    return results


def multi_week_eval(
    con,
    model_path: str,
    train_features_path: str,
    cutoffs: list,
    k_per_source=None,
    label_days: int = 7,
    trailing_days: int = 56,
) -> pd.DataFrame:
    """複数cutoffでのMAP@12を1モデル分まとめて計測する。"""
    payload = load_model(model_path)
    train_categories = load_train_categories(train_features_path, payload["cat_cols"])

    rows = []
    for cutoff in cutoffs:
        score, n_customers = score_model_at_cutoff(
            con, payload, cutoff, train_categories, k_per_source, label_days, trailing_days,
        )
        print(f"  [{os.path.basename(model_path)}] cutoff={cutoff} MAP@12={score:.4f} (n_customers={n_customers:,})")
        rows.append({"cutoff": cutoff, "map12": score, "n_customers": n_customers})

    result = pd.DataFrame(rows)
    result["model"] = os.path.basename(model_path)
    return result
