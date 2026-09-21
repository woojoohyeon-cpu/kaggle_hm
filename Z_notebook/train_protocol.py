"""短期トレンド仮説の検証用: 独立評価を守った学習・評価プロトコル(run_trend_experiment.pyから使う)。

## 05_train_lgbm.ipynb からの変更点(評価の独立性)

05は「valでearly stoppingを掛け、同じvalでMAP@12を報告」していたため、報告値に楽観バイアスがあった
(00_md/pipeline_build_log.md 8節の注記)。ここでは役割を完全に分ける:

| 週 | 役割 |
|---|---|
| train2〜train4 | 学習 |
| train1 | early stoppingの基準のみ(学習にもMAP報告にも使わない) |
| val | MAP@12の報告のみ(学習にもearly stoppingにも一切使わない) |
| holdout1〜3 | 追加の報告週(学習期間より前の週。週ごとのばらつきを見る) |

## 比較の設計(1回の特徴量構築で段階的なablationを取る)

特徴量テーブルは「全列+追加候補ソース込み」で1回だけ作り、読み込み時に列と行を絞って3条件を作る:

- base: 0906版と同じ35列・6ソースのプール(n_sources > 0 の行)。同じプロトコルで学習し直した対照群
- trend_feat: base + 短期トレンド特徴量(プールは同じ) → 特徴量だけの効果
- trend_full: trend_feat + 年齢層別トレンド候補(from_age_trend) → 候補追加の効果

base/trend_featは学習行・評価行が完全に同一なので、差はそのまま「特徴量の効果」になる。
さらに顧客単位のAPの差をブートストラップして、差がノイズの範囲かどうかを信頼区間で判断する
(これまでのLB上の差は±0.0002程度で、ノイズと区別がつかないことが多かったため)。
"""

import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from common import DATA_DIR, apk
from features import BASE_0906_FEATURES, CAT_COLS, TREND_FEATURES

VARIANTS = {
    "base": {"pool": "base", "features": list(BASE_0906_FEATURES)},
    "trend_feat": {"pool": "base", "features": list(BASE_0906_FEATURES) + list(TREND_FEATURES)},
    "trend_full": {"pool": "full", "features": list(BASE_0906_FEATURES) + list(TREND_FEATURES) + ["from_age_trend"]},
}

# 05_train_lgbm.ipynb と同じハイパーパラメータ(比較条件を揃えるため変えない)
LGBM_PARAMS = dict(
    objective="lambdarank",
    n_estimators=300,
    num_leaves=63,
    learning_rate=0.05,
    random_state=42,
    verbosity=-1,
)
EARLY_STOPPING_ROUNDS = 30
KEY_COLS = ["customer_idx", "article_id"]


def category_vocab() -> dict:
    """カテゴリ列の語彙をarticles.csvから固定する(ファイルごとに辞書順が変わる問題を根本的に避ける)。"""
    articles = pd.read_csv(os.path.join(DATA_DIR, "articles.csv"), usecols=CAT_COLS, dtype=str)
    return {c: sorted(articles[c].dropna().unique().tolist()) for c in CAT_COLS}


def _pool_filter(pool: str):
    if pool == "base":
        return ds.field("n_sources") > 0
    if pool == "full":
        return None
    raise ValueError(pool)


def _and(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return a & b


def _to_pandas(table: pa.Table, vocab: dict) -> pd.DataFrame:
    df = table.to_pandas()
    for c in CAT_COLS:
        if c in df.columns:
            df[c] = pd.Categorical(df[c].astype(object), categories=vocab[c])
    return df


def read_week(path: str, columns: list, pool: str, vocab: dict, customers=None) -> pd.DataFrame:
    """1週分の特徴量parquetから、必要な列・プール・顧客だけを読み込む(全列の生読み込みをしない)。"""
    dataset = ds.dataset(path, format="parquet")
    cols = list(dict.fromkeys(KEY_COLS + list(columns)))
    filt = _pool_filter(pool)
    if customers is not None:
        filt = _and(filt, ds.field("customer_idx").isin(pa.array(np.asarray(customers, dtype=np.int32))))
    return _to_pandas(dataset.to_table(columns=cols, filter=filt), vocab)


def sample_groups(path: str, pool: str, neg_frac: float, seed: int, positive_only: bool = False) -> np.ndarray:
    """正解を含む顧客は全員、正解を含まない顧客はneg_fracの割合だけ残す(グループ単位の負例サンプリング)。"""
    dataset = ds.dataset(path, format="parquet")
    t = dataset.to_table(columns=["customer_idx", "label"], filter=_pool_filter(pool))
    lab = pd.DataFrame({"customer_idx": t["customer_idx"].to_numpy(), "label": t["label"].to_numpy()})
    pos = np.unique(lab.loc[lab["label"] == 1, "customer_idx"].to_numpy())
    if positive_only:
        return pos
    neg = np.setdiff1d(np.unique(lab["customer_idx"].to_numpy()), pos)
    rng = np.random.default_rng(seed)
    keep_neg = rng.choice(neg, size=int(len(neg) * neg_frac), replace=False) if len(neg) else neg
    return np.concatenate([pos, keep_neg])


def load_training_frame(paths: list, feature_cols: list, pool: str, vocab: dict, neg_frac: float,
                        seed: int = 42, positive_only: bool = False) -> pd.DataFrame:
    frames = []
    for i, p in enumerate(paths):
        keep = sample_groups(p, pool, neg_frac, seed + i, positive_only=positive_only)
        frames.append(read_week(p, feature_cols + ["label", "cutoff"], pool, vocab, customers=keep))
    df = pd.concat(frames, ignore_index=True)
    del frames
    for c in CAT_COLS:
        if c in df.columns:
            df[c] = pd.Categorical(df[c].astype(object), categories=vocab[c])
    # parquetの行順は保証されない(write_feature_tableは並べ替えずに書き出す)ので、学習結果が再現するよう完全に並べる
    df = df.sort_values(["cutoff", "customer_idx", "article_id"], kind="stable").reset_index(drop=True)
    return df


def group_sizes(df: pd.DataFrame) -> np.ndarray:
    return df.groupby(["cutoff", "customer_idx"], sort=False, observed=True).size().to_numpy()


def fit_ranker(train_df, feature_cols, eval_df=None, n_estimators=None, log_every=50):
    params = dict(LGBM_PARAMS)
    if n_estimators is not None:
        params["n_estimators"] = int(n_estimators)
    cat_cols = [c for c in CAT_COLS if c in feature_cols]
    model = lgb.LGBMRanker(**params)
    fit_kwargs = dict(group=group_sizes(train_df), categorical_feature=cat_cols)
    callbacks = [lgb.log_evaluation(log_every)]
    if eval_df is not None:
        fit_kwargs.update(eval_set=[(eval_df[feature_cols], eval_df["label"])],
                          eval_group=[group_sizes(eval_df)], eval_at=[12])
        callbacks.append(lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True))
    model.fit(train_df[feature_cols], train_df["label"], callbacks=callbacks, **fit_kwargs)
    best_iter = model.best_iteration_ if eval_df is not None and model.best_iteration_ else params["n_estimators"]
    return model, int(best_iter)


def score_week(path: str, payload: dict, vocab: dict, batch_rows: int = 2_000_000) -> pd.DataFrame:
    """1週分のテーブルを分割して読みながらスコアリングし、(customer_idx, article_id, score)を返す。"""
    feature_cols = payload["feature_cols"]
    dataset = ds.dataset(path, format="parquet")
    cols = list(dict.fromkeys(KEY_COLS + feature_cols))
    scanner = dataset.scanner(columns=cols, filter=_pool_filter(payload["pool"]), batch_size=batch_rows)
    parts = []
    model = payload["model"]
    predict_kwargs = {"num_iteration": payload.get("best_iter")} if payload.get("best_iter") else {}
    for batch in scanner.to_batches():
        if batch.num_rows == 0:
            continue
        df = _to_pandas(pa.Table.from_batches([batch]), vocab)
        parts.append(pd.DataFrame({
            "customer_idx": df["customer_idx"].to_numpy(),
            "article_id": df["article_id"].to_numpy(),
            "score": model.predict(df[feature_cols], **predict_kwargs).astype(np.float32),
        }))
    return pd.concat(parts, ignore_index=True)


def top_k_map(scored: pd.DataFrame, k: int = 12) -> dict:
    # 同点スコアの並びを固定する(article_id昇順)。これが無いと行の読み込み順で結果が微妙に揺れる
    top = scored.sort_values(["customer_idx", "score", "article_id"], ascending=[True, False, True], kind="stable")
    top = top.groupby("customer_idx", sort=False).head(k)
    return top.groupby("customer_idx", sort=False)["article_id"].apply(list).to_dict()


def per_customer_ap(actual_map: dict, pred_map: dict, k: int = 12):
    """actual_mapの全顧客(=その週に購入した全員。候補プール外の顧客は0点)について顧客別AP@kを返す。"""
    customers = np.array(sorted(actual_map.keys()), dtype=np.int64)
    ap = np.array([apk(actual_map[c], pred_map.get(c, []), k) for c in customers], dtype=np.float64)
    return customers, ap


def fallback_lists(con, cutoff: str):
    """提出時と同じフォールバック用リスト: ({年齢層: 直近7日の同年齢層人気12件}, 直近7日の全体人気12件)。"""
    from candidates import age_trend_lists, build_popularity_candidates, ensure_cust_age
    ensure_cust_age(con)
    return age_trend_lists(con, cutoff, k=12), build_popularity_candidates(con, cutoff, top_n=12)


def fill_to_k(cands, age_list, global_top, k: int = 12) -> list:
    """モデルの候補(最大k件)を、年齢層人気→全体人気の順で重複なくk件まで埋める。"""
    out = list(cands[:k])
    seen = set(out)
    for a in list(age_list) + list(global_top):
        if len(out) >= k:
            break
        if a not in seen:
            out.append(a)
            seen.add(a)
    return out


def submission_like_ap(actual_map: dict, model_pred_map: dict, active: set, buckets: dict,
                       age_top: dict, global_top: list, cold_start: str, k: int = 12):
    """提出ファイルと同じ作り方で予測を作ってから顧客別AP@kを出す。

    - 直近56日にアクティブな顧客: モデルの上位k件(足りなければフォールバックで補充)
    - それ以外(評価週に初めて/久しぶりに買った顧客): フォールバックのみ
      (特徴量テーブルは学習用にこの顧客たちも含めて作っているが、提出時はモデルの対象外なので評価でも使わない)
    cold_start: "global"(従来の06と同じ全体人気)/ "age_trend"(同年齢層の直近人気→全体人気)
    """
    customers = np.array(sorted(actual_map.keys()), dtype=np.int64)
    ap = np.empty(len(customers), dtype=np.float64)
    for i, c in enumerate(customers):
        c = int(c)
        age_list = age_top.get(buckets.get(c, -1), []) if cold_start == "age_trend" else []
        model_list = model_pred_map.get(c, []) if c in active else []
        ap[i] = apk(actual_map[c], fill_to_k(model_list, age_list, global_top, k), k)
    return customers, ap


def paired_bootstrap(diff: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """顧客単位の差(新-旧)の平均について、顧客のリサンプリングで95%信頼区間を出す。"""
    rng = np.random.default_rng(seed)
    n = len(diff)
    chunk = max(1, int(2e7 // max(n, 1)))  # 一度に作る添字配列を約2,000万要素(160MB)に抑える
    means = []
    for start in range(0, n_boot, chunk):
        m = min(chunk, n_boot - start)
        idx = rng.integers(0, n, size=(m, n))
        means.append(diff[idx].mean(axis=1))
    means = np.concatenate(means)
    return {
        "mean_diff": float(diff.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "p_improve": float((means > 0).mean()),
    }


def save_payload(path: str, **payload) -> None:
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_payload(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def feature_importance(model, feature_cols) -> pd.DataFrame:
    booster = model.booster_
    return pd.DataFrame({
        "feature": feature_cols,
        "gain": booster.feature_importance(importance_type="gain"),
        "split": booster.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False).reset_index(drop=True)
