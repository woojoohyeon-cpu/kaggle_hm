"""候補生成ロジック(自己リピート/トレンド人気/同カテゴリ同色/協調フィルタリング/アイテム共起/テキストコンテンツ類似)。

03_candidate_generation.ipynb でソースごとのRecall@12/MAP@12を検証済み(参照: 00_md/hypothesis_driven_design.md)。
04以降のノートブックはこのモジュールを共有して使う。
"""

import os

import numpy as np
import pandas as pd
import implicit
from scipy.sparse import coo_matrix

_CONTENT_SIM_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "02_output", "features", "content_similarity.parquet"
)
_content_sim_cache = None


def _load_content_similarity() -> dict:
    """build_content_similarity.py が事前計算したtext類似度テーブルを読み込み、
    article_id -> [(neighbor_article_id, similarity), ...] の辞書にキャッシュする。"""
    global _content_sim_cache
    if _content_sim_cache is None:
        df = pd.read_parquet(_CONTENT_SIM_PATH)
        _content_sim_cache = {
            article_id: list(zip(g["neighbor_article_id"], g["similarity"]))
            for article_id, g in df.groupby("article_id")
        }
    return _content_sim_cache


def build_content_similarity_candidates(con, cutoff: str, target_customers, k: int = 12, seed_map=None) -> dict:
    """テキストコンテンツ類似候補: 顧客の自己リピート上位12件をシードにして、
    内容(prod_name/product_type_name/colour_group_name/detail_desc)が類似した商品を集計する。

    類似度テーブル自体はカタログのみに依存しcutoffと無関係なため事前計算済みのものを使い回す
    (build_content_similarity.py, 02_output/features/content_similarity.parquet)。
    seed_mapを渡さない場合はbuild_repeat_candidatesで内部計算する(単体ソースとしての検証用)。
    """
    content_sim = _load_content_similarity()
    if seed_map is None:
        seed_map = build_repeat_candidates(con, cutoff, target_customers=target_customers, k=12)

    preds = {}
    for cid in {int(c) for c in target_customers}:
        scores = {}
        for seed in seed_map.get(cid, []):
            for neighbor, sim in content_sim.get(int(seed), []):
                scores[neighbor] = scores.get(neighbor, 0.0) + sim
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        preds[cid] = [a for a, _ in top]
    return preds


def _register_target_customers(con, target_customers) -> str:
    """target_customersをduckdbに一時登録し、フィルタ用のWHERE句断片を返す(Noneなら絞り込みなし)。"""
    if target_customers is None:
        return ""
    target_df = pd.DataFrame({"customer_idx": list({int(c) for c in target_customers})})
    con.register("_target_customers_tbl", target_df)
    return "AND customer_idx IN (SELECT customer_idx FROM _target_customers_tbl)"


def build_repeat_candidates(con, cutoff: str, target_customers=None, k: int = 12) -> dict:
    """自己リピート候補(H1): 直近購入日→頻度の順。

    target_customersを渡すと、その顧客集合だけに絞って計算する(無駄な全顧客計算を避ける)。
    """
    filter_clause = _register_target_customers(con, target_customers)
    df = con.execute(f"""
        WITH agg AS (
            SELECT customer_idx, article_id, COUNT(*) AS freq, MAX(t_dat) AS last_dat
            FROM trans
            WHERE t_dat <= DATE '{cutoff}' {filter_clause}
            GROUP BY customer_idx, article_id
        ),
        ranked AS (
            SELECT customer_idx, article_id,
                   ROW_NUMBER() OVER (PARTITION BY customer_idx ORDER BY last_dat DESC, freq DESC) AS rn
            FROM agg
        )
        SELECT customer_idx, LIST(article_id ORDER BY rn) AS candidates
        FROM ranked
        WHERE rn <= {k}
        GROUP BY customer_idx
    """).df()
    if target_customers is not None:
        con.unregister("_target_customers_tbl")
    return dict(zip(df.customer_idx, df.candidates))


def build_popularity_candidates(con, cutoff: str, window_days: int = 7, top_n: int = 12) -> list:
    """トレンド人気候補(H10): cutoff直前window_days日間の全体人気トップN。"""
    df = con.execute(f"""
        SELECT article_id, COUNT(*) AS cnt
        FROM trans
        WHERE t_dat <= DATE '{cutoff}' AND t_dat > DATE '{cutoff}' - INTERVAL {window_days} DAY
        GROUP BY article_id
        ORDER BY cnt DESC
        LIMIT {top_n}
    """).df()
    return df.article_id.tolist()


def build_price_band_popularity_candidates(
    con,
    cutoff: str,
    target_customers=None,
    k: int = 12,
    window_days: int = 7,
    pool_size: int = 300,
    band_width: float = 0.3,
) -> dict:
    """価格帯フィルタ人気候補(H5): 顧客ごとの平均購入価格帯に収まる直近人気商品を優先する。

    顧客の価格CVが母集団より小さい(H5)前提のもと、直近window_days日間の人気上位pool_size件を
    候補プールとして、顧客の平均価格±max(価格の標準偏差, 平均価格*band_width)の範囲に入るものだけを
    人気順に残す(購入1回のみでstd_priceがNULLの顧客はband_width側のフォールバック幅を使う)。
    """
    filter_clause = _register_target_customers(con, target_customers)
    df = con.execute(f"""
        WITH cust_price AS (
            SELECT customer_idx, AVG(price) AS avg_price, STDDEV(price) AS std_price
            FROM trans
            WHERE t_dat <= DATE '{cutoff}' {filter_clause}
            GROUP BY customer_idx
        ),
        recent_pop AS (
            SELECT article_id, COUNT(*) AS cnt, AVG(price) AS item_avg_price
            FROM trans
            WHERE t_dat <= DATE '{cutoff}' AND t_dat > DATE '{cutoff}' - INTERVAL {window_days} DAY
            GROUP BY article_id
            ORDER BY cnt DESC
            LIMIT {pool_size}
        ),
        banded AS (
            SELECT c.customer_idx, r.article_id, r.cnt,
                   ROW_NUMBER() OVER (PARTITION BY c.customer_idx ORDER BY r.cnt DESC) AS rn
            FROM cust_price c
            JOIN recent_pop r
                ON r.item_avg_price BETWEEN
                   c.avg_price - GREATEST(COALESCE(c.std_price, 0), c.avg_price * {band_width})
                   AND c.avg_price + GREATEST(COALESCE(c.std_price, 0), c.avg_price * {band_width})
        )
        SELECT customer_idx, LIST(article_id ORDER BY rn) AS candidates
        FROM banded
        WHERE rn <= {k}
        GROUP BY customer_idx
    """).df()
    if target_customers is not None:
        con.unregister("_target_customers_tbl")
    return dict(zip(df.customer_idx, df.candidates))


def build_category_color_candidates(con, cutoff: str, target_customers=None, k: int = 12) -> dict:
    """同カテゴリ/同色候補: product_code色違い(H1)優先、不足分は最頻section_nameの人気商品(H2c)で補完。

    target_customersを渡すと、その顧客集合だけに絞って計算する(無駄な全顧客計算を避ける)。
    """
    filter_clause = _register_target_customers(con, target_customers)
    variant_df = con.execute(f"""
        WITH cust_products AS (
            SELECT DISTINCT customer_idx, product_code
            FROM trans WHERE t_dat <= DATE '{cutoff}' {filter_clause}
        ),
        variant_pop AS (
            SELECT product_code, article_id, COUNT(*) AS cnt
            FROM trans WHERE t_dat <= DATE '{cutoff}'
            GROUP BY product_code, article_id
        ),
        ranked AS (
            SELECT cp.customer_idx, vp.article_id,
                   ROW_NUMBER() OVER (PARTITION BY cp.customer_idx ORDER BY vp.cnt DESC) AS rn
            FROM cust_products cp
            JOIN variant_pop vp USING (product_code)
        )
        SELECT customer_idx, LIST(article_id ORDER BY rn) AS candidates
        FROM ranked
        WHERE rn <= {k}
        GROUP BY customer_idx
    """).df()
    variant_map = dict(zip(variant_df.customer_idx, variant_df.candidates))

    cust_top_section_df = con.execute(f"""
        WITH sec_cnt AS (
            SELECT t.customer_idx, a.section_name, COUNT(*) AS cnt
            FROM trans t
            JOIN articles a ON a.article_id = t.article_id
            WHERE t.t_dat <= DATE '{cutoff}' {filter_clause}
            GROUP BY t.customer_idx, a.section_name
        )
        SELECT customer_idx, section_name
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY customer_idx ORDER BY cnt DESC) AS rn
            FROM sec_cnt
        )
        WHERE rn = 1
    """).df()
    cust_top_section = dict(zip(cust_top_section_df.customer_idx, cust_top_section_df.section_name))
    if target_customers is not None:
        con.unregister("_target_customers_tbl")

    section_pop_df = con.execute(f"""
        WITH sec_item_cnt AS (
            SELECT a.section_name, t.article_id, COUNT(*) AS cnt
            FROM trans t
            JOIN articles a ON a.article_id = t.article_id
            WHERE t.t_dat <= DATE '{cutoff}'
            GROUP BY a.section_name, t.article_id
        ),
        ranked AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY section_name ORDER BY cnt DESC) AS rn
            FROM sec_item_cnt
        )
        SELECT section_name, LIST(article_id ORDER BY rn) AS item_list
        FROM ranked
        WHERE rn <= 30
        GROUP BY section_name
    """).df()
    section_pop = dict(zip(section_pop_df.section_name, section_pop_df.item_list))

    preds = {}
    for cid, candidates in variant_map.items():
        candidates = list(candidates)
        if len(candidates) < k:
            seen = set(candidates)
            section = cust_top_section.get(cid)
            for a in section_pop.get(section, []):
                if a not in seen:
                    candidates.append(a)
                    seen.add(a)
                if len(candidates) >= k:
                    break
        preds[cid] = candidates[:k]
    for cid, section in cust_top_section.items():
        if cid not in preds:
            preds[cid] = list(section_pop.get(section, []))[:k]
    return preds


def build_item_cooccurrence_table(con, cutoff: str, top_partners_per_item: int = 20) -> None:
    """同一顧客・同一日(バスケット、H8)の同時購入からitem-to-item共起テーブルを作る。

    article_aごとに共起回数上位top_partners_per_item件だけ残した`cooc_top`をduckdbに作成する。
    """
    con.execute(f"""
        CREATE OR REPLACE TABLE cooc_raw AS
        WITH basket_items AS (
            SELECT DISTINCT customer_idx, t_dat, article_id
            FROM trans
            WHERE t_dat <= DATE '{cutoff}'
        )
        SELECT a.article_id AS article_a, b.article_id AS article_b, COUNT(*) AS cnt
        FROM basket_items a
        JOIN basket_items b
            ON a.customer_idx = b.customer_idx
           AND a.t_dat = b.t_dat
           AND a.article_id != b.article_id
        GROUP BY a.article_id, b.article_id
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE cooc_top AS
        SELECT article_a, article_b, cnt
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY article_a ORDER BY cnt DESC) AS rn
            FROM cooc_raw
        )
        WHERE rn <= {top_partners_per_item}
    """)


def build_cooccurrence_candidates(
    con,
    cutoff: str,
    target_customers,
    k: int = 12,
    top_partners_per_item: int = 20,
    seed_map: dict | None = None,
) -> dict:
    """アイテム共起候補: 顧客の自己リピート上位12件をシードにして、それらとよく一緒に買われる商品を集計する。

    seed_mapを渡さない場合はbuild_repeat_candidatesで内部計算する(単体ソースとしての検証用)。
    """
    build_item_cooccurrence_table(con, cutoff, top_partners_per_item)
    if seed_map is None:
        seed_map = build_repeat_candidates(con, cutoff, k=12)

    target_customers = list({int(c) for c in target_customers})
    seed_rows = [
        (cid, int(a)) for cid in target_customers for a in seed_map.get(cid, [])
    ]
    if not seed_rows:
        return {}

    seed_df = pd.DataFrame(seed_rows, columns=["customer_idx", "article_a"])
    con.register("seed_df", seed_df)
    result = con.execute(f"""
        WITH scored AS (
            SELECT s.customer_idx, t.article_b AS article_id, SUM(t.cnt) AS score
            FROM seed_df s
            JOIN cooc_top t ON t.article_a = s.article_a
            GROUP BY s.customer_idx, t.article_b
        ),
        ranked AS (
            SELECT customer_idx, article_id,
                   ROW_NUMBER() OVER (PARTITION BY customer_idx ORDER BY score DESC) AS rn
            FROM scored
        )
        SELECT customer_idx, LIST(article_id ORDER BY rn) AS candidates
        FROM ranked
        WHERE rn <= {k}
        GROUP BY customer_idx
    """).df()
    con.unregister("seed_df")
    return dict(zip(result.customer_idx, result.candidates))


def build_cf_candidates(
    con,
    cutoff: str,
    target_customers,
    k: int = 12,
    factors: int = 50,
    iterations: int = 12,
    alpha: float = 40.0,
    random_state: int = 42,
) -> dict:
    """協調フィルタリング候補: cutoff以前の購入回数を信頼度としたimplicit ALS。"""
    interactions = con.execute(f"""
        SELECT customer_idx, article_id, COUNT(*) AS cnt
        FROM trans
        WHERE t_dat <= DATE '{cutoff}'
        GROUP BY customer_idx, article_id
    """).df()

    item_ids = interactions["article_id"].unique()
    item_id_to_idx = {a: i for i, a in enumerate(item_ids)}
    idx_to_item_id = {i: a for a, i in item_id_to_idx.items()}
    interactions["item_idx"] = interactions["article_id"].map(item_id_to_idx)

    n_users = int(con.execute("SELECT COUNT(*) FROM cust_map").fetchone()[0])
    n_items = len(item_ids)

    confidence = 1.0 + alpha * interactions["cnt"].to_numpy(dtype=np.float32)
    user_items = coo_matrix(
        (confidence, (interactions["customer_idx"].to_numpy(), interactions["item_idx"].to_numpy())),
        shape=(n_users, n_items),
    ).tocsr()

    model = implicit.als.AlternatingLeastSquares(
        factors=factors, regularization=0.01, iterations=iterations, random_state=random_state
    )
    model.fit(user_items)

    target_customers = np.array(sorted(set(int(c) for c in target_customers)))
    target_customers = target_customers[target_customers < n_users]

    rec_ids, _ = model.recommend(
        target_customers, user_items[target_customers], N=k, filter_already_liked_items=False
    )

    return {
        int(cid): [int(idx_to_item_id[i]) for i in rec_ids[row_i]]
        for row_i, cid in enumerate(target_customers)
    }


def build_candidate_pool(con, cutoff: str, target_customers, k_per_source: int = 12) -> pd.DataFrame:
    """6ソースを統合し、(customer_idx, article_id)ごとにどのソース由来かのフラグを持つDataFrameを返す。

    候補ソースG(build_price_band_popularity_candidates, H5価格帯フィルタ人気)はbacktestは改善したが
    実LB(Public/Private)が悪化したため不採用(00_md/pipeline_build_log.md参照)。関数自体は残す。
    """
    target_customers = list({int(c) for c in target_customers})

    repeat_map = build_repeat_candidates(con, cutoff, target_customers=target_customers, k=k_per_source)
    pop_list = build_popularity_candidates(con, cutoff, top_n=k_per_source)
    category_map = build_category_color_candidates(con, cutoff, target_customers=target_customers, k=k_per_source)
    cf_map = build_cf_candidates(con, cutoff, target_customers, k_per_source)
    cooc_map = build_cooccurrence_candidates(con, cutoff, target_customers, k_per_source, seed_map=repeat_map)
    content_map = build_content_similarity_candidates(
        con, cutoff, target_customers, k=k_per_source, seed_map=repeat_map
    )

    rows = {}
    for cid in target_customers:
        sources = {
            "from_repeat": repeat_map.get(cid, []),
            "from_trend": pop_list,
            "from_category_color": category_map.get(cid, []),
            "from_cf": cf_map.get(cid, []),
            "from_cooccurrence": cooc_map.get(cid, []),
            "from_content_similarity": content_map.get(cid, []),
        }
        for src_name, articles in sources.items():
            for article_id in articles:
                key = (cid, int(article_id))
                if key not in rows:
                    rows[key] = {
                        "customer_idx": cid,
                        "article_id": int(article_id),
                        "from_repeat": 0,
                        "from_trend": 0,
                        "from_category_color": 0,
                        "from_cf": 0,
                        "from_cooccurrence": 0,
                        "from_content_similarity": 0,
                    }
                rows[key][src_name] = 1

    return pd.DataFrame(rows.values())
