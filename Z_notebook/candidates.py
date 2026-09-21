"""候補生成ロジック。

既定の6ソース: A自己リピート / Bトレンド人気(全体) / C同カテゴリ同色 / D協調フィルタリング(ALS) /
Eアイテム共起 / Fテキストコンテンツ類似。
追加ソース(extra_sourcesで明示的に有効化した時のみ): H年齢層別トレンド人気(`age_trend`)。

03_candidate_generation.ipynb でソースごとのRecall@12/MAP@12を検証済み(参照: 00_md/hypothesis_driven_design.md)。
04以降のノートブック・CLIはこのモジュールを共有して使う。

候補プールの統合(`build_candidate_pool_table`)は、(customer_idx, article_id, source)の縦持ち配列を作ってから
duckdbでGROUP BYしてフラグ列に展開する。以前の「(cid, aid)をキーにしたPython辞書の辞書」方式は
3,000万ペア規模でPythonオブジェクトだけで十数GBを消費し、Colab(スワップなし)でOOM killされる主因だった
(00_md/pipeline_build_log.md「短期トレンド仮説での再設計」節参照)。出力の中身(ペア集合とフラグ)は同一。
"""

import itertools
import os

import numpy as np
import pandas as pd
import implicit
from scipy.sparse import coo_matrix

from common import DATA_DIR

# 候補プールのフラグ列。順序はbuild_candidate_pool(旧実装)の列順と同じ。
BASE_SOURCE_COLS = [
    "from_repeat",
    "from_trend",
    "from_category_color",
    "from_cf",
    "from_cooccurrence",
    "from_content_similarity",
]
EXTRA_SOURCE_COLS = {"age_trend": "from_age_trend"}

# 年齢層の区切り。NULL(年齢不明)は-1として独立した層にする。
AGE_BUCKET_EDGES = [20, 25, 30, 40, 50, 60]


def age_bucket_sql(col: str) -> str:
    """年齢列を年齢層番号(-1: 不明, 0: <20, 1: 20-24, ..., 6: 60+)に変換するSQL式。"""
    cases = " ".join(f"WHEN {col} < {edge} THEN {i}" for i, edge in enumerate(AGE_BUCKET_EDGES))
    return f"(CASE WHEN {col} IS NULL THEN -1 {cases} ELSE {len(AGE_BUCKET_EDGES)} END)"


def ensure_cust_age(con) -> None:
    """customer_idx -> (age, age_bucket) の対応表 cust_age をduckdbに作る(既にあれば何もしない)。

    cust_mapはget_con()で作られている前提。trans_liteにもage列はあるが、期間内に購入のない顧客
    (コールドスタート)の年齢も必要なため customers.csv から作る。
    """
    exists = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'cust_age'"
    ).fetchone()[0]
    if exists:
        return
    con.execute(f"""
        CREATE TABLE cust_age AS
        SELECT cm.customer_idx, c.age::SMALLINT AS age, {age_bucket_sql('c.age')}::TINYINT AS age_bucket
        FROM cust_map cm
        LEFT JOIN read_csv_auto('{DATA_DIR}/customers.csv') c USING (customer_id)
    """)


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
        top = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]
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
                   ROW_NUMBER() OVER (PARTITION BY customer_idx ORDER BY last_dat DESC, freq DESC, article_id) AS rn
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
        ORDER BY cnt DESC, article_id
        LIMIT {top_n}
    """).df()
    return df.article_id.tolist()


def build_age_trend_table(con, cutoff: str, k: int = 12, window_days: int = 7, table_name: str = "_age_trend_top") -> str:
    """年齢層別トレンド人気(H): cutoff直前window_days日間に、同じ年齢層の顧客に最も売れた上位k件。

    短期トレンド仮説(ファストファッションは季節より「今この層で流行っているもの」が売れる)に基づく
    候補ソース。全体トレンド(B)は全員に同じ12〜20件を出すが、こちらは年齢層ごとに異なる。
    (age_bucket, article_id, rn) のテーブルを作ってテーブル名を返す。
    """
    con.execute(f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT age_bucket, article_id, rn
        FROM (
            SELECT age_bucket, article_id,
                   ROW_NUMBER() OVER (PARTITION BY age_bucket ORDER BY cnt DESC, article_id) AS rn
            FROM (
                SELECT {age_bucket_sql('age')} AS age_bucket, article_id, COUNT(*) AS cnt
                FROM trans
                WHERE t_dat <= DATE '{cutoff}' AND t_dat > DATE '{cutoff}' - INTERVAL {window_days} DAY
                GROUP BY 1, 2
            )
        )
        WHERE rn <= {k}
    """)
    return table_name


def build_age_trend_candidates(con, cutoff: str, target_customers, k: int = 12, window_days: int = 7) -> dict:
    """年齢層別トレンド人気候補を {customer_idx: [article_id, ...]} で返す(03のような単体ソース検証用)。"""
    ensure_cust_age(con)
    table = build_age_trend_table(con, cutoff, k=k, window_days=window_days)
    target_df = pd.DataFrame({"customer_idx": np.array(sorted({int(c) for c in target_customers}), dtype=np.int32)})
    con.register("_age_targets", target_df)
    df = con.execute(f"""
        SELECT t.customer_idx, LIST(a.article_id ORDER BY a.rn) AS candidates
        FROM _age_targets t
        JOIN cust_age ca USING (customer_idx)
        JOIN {table} a ON a.age_bucket = ca.age_bucket
        GROUP BY t.customer_idx
    """).df()
    con.unregister("_age_targets")
    return dict(zip(df.customer_idx, df.candidates))


def age_trend_lists(con, cutoff: str, k: int = 12, window_days: int = 7) -> dict:
    """{age_bucket: [article_id, ...]}。提出時のコールドスタート顧客向けフォールバックに使う。"""
    table = build_age_trend_table(con, cutoff, k=k, window_days=window_days)
    df = con.execute(f"""
        SELECT age_bucket, LIST(article_id ORDER BY rn) AS items FROM {table} GROUP BY age_bucket
    """).df()
    return {int(b): [int(a) for a in items] for b, items in zip(df.age_bucket, df["items"])}


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
                   ROW_NUMBER() OVER (PARTITION BY cp.customer_idx ORDER BY vp.cnt DESC, vp.article_id) AS rn
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
            SELECT *, ROW_NUMBER() OVER (PARTITION BY customer_idx ORDER BY cnt DESC, section_name) AS rn
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
            SELECT *, ROW_NUMBER() OVER (PARTITION BY section_name ORDER BY cnt DESC, article_id) AS rn
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
            SELECT *, ROW_NUMBER() OVER (PARTITION BY article_a ORDER BY cnt DESC, article_b) AS rn
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
                   ROW_NUMBER() OVER (PARTITION BY customer_idx ORDER BY score DESC, article_id) AS rn
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

    # duckdbのGROUP BY結果は行順が実行ごとに変わるため、商品インデックスの採番を固定する
    # (採番が変わるとALSの初期値の割り当てが変わり、同じrandom_stateでも結果が揺れる)
    item_ids = np.sort(interactions["article_id"].unique())
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

    # スレッド数は HM_ALS_THREADS で指定できる(既定0=全コア。商品インデックスを固定したのでスレッド数によらず再現する)
    model = implicit.als.AlternatingLeastSquares(
        factors=factors, regularization=0.01, iterations=iterations, random_state=random_state,
        num_threads=int(os.environ.get("HM_ALS_THREADS", 0)),
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


DEFAULT_K_PER_SOURCE = {
    "from_repeat": 20,
    "from_trend": 20,
    "from_category_color": 12,
    "from_cf": 12,
    "from_cooccurrence": 12,
    "from_content_similarity": 12,
}


def _map_to_long(candidate_map: dict, src: int) -> pd.DataFrame:
    """{customer_idx: [article_id, ...]} を (customer_idx, article_id, src) の縦持ちDataFrameにする。"""
    if not candidate_map:
        return pd.DataFrame({
            "customer_idx": np.array([], dtype=np.int32),
            "article_id": np.array([], dtype=np.int64),
            "src": np.array([], dtype=np.int8),
        })
    n = len(candidate_map)
    cids = np.fromiter(candidate_map.keys(), dtype=np.int64, count=n)
    lens = np.fromiter((len(v) for v in candidate_map.values()), dtype=np.int64, count=n)
    total = int(lens.sum())
    arts = np.fromiter(itertools.chain.from_iterable(candidate_map.values()), dtype=np.int64, count=total)
    return pd.DataFrame({
        "customer_idx": np.repeat(cids, lens).astype(np.int32),
        "article_id": arts,
        "src": np.full(total, src, dtype=np.int8),
    })


def _resolve_k_map(k_per_source) -> dict:
    if k_per_source is None:
        return dict(DEFAULT_K_PER_SOURCE)
    if isinstance(k_per_source, int):
        return {name: k_per_source for name in DEFAULT_K_PER_SOURCE}
    return {**DEFAULT_K_PER_SOURCE, **k_per_source}


def build_candidate_pool_table(
    con,
    cutoff: str,
    target_customers,
    k_per_source=None,
    seed_k: int = 12,
    extra_sources=(),
    table_name: str = "candidate_pool",
    verbose: bool = False,
) -> int:
    """候補ソースを統合した (customer_idx, article_id, from_*フラグ...) のテーブルをduckdb内に作り、行数を返す。

    k_per_source / seed_k の意味は build_candidate_pool と同じ。
    extra_sources: 既定6ソースに加えて有効化する追加ソース名のタプル。現在は "age_trend" のみ
        (列名 from_age_trend)。既定では空で、既存モデル(0906版など)と同じ6ソースのプールになる。
    """
    unknown = set(extra_sources) - set(EXTRA_SOURCE_COLS)
    if unknown:
        raise ValueError(f"未知の追加ソース: {unknown}")
    k_map = _resolve_k_map(k_per_source)
    src_cols = BASE_SOURCE_COLS + [EXTRA_SOURCE_COLS[s] for s in extra_sources]
    src_code = {name: i for i, name in enumerate(src_cols)}

    target_customers = sorted({int(c) for c in target_customers})

    def log(msg):
        if verbose:
            print(f"  [{cutoff}] {msg}", flush=True)

    parts = []
    repeat_map = build_repeat_candidates(con, cutoff, target_customers=target_customers, k=k_map["from_repeat"])
    log(f"A 自己リピート: {len(repeat_map):,}顧客")
    pop_list = build_popularity_candidates(con, cutoff, top_n=k_map["from_trend"])

    category_map = build_category_color_candidates(
        con, cutoff, target_customers=target_customers, k=k_map["from_category_color"]
    )
    parts.append(_map_to_long(category_map, src_code["from_category_color"]))
    del category_map
    log("C 同カテゴリ/同色: done")

    cf_map = build_cf_candidates(con, cutoff, target_customers, k_map["from_cf"])
    parts.append(_map_to_long(cf_map, src_code["from_cf"]))
    del cf_map
    log("D CF(ALS): done")

    if seed_k == k_map["from_repeat"]:
        seed_map = repeat_map
    else:
        seed_map = build_repeat_candidates(con, cutoff, target_customers=target_customers, k=seed_k)

    cooc_map = build_cooccurrence_candidates(
        con, cutoff, target_customers, k_map["from_cooccurrence"], seed_map=seed_map
    )
    parts.append(_map_to_long(cooc_map, src_code["from_cooccurrence"]))
    del cooc_map
    con.execute("DROP TABLE IF EXISTS cooc_raw")  # 共起の中間テーブルは大きいので即解放
    log("E 共起: done")

    content_map = build_content_similarity_candidates(
        con, cutoff, target_customers, k=k_map["from_content_similarity"], seed_map=seed_map
    )
    parts.append(_map_to_long(content_map, src_code["from_content_similarity"]))
    del content_map, seed_map
    log("F テキスト類似: done")

    parts.append(_map_to_long(repeat_map, src_code["from_repeat"]))
    del repeat_map

    long_df = pd.concat(parts, ignore_index=True)
    del parts
    con.register("_cand_long", long_df)
    con.register("_cand_targets", pd.DataFrame({"customer_idx": np.array(target_customers, dtype=np.int32)}))
    con.register("_cand_pop", pd.DataFrame({"article_id": np.array(pop_list, dtype=np.int64)}))

    union_sql = [
        "SELECT customer_idx, article_id, src FROM _cand_long",
        f"SELECT t.customer_idx, p.article_id, {src_code['from_trend']}::TINYINT AS src "
        "FROM _cand_targets t CROSS JOIN _cand_pop p",
    ]
    if "age_trend" in extra_sources:
        ensure_cust_age(con)
        age_table = build_age_trend_table(con, cutoff, k=k_map.get("from_age_trend", 12))
        union_sql.append(
            f"SELECT t.customer_idx, a.article_id, {src_code['from_age_trend']}::TINYINT AS src "
            f"FROM _cand_targets t JOIN cust_age ca USING (customer_idx) "
            f"JOIN {age_table} a ON a.age_bucket = ca.age_bucket"
        )
    flags = ",\n            ".join(
        f"MAX(CASE WHEN src = {i} THEN 1 ELSE 0 END)::TINYINT AS {name}" for i, name in enumerate(src_cols)
    )
    con.execute(f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT customer_idx::INTEGER AS customer_idx, article_id::BIGINT AS article_id,
            {flags}
        FROM ({' UNION ALL '.join(union_sql)})
        GROUP BY 1, 2
    """)
    for name in ["_cand_long", "_cand_targets", "_cand_pop"]:
        con.unregister(name)
    del long_df
    n_rows = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    log(f"候補プール: {n_rows:,}ペア")
    return n_rows


def build_candidate_pool(con, cutoff: str, target_customers, k_per_source=None, seed_k: int = 12,
                         extra_sources=()) -> pd.DataFrame:
    """候補ソースを統合し、(customer_idx, article_id)ごとにどのソース由来かのフラグを持つDataFrameを返す。

    候補ソースG(build_price_band_popularity_candidates, H5価格帯フィルタ人気)はbacktestは改善したが
    実LB(Public/Private)が悪化したため不採用(00_md/pipeline_build_log.md参照)。関数自体は残す。

    k_per_source: ソースごとの取得件数。Noneなら DEFAULT_K_PER_SOURCE を使う
    (03の単体Recall@12が高かったA:自己リピート/B:トレンド人気を20件に拡張、他は12件のまま)。
    intを渡すと全ソース一律その件数になる(従来の挙動、後方互換用)。dictを渡すと
    一部のソースだけ上書きできる(未指定キーはDEFAULT_K_PER_SOURCEを使う)。
    seed_k: アイテム共起(E)・テキスト類似(F)のシードに使う自己リピート候補の件数。
    from_repeatの出力件数を広げてもシードの質は12件のまま据え置きたい場合などに、
    k_per_source["from_repeat"]とは独立に指定できる。

    実体は build_candidate_pool_table(duckdb内で統合)で、ここではそれをDataFrameとして取り出すだけ。
    大規模データではDataFrame化せず build_candidate_pool_table を直接使うこと(features.pyはそうしている)。
    """
    build_candidate_pool_table(con, cutoff, target_customers, k_per_source, seed_k, extra_sources,
                               table_name="_candidate_pool_df")
    df = con.execute("SELECT * FROM _candidate_pool_df").df()
    con.execute("DROP TABLE _candidate_pool_df")
    return df
