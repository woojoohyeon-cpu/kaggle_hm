"""候補×特徴量テーブル構築(04_feature_engineering.ipynbから抽出)。

user系・item系・user-item交差系・時間系の特徴量を作る。05/06/08/09・CLIから共通で使う。

## 特徴量セット

- `BASE_0906_FEATURES`: 現行ベスト(model_20260906_0924_ranker, Private LB 0.02620)が使っている35列。
  既存の列名・定義は一切変えていない(0906版モデルでそのままスコアリングできる)。
- `TREND_FEATURES`: 短期トレンド仮説(00_md/pipeline_build_log.md「短期トレンド仮説での再設計」節)で追加した列。
  「ファストファッションでは季節性(前年同週)より、直近数日〜数週の流行の勢い・新しさ・同年代での流行が効く」
  という仮説に対応する。商品の勢い(1/3/14日販売数、前週比、指数減衰スコア、購入者数、順位)、
  新しさ(発売からの日数、ユーザーの新作志向)、値下げ(直近価格/通常価格)、年齢層内の流行
  (同年齢層での直近7日販売数・シェア・順位)、売場(section)単位の前週比。
- `YOY_FEATURES`: 0917版の季節性特徴量。季節横断holdoutで効果なしと判断したため既定では計算しない
  (include_yoy=Trueで従来どおり計算できる。0917版モデルの再評価用)。

既定(include_trend=False, include_yoy=False, extra_sources=())では0906版と同じ列構成になる
(04/05/06を再実行したときの挙動を変えないため)。新しい列はrun_trend_experiment.pyなどで明示的に有効化する。

## 実装方針(メモリ)

全ての集計・結合・ラベル付与・型の縮小(TINYINT/INTEGER/FLOAT)をduckdb内で行い、
`write_feature_table` はそのまま `COPY ... TO parquet` で書き出す(Python側に巨大なDataFrameを作らない)。
以前の実装は `.df()` で文字列列をobject dtypeのまま3,000万行展開し、さらにpandasでラベルをmergeしていたため、
ColabのHigh-RAM(51GB)でもcutoff=2020-07-28でOOM killされていた。
`build_feature_table` は後方互換用で、同じSQLの結果をArrow経由でpandasに変換して返す。
"""

import pandas as pd

from candidates import (
    BASE_SOURCE_COLS,
    EXTRA_SOURCE_COLS,
    age_bucket_sql,
    build_candidate_pool_table,
    ensure_cust_age,
)

CAT_COLS = ["product_group_name", "index_group_name", "section_name"]

BASE_0906_FEATURES = [
    "from_repeat", "from_trend", "from_category_color", "from_cf", "from_cooccurrence",
    "from_content_similarity", "n_sources",
    "product_group_name", "index_group_name", "section_name",
    "u_total_purchases", "u_distinct_articles", "u_distinct_product_codes", "u_avg_price", "u_price_cv",
    "u_days_since_last_purchase", "u_online_ratio", "u_active_weeks",
    "u_menswear_share", "u_babychildren_share", "u_ladieswear_share",
    "u_purchases_7d", "u_purchases_28d", "u_purchases_91d",
    "i_total_sales", "i_avg_price", "i_days_since_last_sale", "i_sales_7d", "i_sales_28d",
    "ui_times_bought", "ui_days_since_last_bought", "ui_times_bought_product_code",
    "ui_days_since_last_bought_product_code", "ui_price_diff",
    "target_week_start_dom",
]

TREND_FEATURES = [
    # 商品の勢い
    "i_sales_1d", "i_sales_3d", "i_sales_14d", "i_sales_prev7d", "i_wow_ratio", "i_sales_decay",
    "i_buyers_7d", "i_rank_7d",
    # 新しさ・値下げ
    "i_days_since_first_sale", "i_price_ratio_7d",
    # 同年齢層での流行
    "u_age", "ia_sales_7d", "ia_share_7d", "ia_rank_7d",
    # 売場単位の勢い
    "sec_sales_7d", "sec_wow_ratio",
    # ユーザーの新作志向
    "u_item_age_at_purchase_avg", "u_new_item_share",
]

YOY_FEATURES = ["i_sales_yoy_7d", "cat_sales_yoy_7d", "cat_sales_recent_28d", "cat_seasonal_lift"]


def _d(cutoff: str, days: int = 0) -> str:
    """cutoffからdays日前の日付を表すSQL式。"""
    return f"DATE '{cutoff}'" if days == 0 else f"(DATE '{cutoff}' - INTERVAL {days} DAY)"


def prepare_feature_query(
    con,
    cutoff: str,
    label_days: int = 7,
    trailing_days: int = 56,
    k_per_source=None,
    with_label: bool = True,
    extra_sources=(),
    include_trend: bool = False,
    include_yoy: bool = False,
    verbose: bool = True,
) -> str:
    """候補プールと各特徴量テーブルをduckdb内に作り、最終テーブルを返すSELECT文を返す。"""
    c = cutoff

    # --- 対象顧客: cutoff以前trailing_days日以内の購入者 ∪ (学習時は)ラベル期間の購入者 ---
    label_customers_sql = (
        f"UNION SELECT DISTINCT customer_idx FROM trans "
        f"WHERE t_dat > {_d(c)} AND t_dat <= {_d(c)} + INTERVAL {label_days} DAY"
        if with_label else ""
    )
    con.execute(f"""
        CREATE OR REPLACE TABLE _targets AS
        SELECT DISTINCT customer_idx FROM trans
        WHERE t_dat <= {_d(c)} AND t_dat > {_d(c, trailing_days)}
        {label_customers_sql}
    """)
    target_customers = con.execute("SELECT customer_idx FROM _targets").fetchnumpy()["customer_idx"].tolist()
    if verbose:
        print(f"  [{c}] 対象顧客数: {len(target_customers):,}", flush=True)

    n_pairs = build_candidate_pool_table(
        con, c, target_customers, k_per_source, extra_sources=extra_sources, verbose=verbose,
    )
    del target_customers, n_pairs
    con.execute("CREATE OR REPLACE TABLE _pool_customers AS SELECT DISTINCT customer_idx FROM candidate_pool")

    if with_label:
        con.execute(f"""
            CREATE OR REPLACE TABLE _labels AS
            SELECT DISTINCT customer_idx, article_id FROM trans
            WHERE t_dat > {_d(c)} AND t_dat <= {_d(c)} + INTERVAL {label_days} DAY
        """)

    pool_filter = "customer_idx IN (SELECT customer_idx FROM _pool_customers)"

    # --- user系(候補プールに登場する顧客だけ集計する) ---
    con.execute(f"""
        CREATE OR REPLACE TABLE user_feat AS
        WITH ut AS (
            SELECT * FROM trans WHERE t_dat <= {_d(c)} AND {pool_filter}
        ),
        base AS (
            SELECT customer_idx,
                   COUNT(*) AS u_total_purchases,
                   COUNT(DISTINCT article_id) AS u_distinct_articles,
                   COUNT(DISTINCT product_code) AS u_distinct_product_codes,
                   AVG(price) AS u_avg_price,
                   STDDEV(price) / NULLIF(AVG(price), 0) AS u_price_cv,
                   MAX(t_dat) AS u_last_purchase_dat,
                   AVG(CASE WHEN sales_channel_id = 2 THEN 1.0 ELSE 0.0 END) AS u_online_ratio,
                   COUNT(DISTINCT DATE_TRUNC('week', t_dat)) AS u_active_weeks,
                   AVG(CASE WHEN index_group_name = 'Menswear' THEN 1.0 ELSE 0.0 END) AS u_menswear_share,
                   AVG(CASE WHEN index_group_name = 'Baby/Children' THEN 1.0 ELSE 0.0 END) AS u_babychildren_share,
                   AVG(CASE WHEN index_group_name = 'Ladieswear' THEN 1.0 ELSE 0.0 END) AS u_ladieswear_share,
                   COUNT(*) FILTER (WHERE t_dat > {_d(c, 7)}) AS u_purchases_7d,
                   COUNT(*) FILTER (WHERE t_dat > {_d(c, 28)}) AS u_purchases_28d,
                   COUNT(*) FILTER (WHERE t_dat > {_d(c, 91)}) AS u_purchases_91d
            FROM ut
            GROUP BY customer_idx
        )
        SELECT b.* EXCLUDE (u_last_purchase_dat),
               DATE_DIFF('day', b.u_last_purchase_dat, {_d(c)}) AS u_days_since_last_purchase
        FROM base b
    """)

    # --- item系(全商品) ---
    yoy_cte = f""",
        yoy AS (
            -- target週(cutoff, cutoff+7]のちょうど1年前にあたる週の売上(0917版の季節性特徴量)
            SELECT article_id, COUNT(*) AS i_sales_yoy_7d
            FROM trans
            WHERE t_dat > {_d(c, 365)} AND t_dat <= {_d(c, 358)}
            GROUP BY article_id
        )""" if include_yoy else ""
    con.execute(f"""
        CREATE OR REPLACE TABLE item_feat AS
        WITH base AS (
            SELECT article_id,
                   COUNT(*) AS i_total_sales,
                   AVG(price) AS i_avg_price,
                   MAX(t_dat) AS i_last_sale_dat,
                   MIN(t_dat) AS i_first_sale_dat
            FROM trans
            WHERE t_dat <= {_d(c)}
            GROUP BY article_id
        ),
        recent AS (
            SELECT article_id,
                   COUNT(*) FILTER (WHERE t_dat > {_d(c, 1)}) AS i_sales_1d,
                   COUNT(*) FILTER (WHERE t_dat > {_d(c, 3)}) AS i_sales_3d,
                   COUNT(*) FILTER (WHERE t_dat > {_d(c, 7)}) AS i_sales_7d,
                   COUNT(*) FILTER (WHERE t_dat <= {_d(c, 7)} AND t_dat > {_d(c, 14)}) AS i_sales_prev7d,
                   COUNT(*) FILTER (WHERE t_dat > {_d(c, 14)}) AS i_sales_14d,
                   COUNT(*) AS i_sales_28d,
                   COUNT(DISTINCT customer_idx) FILTER (WHERE t_dat > {_d(c, 7)}) AS i_buyers_7d,
                   SUM(EXP(-DATE_DIFF('day', t_dat, {_d(c)}) / 7.0)) AS i_sales_decay,
                   AVG(price) FILTER (WHERE t_dat > {_d(c, 7)}) AS i_avg_price_7d
            FROM trans
            WHERE t_dat <= {_d(c)} AND t_dat > {_d(c, 28)}
            GROUP BY article_id
        ){yoy_cte}
        SELECT b.article_id, b.i_total_sales, b.i_avg_price,
               DATE_DIFF('day', b.i_last_sale_dat, {_d(c)}) AS i_days_since_last_sale,
               -- データ開始日(2018-09-20)より前の発売日は観測できないため365日で頭打ちにする
               -- (cutoffが2019年だと最大約366日、2020年だと約700日になり、時期によって分布がずれるのを防ぐ)
               LEAST(DATE_DIFF('day', b.i_first_sale_dat, {_d(c)}), 365) AS i_days_since_first_sale,
               COALESCE(r.i_sales_1d, 0) AS i_sales_1d,
               COALESCE(r.i_sales_3d, 0) AS i_sales_3d,
               COALESCE(r.i_sales_7d, 0) AS i_sales_7d,
               COALESCE(r.i_sales_prev7d, 0) AS i_sales_prev7d,
               COALESCE(r.i_sales_14d, 0) AS i_sales_14d,
               COALESCE(r.i_sales_28d, 0) AS i_sales_28d,
               COALESCE(r.i_buyers_7d, 0) AS i_buyers_7d,
               COALESCE(r.i_sales_decay, 0) AS i_sales_decay,
               (COALESCE(r.i_sales_7d, 0) + 1.0) / (COALESCE(r.i_sales_prev7d, 0) + 1.0) AS i_wow_ratio,
               r.i_avg_price_7d / NULLIF(b.i_avg_price, 0) AS i_price_ratio_7d,
               CASE WHEN COALESCE(r.i_sales_7d, 0) > 0
                    THEN RANK() OVER (ORDER BY COALESCE(r.i_sales_7d, 0) DESC) END AS i_rank_7d
               {', COALESCE(y.i_sales_yoy_7d, 0) AS i_sales_yoy_7d' if include_yoy else ''}
        FROM base b
        LEFT JOIN recent r USING (article_id)
        {'LEFT JOIN yoy y USING (article_id)' if include_yoy else ''}
    """)

    trend_selects = ""
    trend_joins = ""
    if include_trend:
        ensure_cust_age(con)
        # 同年齢層での直近7日の流行
        con.execute(f"""
            CREATE OR REPLACE TABLE ia_feat AS
            SELECT age_bucket, article_id, ia_sales_7d,
                   ia_sales_7d * 1.0 / SUM(ia_sales_7d) OVER (PARTITION BY age_bucket) AS ia_share_7d,
                   RANK() OVER (PARTITION BY age_bucket ORDER BY ia_sales_7d DESC) AS ia_rank_7d
            FROM (
                SELECT {age_bucket_sql('age')} AS age_bucket, article_id, COUNT(*) AS ia_sales_7d
                FROM trans
                WHERE t_dat <= {_d(c)} AND t_dat > {_d(c, 7)}
                GROUP BY 1, 2
            )
        """)
        # 売場(section)単位の前週比
        con.execute(f"""
            CREATE OR REPLACE TABLE sec_feat AS
            SELECT a.section_name,
                   COUNT(*) FILTER (WHERE t.t_dat > {_d(c, 7)}) AS sec_sales_7d,
                   COUNT(*) FILTER (WHERE t.t_dat <= {_d(c, 7)}) AS sec_sales_prev7d
            FROM trans t
            JOIN articles a ON a.article_id = t.article_id
            WHERE t.t_dat <= {_d(c)} AND t.t_dat > {_d(c, 14)}
            GROUP BY a.section_name
        """)
        # ユーザーの新作志向: 直近91日に買った商品が、その時点で発売から何日目だったか
        con.execute(f"""
            CREATE OR REPLACE TABLE user_trend_feat AS
            WITH first_sale AS (
                SELECT article_id, MIN(t_dat) AS first_dat FROM trans WHERE t_dat <= {_d(c)} GROUP BY article_id
            )
            SELECT t.customer_idx,
                   AVG(LEAST(DATE_DIFF('day', f.first_dat, t.t_dat), 365)) AS u_item_age_at_purchase_avg,
                   AVG(CASE WHEN DATE_DIFF('day', f.first_dat, t.t_dat) <= 14 THEN 1.0 ELSE 0.0 END)
                       AS u_new_item_share
            FROM trans t
            JOIN first_sale f USING (article_id)
            WHERE t.t_dat <= {_d(c)} AND t.t_dat > {_d(c, 91)} AND t.{pool_filter}
            GROUP BY t.customer_idx
        """)
        trend_selects = """,
            COALESCE(itf.i_sales_1d, 0)::INTEGER AS i_sales_1d,
            COALESCE(itf.i_sales_3d, 0)::INTEGER AS i_sales_3d,
            COALESCE(itf.i_sales_14d, 0)::INTEGER AS i_sales_14d,
            COALESCE(itf.i_sales_prev7d, 0)::INTEGER AS i_sales_prev7d,
            COALESCE(itf.i_wow_ratio, 1.0)::FLOAT AS i_wow_ratio,
            COALESCE(itf.i_sales_decay, 0)::FLOAT AS i_sales_decay,
            COALESCE(itf.i_buyers_7d, 0)::INTEGER AS i_buyers_7d,
            itf.i_rank_7d::FLOAT AS i_rank_7d,
            itf.i_days_since_first_sale::FLOAT AS i_days_since_first_sale,
            itf.i_price_ratio_7d::FLOAT AS i_price_ratio_7d,
            ca.age::FLOAT AS u_age,
            COALESCE(ia.ia_sales_7d, 0)::INTEGER AS ia_sales_7d,
            COALESCE(ia.ia_share_7d, 0)::FLOAT AS ia_share_7d,
            ia.ia_rank_7d::FLOAT AS ia_rank_7d,
            COALESCE(sf.sec_sales_7d, 0)::INTEGER AS sec_sales_7d,
            ((COALESCE(sf.sec_sales_7d, 0) + 1.0) / (COALESCE(sf.sec_sales_prev7d, 0) + 1.0))::FLOAT AS sec_wow_ratio,
            utf.u_item_age_at_purchase_avg::FLOAT AS u_item_age_at_purchase_avg,
            utf.u_new_item_share::FLOAT AS u_new_item_share"""
        trend_joins = """
        LEFT JOIN cust_age ca ON ca.customer_idx = cp.customer_idx
        LEFT JOIN ia_feat ia ON ia.age_bucket = ca.age_bucket AND ia.article_id = cp.article_id
        LEFT JOIN sec_feat sf ON sf.section_name = a.section_name
        LEFT JOIN user_trend_feat utf ON utf.customer_idx = cp.customer_idx"""

    yoy_selects = ""
    yoy_joins = ""
    if include_yoy:
        con.execute(f"""
            CREATE OR REPLACE TABLE cat_feat AS
            WITH yoy AS (
                SELECT a.section_name, COUNT(*) AS cnt
                FROM trans t JOIN articles a ON a.article_id = t.article_id
                WHERE t.t_dat > {_d(c, 365)} AND t.t_dat <= {_d(c, 358)}
                GROUP BY a.section_name
            ),
            recent AS (
                SELECT a.section_name, COUNT(*) AS cnt
                FROM trans t JOIN articles a ON a.article_id = t.article_id
                WHERE t.t_dat <= {_d(c)} AND t.t_dat > {_d(c, 28)}
                GROUP BY a.section_name
            )
            SELECT COALESCE(y.section_name, r.section_name) AS section_name,
                   COALESCE(y.cnt, 0) AS cat_sales_yoy_7d,
                   COALESCE(r.cnt, 0) AS cat_sales_recent_28d
            FROM yoy y
            FULL OUTER JOIN recent r USING (section_name)
        """)
        yoy_selects = """,
            COALESCE(itf.i_sales_yoy_7d, 0)::INTEGER AS i_sales_yoy_7d,
            COALESCE(cf.cat_sales_yoy_7d, 0)::INTEGER AS cat_sales_yoy_7d,
            COALESCE(cf.cat_sales_recent_28d, 0)::INTEGER AS cat_sales_recent_28d,
            (COALESCE(cf.cat_sales_yoy_7d, 0) / NULLIF(COALESCE(cf.cat_sales_recent_28d, 0) / 4.0, 0))::FLOAT
                AS cat_seasonal_lift"""
        yoy_joins = "\n        LEFT JOIN cat_feat cf ON cf.section_name = a.section_name"

    # --- user-item交差系(候補プールに登場する顧客だけ) ---
    con.execute(f"""
        CREATE OR REPLACE TABLE ui_feat AS
        SELECT customer_idx, article_id,
               COUNT(*) AS ui_times_bought,
               DATE_DIFF('day', MAX(t_dat), {_d(c)}) AS ui_days_since_last_bought
        FROM trans
        WHERE t_dat <= {_d(c)} AND {pool_filter}
        GROUP BY customer_idx, article_id
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE ui_product_feat AS
        SELECT customer_idx, product_code,
               COUNT(*) AS ui_times_bought_product_code,
               DATE_DIFF('day', MAX(t_dat), {_d(c)}) AS ui_days_since_last_bought_product_code
        FROM trans
        WHERE t_dat <= {_d(c)} AND {pool_filter}
        GROUP BY customer_idx, product_code
    """)

    target_week_start_dom = (pd.Timestamp(c) + pd.Timedelta(days=1)).day
    extra_flag_selects = "".join(
        f",\n            cp.{EXTRA_SOURCE_COLS[s]}" for s in extra_sources
    )
    label_select = ",\n            COALESCE(lb.label, 0)::TINYINT AS label" if with_label else ""
    label_join = (
        "\n        LEFT JOIN (SELECT customer_idx, article_id, 1 AS label FROM _labels) lb"
        "\n            ON lb.customer_idx = cp.customer_idx AND lb.article_id = cp.article_id"
        if with_label else ""
    )
    n_sources_expr = " + ".join(f"cp.{s}" for s in BASE_SOURCE_COLS)

    return f"""
        SELECT
            cp.customer_idx::INTEGER AS customer_idx,
            cp.article_id::INTEGER AS article_id,
            cp.from_repeat, cp.from_trend, cp.from_category_color, cp.from_cf, cp.from_cooccurrence,
            cp.from_content_similarity{extra_flag_selects},
            ({n_sources_expr})::TINYINT AS n_sources,

            a.product_code::INTEGER AS product_code,
            a.product_group_name, a.index_group_name, a.section_name,

            COALESCE(uf.u_total_purchases, 0)::INTEGER AS u_total_purchases,
            COALESCE(uf.u_distinct_articles, 0)::INTEGER AS u_distinct_articles,
            COALESCE(uf.u_distinct_product_codes, 0)::INTEGER AS u_distinct_product_codes,
            uf.u_avg_price::FLOAT AS u_avg_price,
            uf.u_price_cv::FLOAT AS u_price_cv,
            uf.u_days_since_last_purchase::FLOAT AS u_days_since_last_purchase,
            uf.u_online_ratio::FLOAT AS u_online_ratio,
            COALESCE(uf.u_active_weeks, 0)::INTEGER AS u_active_weeks,
            uf.u_menswear_share::FLOAT AS u_menswear_share,
            uf.u_babychildren_share::FLOAT AS u_babychildren_share,
            uf.u_ladieswear_share::FLOAT AS u_ladieswear_share,
            COALESCE(uf.u_purchases_7d, 0)::INTEGER AS u_purchases_7d,
            COALESCE(uf.u_purchases_28d, 0)::INTEGER AS u_purchases_28d,
            COALESCE(uf.u_purchases_91d, 0)::INTEGER AS u_purchases_91d,

            COALESCE(itf.i_total_sales, 0)::INTEGER AS i_total_sales,
            itf.i_avg_price::FLOAT AS i_avg_price,
            COALESCE(itf.i_days_since_last_sale, 9999)::INTEGER AS i_days_since_last_sale,
            COALESCE(itf.i_sales_7d, 0)::INTEGER AS i_sales_7d,
            COALESCE(itf.i_sales_28d, 0)::INTEGER AS i_sales_28d,

            COALESCE(uif.ui_times_bought, 0)::INTEGER AS ui_times_bought,
            uif.ui_days_since_last_bought::FLOAT AS ui_days_since_last_bought,
            COALESCE(upf.ui_times_bought_product_code, 0)::INTEGER AS ui_times_bought_product_code,
            upf.ui_days_since_last_bought_product_code::FLOAT AS ui_days_since_last_bought_product_code,
            (itf.i_avg_price - uf.u_avg_price)::FLOAT AS ui_price_diff,

            {target_week_start_dom}::INTEGER AS target_week_start_dom{trend_selects}{yoy_selects}{label_select},
            '{c}' AS cutoff
        FROM candidate_pool cp
        LEFT JOIN articles a ON a.article_id = cp.article_id
        LEFT JOIN user_feat uf ON uf.customer_idx = cp.customer_idx
        LEFT JOIN item_feat itf ON itf.article_id = cp.article_id
        LEFT JOIN ui_feat uif ON uif.customer_idx = cp.customer_idx AND uif.article_id = cp.article_id
        LEFT JOIN ui_product_feat upf
            ON upf.customer_idx = cp.customer_idx AND upf.product_code = a.product_code{trend_joins}{yoy_joins}{label_join}
    """


def _cleanup(con) -> None:
    for t in ["candidate_pool", "_pool_customers", "_labels", "_targets", "user_feat", "item_feat", "ia_feat",
              "sec_feat", "user_trend_feat", "cat_feat", "ui_feat", "ui_product_feat", "cooc_top", "cooc_raw",
              "_age_trend_top"]:
        con.execute(f"DROP TABLE IF EXISTS {t}")


def _to_arrow_table(con, sql: str):
    res = con.execute(sql)
    if hasattr(res, "to_arrow_table"):  # duckdb >= 1.4
        return res.to_arrow_table()
    return res.fetch_arrow_table()


def write_feature_table(con, cutoff: str, out_path: str, **kwargs) -> int:
    """1cutoff分の候補×特徴量(×ラベル)テーブルを、Pythonを経由せずparquetに直接書き出して行数を返す。

    kwargs は prepare_feature_query と同じ(label_days, trailing_days, k_per_source, with_label,
    extra_sources, include_trend, include_yoy, verbose)。行順は保証しない(使う側でソートする)。
    """
    sql = prepare_feature_query(con, cutoff, **kwargs)
    # 行順は不要(読み込み側で並べ替える)。全体ソートや挿入順の保持をしないことでメモリとスピルを減らす
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"COPY ({sql}) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.execute("SET preserve_insertion_order = true")
    n_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path}')").fetchone()[0]
    _cleanup(con)
    return n_rows


def build_feature_table(
    con,
    cutoff: str,
    label_days: int = 7,
    trailing_days: int = 56,
    k_per_source=None,
    with_label: bool = True,
    extra_sources=(),
    include_trend: bool = False,
    include_yoy: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """後方互換用: 同じテーブルをpandas DataFrame(カテゴリ列はcategory dtype)で返す。

    0906版モデル(BASE_0906_FEATURES)は追加列を無視するのでそのままスコアリングできる。
    0917版モデルを再評価する場合は include_yoy=True を渡すこと。
    大規模データでは write_feature_table でparquetに書き出してから必要な列だけ読む方が安全。
    """
    sql = prepare_feature_query(
        con, cutoff, label_days=label_days, trailing_days=trailing_days, k_per_source=k_per_source,
        with_label=with_label, extra_sources=extra_sources, include_trend=include_trend,
        include_yoy=include_yoy, verbose=verbose,
    )
    table = _to_arrow_table(con, sql)
    _cleanup(con)
    df = table.to_pandas(categories=CAT_COLS)
    del table
    return df
