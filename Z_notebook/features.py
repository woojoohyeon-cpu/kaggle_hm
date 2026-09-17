"""候補×特徴量テーブル構築(04_feature_engineering.ipynbから抽出)。

user系・item系・user-item交差系・時間系・季節性系の特徴量を作る。05/06からも共通で使う。

季節性特徴量(i_sales_yoy_7d/cat_sales_yoy_7d/cat_sales_recent_28d/cat_seasonal_lift)は
「target週(cutoff直後7日)のちょうど1年前にあたる週」の実績を使う。articleそのものは毎年
入れ替わる(YoYで存在しないことが多い)ため、より安定したsection_name単位でも同じ発想の
特徴量を用意している。行ごとに値が変わるため、学習cutoffが特定の時期(例: 8〜9月)に
偏っていても学習可能(target_week_start_domのような「暦上の位置そのもの」を表す特徴量は、
学習cutoffの時期にバリエーションがないと木が分岐できず実質無意味になる点と対照的)。
"""

import pandas as pd

from candidates import build_candidate_pool
from common import active_customers, label_window


def build_feature_table(
    con,
    cutoff: str,
    label_days: int = 7,
    trailing_days: int = 56,
    k_per_source=None,
    with_label: bool = True,
) -> pd.DataFrame:
    target_customers = set(active_customers(con, cutoff, trailing_days))
    label_map = None
    if with_label:
        label_map = label_window(con, cutoff, label_days)
        target_customers |= set(label_map.keys())

    print(f"  [{cutoff}] 対象顧客数: {len(target_customers):,}")

    candidate_pool = build_candidate_pool(con, cutoff, target_customers, k_per_source)
    print(f"  [{cutoff}] 候補ペア数: {len(candidate_pool):,}")
    con.register("candidate_pool", candidate_pool)

    con.execute(f"""
        CREATE OR REPLACE TABLE user_feat AS
        WITH base AS (
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
                   AVG(CASE WHEN index_group_name = 'Ladieswear' THEN 1.0 ELSE 0.0 END) AS u_ladieswear_share
            FROM trans
            WHERE t_dat <= DATE '{cutoff}'
            GROUP BY customer_idx
        ),
        recent7 AS (
            SELECT customer_idx, COUNT(*) AS u_purchases_7d
            FROM trans
            WHERE t_dat <= DATE '{cutoff}' AND t_dat > DATE '{cutoff}' - INTERVAL 7 DAY
            GROUP BY customer_idx
        ),
        recent28 AS (
            SELECT customer_idx, COUNT(*) AS u_purchases_28d
            FROM trans
            WHERE t_dat <= DATE '{cutoff}' AND t_dat > DATE '{cutoff}' - INTERVAL 28 DAY
            GROUP BY customer_idx
        ),
        recent91 AS (
            SELECT customer_idx, COUNT(*) AS u_purchases_91d
            FROM trans
            WHERE t_dat <= DATE '{cutoff}' AND t_dat > DATE '{cutoff}' - INTERVAL 91 DAY
            GROUP BY customer_idx
        )

        SELECT b.customer_idx, b.u_total_purchases, b.u_distinct_articles, b.u_distinct_product_codes,
               b.u_avg_price, b.u_price_cv,
               DATE_DIFF('day', b.u_last_purchase_dat, DATE '{cutoff}') AS u_days_since_last_purchase,
               b.u_online_ratio, b.u_active_weeks,
               b.u_menswear_share, b.u_babychildren_share, b.u_ladieswear_share,
               COALESCE(r7.u_purchases_7d, 0) AS u_purchases_7d,
               COALESCE(r28.u_purchases_28d, 0) AS u_purchases_28d,
               COALESCE(r91.u_purchases_91d, 0) AS u_purchases_91d
        FROM base b
        LEFT JOIN recent7 r7 USING (customer_idx)
        LEFT JOIN recent28 r28 USING (customer_idx)
        LEFT JOIN recent91 r91 USING (customer_idx)
    """)

    con.execute(f"""
        CREATE OR REPLACE TABLE item_feat AS
        WITH base AS (
            SELECT article_id,
                   COUNT(*) AS i_total_sales,
                   AVG(price) AS i_avg_price,
                   MAX(t_dat) AS i_last_sale_dat
            FROM trans
            WHERE t_dat <= DATE '{cutoff}'
            GROUP BY article_id
        ),
        recent7 AS (
            SELECT article_id, COUNT(*) AS i_sales_7d
            FROM trans
            WHERE t_dat <= DATE '{cutoff}' AND t_dat > DATE '{cutoff}' - INTERVAL 7 DAY
            GROUP BY article_id
        ),
        recent28 AS (
            SELECT article_id, COUNT(*) AS i_sales_28d
            FROM trans
            WHERE t_dat <= DATE '{cutoff}' AND t_dat > DATE '{cutoff}' - INTERVAL 28 DAY
            GROUP BY article_id
        ),
        yoy AS (
            -- target週(cutoff, cutoff+7]のちょうど1年前にあたる週の売上。季節性シグナル。
            SELECT article_id, COUNT(*) AS i_sales_yoy_7d
            FROM trans
            WHERE t_dat > DATE '{cutoff}' - INTERVAL 365 DAY AND t_dat <= DATE '{cutoff}' - INTERVAL 358 DAY
            GROUP BY article_id
        )
        SELECT b.article_id, b.i_total_sales, b.i_avg_price,
               DATE_DIFF('day', b.i_last_sale_dat, DATE '{cutoff}') AS i_days_since_last_sale,
               COALESCE(r7.i_sales_7d, 0) AS i_sales_7d,
               COALESCE(r28.i_sales_28d, 0) AS i_sales_28d,
               COALESCE(y.i_sales_yoy_7d, 0) AS i_sales_yoy_7d
        FROM base b
        LEFT JOIN recent7 r7 USING (article_id)
        LEFT JOIN recent28 r28 USING (article_id)
        LEFT JOIN yoy y USING (article_id)
    """)

    con.execute(f"""
        CREATE OR REPLACE TABLE cat_feat AS
        WITH yoy AS (
            -- section_name単位。articleそのものは毎年入れ替わるが部門(section)は安定して存在するため、
            -- 個々のarticleがまだ存在しなかった年でも「このカテゴリは今の時期に売れるか」を測れる。
            SELECT a.section_name, COUNT(*) AS cnt
            FROM trans t JOIN articles a ON a.article_id = t.article_id
            WHERE t.t_dat > DATE '{cutoff}' - INTERVAL 365 DAY AND t.t_dat <= DATE '{cutoff}' - INTERVAL 358 DAY
            GROUP BY a.section_name
        ),
        recent AS (
            SELECT a.section_name, COUNT(*) AS cnt
            FROM trans t JOIN articles a ON a.article_id = t.article_id
            WHERE t.t_dat <= DATE '{cutoff}' AND t.t_dat > DATE '{cutoff}' - INTERVAL 28 DAY
            GROUP BY a.section_name
        )
        SELECT COALESCE(y.section_name, r.section_name) AS section_name,
               COALESCE(y.cnt, 0) AS cat_sales_yoy_7d,
               COALESCE(r.cnt, 0) AS cat_sales_recent_28d
        FROM yoy y
        FULL OUTER JOIN recent r USING (section_name)
    """)

    con.execute(f"""
        CREATE OR REPLACE TABLE ui_feat AS
        SELECT customer_idx, article_id,
               COUNT(*) AS ui_times_bought,
               DATE_DIFF('day', MAX(t_dat), DATE '{cutoff}') AS ui_days_since_last_bought
        FROM trans
        WHERE t_dat <= DATE '{cutoff}'
        GROUP BY customer_idx, article_id
    """)

    con.execute(f"""
        CREATE OR REPLACE TABLE ui_product_feat AS
        SELECT customer_idx, product_code,
               COUNT(*) AS ui_times_bought_product_code,
               DATE_DIFF('day', MAX(t_dat), DATE '{cutoff}') AS ui_days_since_last_bought_product_code
        FROM trans
        WHERE t_dat <= DATE '{cutoff}'
        GROUP BY customer_idx, product_code
    """)

    target_week_start_dom = (pd.Timestamp(cutoff) + pd.Timedelta(days=1)).day

    feature_df = con.execute(f"""
        SELECT
            cp.customer_idx, cp.article_id,
            cp.from_repeat, cp.from_trend, cp.from_category_color, cp.from_cf, cp.from_cooccurrence,
            cp.from_content_similarity,
            (cp.from_repeat + cp.from_trend + cp.from_category_color + cp.from_cf + cp.from_cooccurrence
             + cp.from_content_similarity) AS n_sources,

            a.product_code, a.product_group_name, a.index_group_name, a.section_name,

            COALESCE(uf.u_total_purchases, 0) AS u_total_purchases,
            COALESCE(uf.u_distinct_articles, 0) AS u_distinct_articles,
            COALESCE(uf.u_distinct_product_codes, 0) AS u_distinct_product_codes,
            uf.u_avg_price,
            uf.u_price_cv,
            uf.u_days_since_last_purchase,
            uf.u_online_ratio,
            COALESCE(uf.u_active_weeks, 0) AS u_active_weeks,
            uf.u_menswear_share, uf.u_babychildren_share, uf.u_ladieswear_share,
            COALESCE(uf.u_purchases_7d, 0) AS u_purchases_7d,
            COALESCE(uf.u_purchases_28d, 0) AS u_purchases_28d,
            COALESCE(uf.u_purchases_91d, 0) AS u_purchases_91d,

            COALESCE(itf.i_total_sales, 0) AS i_total_sales,
            itf.i_avg_price,
            COALESCE(itf.i_days_since_last_sale, 9999) AS i_days_since_last_sale,
            COALESCE(itf.i_sales_7d, 0) AS i_sales_7d,
            COALESCE(itf.i_sales_28d, 0) AS i_sales_28d,
            COALESCE(itf.i_sales_yoy_7d, 0) AS i_sales_yoy_7d,

            COALESCE(uif.ui_times_bought, 0) AS ui_times_bought,
            uif.ui_days_since_last_bought,
            COALESCE(upf.ui_times_bought_product_code, 0) AS ui_times_bought_product_code,
            upf.ui_days_since_last_bought_product_code,
            (itf.i_avg_price - uf.u_avg_price) AS ui_price_diff,

            COALESCE(cf.cat_sales_yoy_7d, 0) AS cat_sales_yoy_7d,
            COALESCE(cf.cat_sales_recent_28d, 0) AS cat_sales_recent_28d,
            COALESCE(cf.cat_sales_yoy_7d, 0) / NULLIF(COALESCE(cf.cat_sales_recent_28d, 0) / 4.0, 0)
                AS cat_seasonal_lift,

            {target_week_start_dom} AS target_week_start_dom
        FROM candidate_pool cp
        LEFT JOIN articles a ON a.article_id = cp.article_id
        LEFT JOIN user_feat uf ON uf.customer_idx = cp.customer_idx
        LEFT JOIN item_feat itf ON itf.article_id = cp.article_id
        LEFT JOIN ui_feat uif ON uif.customer_idx = cp.customer_idx AND uif.article_id = cp.article_id
        LEFT JOIN ui_product_feat upf
            ON upf.customer_idx = cp.customer_idx AND upf.product_code = a.product_code
        LEFT JOIN cat_feat cf ON cf.section_name = a.section_name
    """).df()

    if with_label:
        label_rows = [(cid, int(aid)) for cid, articles in label_map.items() for aid in articles]
        label_df = pd.DataFrame(label_rows, columns=["customer_idx", "article_id"])
        label_df["label"] = 1
        feature_df = feature_df.merge(label_df, on=["customer_idx", "article_id"], how="left")
        feature_df["label"] = feature_df["label"].fillna(0).astype(int)

    feature_df["cutoff"] = cutoff
    con.unregister("candidate_pool")
    return feature_df
