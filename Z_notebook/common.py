"""H&M コンペ用の共有ユーティリティ(duckdb接続・バックテスト分割・評価指標・提出/実験ログ)。

Z_notebook/ 配下の各ノートブック・CLIから `from common import ...` で読み込む想定。
候補生成は `candidates.py`、特徴量は `features.py` に置く。

duckdbのメモリ上限・スレッド数・スピル先は環境変数で上書きできる(コードを書き換えずにMac/Colabを切り替えるため):
    HM_DUCKDB_MEMORY  (既定 '12GB'。Colab High-RAM(51GB)なら '20GB' 程度)
    HM_DUCKDB_THREADS (既定 4)
    HM_DUCKDB_TMP     (既定 未設定=duckdbの既定。Colabなら '/content/duckdb_tmp' などローカルディスクを指定)
"""

import os

import duckdb
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "..", "01_input")
FEATURES_DIR = os.path.join(_HERE, "..", "02_output", "features")
TRANS_LITE_PATH = os.path.join(FEATURES_DIR, "trans_lite.parquet")
SUBMISSION_DIR = os.path.join(_HERE, "..", "02_output", "submission")
MODELS_DIR = os.path.join(_HERE, "..", "02_output", "models")
EXPERIMENT_LOG_PATH = os.path.join(_HERE, "..", "experiment_log.csv")


def configure_con(con, memory_limit=None, threads=None, temp_dir=None):
    """duckdb接続にメモリ上限・スレッド数・スピル先を設定する(引数 > 環境変数 > 既定値)。"""
    memory_limit = memory_limit or os.environ.get("HM_DUCKDB_MEMORY", "12GB")
    threads = int(threads or os.environ.get("HM_DUCKDB_THREADS", 4))
    temp_dir = temp_dir or os.environ.get("HM_DUCKDB_TMP")
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    if temp_dir:
        os.makedirs(temp_dir, exist_ok=True)
        con.execute(f"PRAGMA temp_directory='{temp_dir}'")
    return con


def get_con(memory_limit=None, threads=None, temp_dir=None):
    """trans(取引) / cust_map(customer_id<->customer_idx) / articles を使えるduckdb接続を返す。"""
    con = configure_con(duckdb.connect(), memory_limit, threads, temp_dir)
    con.execute(f"CREATE OR REPLACE VIEW trans AS SELECT * FROM read_parquet('{TRANS_LITE_PATH}')")
    con.execute(f"""
        CREATE OR REPLACE TABLE cust_map AS
        SELECT customer_id, (ROW_NUMBER() OVER () - 1)::INTEGER AS customer_idx
        FROM read_csv_auto('{DATA_DIR}/customers.csv')
    """)
    con.execute(f"CREATE OR REPLACE VIEW articles AS SELECT * FROM read_csv_auto('{DATA_DIR}/articles.csv')")
    return con


def ensure_trans_lite(verbose: bool = True) -> str:
    """01_eda.ipynbと同じロジックで trans_lite.parquet を作る(既にあれば何もしない)。

    Colabなど新しい環境で01を開かずにパイプラインを始められるようにするためのもの。
    customer_idxの採番(ROW_NUMBER() OVER ())はget_con()のcust_mapと同一の式であること。
    """
    if os.path.exists(TRANS_LITE_PATH):
        if verbose:
            print(f"[skip] {TRANS_LITE_PATH} は既に存在します")
        return TRANS_LITE_PATH
    os.makedirs(os.path.dirname(TRANS_LITE_PATH), exist_ok=True)
    con = configure_con(duckdb.connect())
    con.execute(f"""
        CREATE TABLE cust_map AS
        SELECT customer_id, (ROW_NUMBER() OVER () - 1)::INTEGER AS customer_idx
        FROM read_csv_auto('{DATA_DIR}/customers.csv')
    """)
    con.execute(f"""
        COPY (
          SELECT cm.customer_idx, t.t_dat,
                 CAST(t.article_id AS BIGINT) AS article_id,
                 t.price, t.sales_channel_id::TINYINT AS sales_channel_id,
                 CAST(a.product_code AS INTEGER) AS product_code,
                 a.product_group_name, a.index_group_name,
                 c.age::SMALLINT AS age
          FROM read_csv_auto('{DATA_DIR}/transactions_train.csv') t
          JOIN cust_map cm ON cm.customer_id = t.customer_id
          LEFT JOIN read_csv_auto('{DATA_DIR}/articles.csv') a
            ON CAST(a.article_id AS BIGINT) = CAST(t.article_id AS BIGINT)
          LEFT JOIN read_csv_auto('{DATA_DIR}/customers.csv') c
            ON c.customer_id = t.customer_id
        ) TO '{TRANS_LITE_PATH}' (FORMAT PARQUET)
    """)
    con.close()
    if verbose:
        print(f"[ok] {TRANS_LITE_PATH} を作成しました")
    return TRANS_LITE_PATH


def backtest_split(con, holdout_days: int = 7):
    """学習データ最終日から holdout_days 日分をホールドアウトし、(cutoff文字列, {customer_idx: {article_id,...}}) を返す。

    時系列データのためランダム分割は使わない(hypothesis_driven_design.md 7節)。
    """
    max_dat = con.execute("SELECT MAX(t_dat) FROM trans").fetchone()[0]
    cutoff = (max_dat - pd.Timedelta(days=holdout_days)).strftime("%Y-%m-%d")
    actual_df = con.execute(f"""
        SELECT DISTINCT customer_idx, article_id
        FROM trans
        WHERE t_dat > DATE '{cutoff}'
    """).df()
    actual_map = actual_df.groupby("customer_idx")["article_id"].apply(set).to_dict()
    return cutoff, actual_map


def label_window(con, cutoff: str, days: int = 7) -> dict:
    """cutoff直後days日間に購入されたarticleを {customer_idx: {article_id,...}} で返す(学習ラベル用)。

    backtest_split はデータ最終日基準の1本のホールドアウトに固定されているのに対し、
    こちらは任意のcutoffに対して同じ形の正解集合を作れる(複数週の学習データ生成用)。
    """
    df = con.execute(f"""
        SELECT DISTINCT customer_idx, article_id
        FROM trans
        WHERE t_dat > DATE '{cutoff}' AND t_dat <= DATE '{cutoff}' + INTERVAL {days} DAY
    """).df()
    return df.groupby("customer_idx")["article_id"].apply(set).to_dict()


def active_customers(con, cutoff: str, trailing_days: int = 56) -> set:
    """cutoff以前 trailing_days 日以内に購入実績のある顧客のcustomer_idx集合。

    候補生成・特徴量計算の対象を絞るために使う(何ヶ月も購入のない顧客まで毎回全件処理すると計算コストが跳ね上がるため)。
    """
    df = con.execute(f"""
        SELECT DISTINCT customer_idx
        FROM trans
        WHERE t_dat <= DATE '{cutoff}' AND t_dat > DATE '{cutoff}' - INTERVAL {trailing_days} DAY
    """).df()
    return set(df.customer_idx.tolist())


def apk(actual: set, predicted: list, k: int = 12) -> float:
    if not actual:
        return 0.0
    predicted = predicted[:k]
    hits = 0
    score = 0.0
    for i, p in enumerate(predicted):
        if p in actual:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(actual), k)


def mapk(actual_map: dict, pred_map: dict, k: int = 12) -> float:
    scores = [apk(actual, pred_map.get(cid, []), k) for cid, actual in actual_map.items()]
    return float(np.mean(scores)) if scores else 0.0


def recall_at_k(actual: set, predicted: list, k: int = 12) -> float:
    """predicted の上位k件に、actual(実購入)のうち何割が含まれるか。"""
    if not actual:
        return np.nan
    hit = len(actual & set(predicted[:k]))
    return hit / len(actual)


def mean_recall_at_k(actual_map: dict, pred_map: dict, k: int = 12) -> float:
    scores = [recall_at_k(actual, pred_map.get(cid, []), k) for cid, actual in actual_map.items()]
    scores = [s for s in scores if not np.isnan(s)]
    return float(np.mean(scores)) if scores else 0.0


def format_prediction(article_ids) -> str:
    return " ".join(str(a).zfill(10) for a in article_ids)


def append_experiment_log(**row) -> pd.DataFrame:
    log_row = pd.DataFrame([row])
    if os.path.exists(EXPERIMENT_LOG_PATH):
        existing = pd.read_csv(EXPERIMENT_LOG_PATH)
        updated = pd.concat([existing, log_row], ignore_index=True)
    else:
        updated = log_row
    updated.to_csv(EXPERIMENT_LOG_PATH, index=False)
    return updated
