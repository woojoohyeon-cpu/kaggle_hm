"""1つのcutoffの候補+特徴量テーブルを作ってparquetに保存するCLIスクリプト。

04_feature_engineering.ipynb / run_trend_experiment.py からサブプロセスとして呼び出す想定
(cutoffごとにOSプロセスを分離することで、複数週分を1つのカーネルに溜め込んでOOM killされる問題を回避する)。

テーブルはduckdbから直接parquetに書き出す(features.write_feature_table)。Python側に巨大なDataFrameを
作らないため、以前の実装より1プロセスあたりのピークメモリが大幅に小さい。

例:
    python build_one_cutoff.py --name train1 --cutoff 2020-09-08 --out ../02_output/features/x.parquet
    python build_one_cutoff.py --name val --cutoff 2020-09-15 --out x.parquet --trend --extra-sources age_trend
"""

import argparse
import os
import time

# implicit(ALS)はOpenBLASのスレッドと自前スレッドが競合すると極端に遅くなるため1に固定する
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from common import get_con  # noqa: E402
from features import write_feature_table  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-label", action="store_true")
    parser.add_argument("--extra-sources", default="", help="カンマ区切り。例: age_trend")
    parser.add_argument("--trend", action="store_true",
                        help="短期トレンド特徴量(features.TREND_FEATURES)も計算する。既定はOFF(04/05の従来構成と同じ列)")
    parser.add_argument("--include-yoy", action="store_true", help="0917版の季節性(YoY)特徴量も計算する")
    parser.add_argument("--trailing-days", type=int, default=56)
    args = parser.parse_args()

    extra_sources = tuple(s for s in args.extra_sources.split(",") if s)
    t0 = time.time()
    con = get_con()
    tmp_out = args.out + ".tmp"
    n_rows = write_feature_table(
        con, args.cutoff, tmp_out,
        with_label=not args.no_label,
        trailing_days=args.trailing_days,
        extra_sources=extra_sources,
        include_trend=args.trend,
        include_yoy=args.include_yoy,
    )
    positive = ""
    if not args.no_label:
        rate = con.execute(f"SELECT AVG(label) FROM read_parquet('{tmp_out}')").fetchone()[0]
        positive = f" positive_rate={rate:.6f}"
    con.close()
    # 書き込み途中で落ちたファイルを「完成品」と誤認しないよう、完了後にrenameする
    os.replace(tmp_out, args.out)
    print(f"[{args.name}] cutoff={args.cutoff} rows={n_rows:,}{positive} ({(time.time() - t0) / 60:.1f}分)")
    print(f"[{args.name}] 保存しました: {args.out}")


if __name__ == "__main__":
    main()
