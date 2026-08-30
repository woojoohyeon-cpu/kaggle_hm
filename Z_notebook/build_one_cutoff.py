"""1つのcutoffの候補+特徴量テーブルを作ってparquetに保存するCLIスクリプト。

04_feature_engineering.ipynbからサブプロセスとして呼び出す想定(cutoffごとにOSプロセスを分離することで、
5週分を1つのカーネルに溜め込んでOOM killされる問題を回避する。jetsamログでPythonプロセスの
OOM killを確認済み)。
"""

import argparse

from common import get_con
from features import build_feature_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-label", action="store_true")
    args = parser.parse_args()

    con = get_con()
    feature_df = build_feature_table(con, args.cutoff, with_label=not args.no_label)
    feature_df.to_parquet(args.out, index=False)

    print(f"[{args.name}] cutoff={args.cutoff} shape={feature_df.shape}", end="")
    if not args.no_label:
        print(f" positive_rate={feature_df['label'].mean():.6f}")
    else:
        print()
    print(f"[{args.name}] 保存しました: {args.out}")


if __name__ == "__main__":
    main()
