"""商品テキスト(prod_name/product_type_name/colour_group_name/detail_desc)からTF-IDF+コサイン類似度で
item-to-itemの類似テーブルを事前計算するCLIスクリプト。

articles.csvだけに依存し取引データを見ないため、cutoffに関係なくカタログ全体で1回計算すれば使い回せる
(CF・共起とは違いcutoffごとの再計算が不要)。02_output/features/content_similarity.parquetに保存する。
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "01_input")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    articles = pd.read_csv(
        os.path.join(DATA_DIR, "articles.csv"),
        usecols=["article_id", "prod_name", "product_type_name", "colour_group_name", "detail_desc"],
    )
    articles["detail_desc"] = articles["detail_desc"].fillna("")
    text = (
        articles["prod_name"] + " " + articles["product_type_name"] + " "
        + articles["colour_group_name"] + " " + articles["detail_desc"]
    )

    vectorizer = TfidfVectorizer(max_features=20000, stop_words="english")
    tfidf = vectorizer.fit_transform(text)
    print(f"tfidf shape: {tfidf.shape}")

    nn = NearestNeighbors(n_neighbors=args.top_n + 1, metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(tfidf)
    distances, indices = nn.kneighbors(tfidf)
    print("kneighbors done")

    article_ids = articles["article_id"].to_numpy()
    rows = []
    for i in range(len(article_ids)):
        for rank, (dist, j) in enumerate(zip(distances[i], indices[i])):
            if j == i:
                continue
            rows.append((int(article_ids[i]), int(article_ids[j]), 1.0 - float(dist)))

    result = pd.DataFrame(rows, columns=["article_id", "neighbor_article_id", "similarity"])
    result = result.sort_values(["article_id", "similarity"], ascending=[True, False])
    result = result.groupby("article_id").head(args.top_n).reset_index(drop=True)
    result.to_parquet(args.out, index=False)
    print(f"保存しました: {args.out} shape={result.shape}")


if __name__ == "__main__":
    main()
