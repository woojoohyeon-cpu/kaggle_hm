"""短期トレンド仮説の検証パイプライン(CLI)。Mac(ローカル)でもGoogle Colabでも同じコマンドで動く。

仮説: ファストファッションでは季節性(前年同週)より、直近数日〜数週の流行の勢い・新しさ・同年代での流行が
購買を決める(季節横断holdoutで0917版のYoY特徴量に効果がなかったこと、上位解法の多くが直近1〜6週の
情報だけで組まれていたことが根拠。00_md/pipeline_build_log.md「短期トレンド仮説での再設計」節参照)。

手順(各ステップは途中で落ちても、再実行すれば終わったところから再開する):

    python run_trend_experiment.py build   [--only val,train1,...] # 週ごとの特徴量テーブル(サブプロセス)
    python run_trend_experiment.py train   [--variants ...]        # 3条件を同じプロトコルで学習・評価
    python run_trend_experiment.py report                          # 週別MAP@12・ブートストラップ信頼区間
    python run_trend_experiment.py refit   --variant trend_full    # 提出用に直近4週で学習し直す
    python run_trend_experiment.py predict --variant trend_full    # 提出ファイル作成

共通オプション:
    --work-dir   作業ディレクトリ(既定: 02_output/experiments/trend。Colabなら /content/hm_work 推奨)
    --sync-dir   小さい成果物(モデル・結果・提出ファイル・ログ)をコピーする先(ColabならGoogle Drive)
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common import (  # noqa: E402
    MODELS_DIR, SUBMISSION_DIR, append_experiment_log, format_prediction, get_con, label_window,
)
from features import TREND_FEATURES  # noqa: E402
import train_protocol as tp  # noqa: E402

DEFAULT_WORK_DIR = os.path.join(HERE, "..", "02_output", "experiments", "trend")
TRAIN_WEEKS = ["train2", "train3", "train4"]
STOP_WEEK = "train1"
REPORT_WEEKS_FIXED = ["val"]
# 季節の異なる追加の報告週(任意)。前回の季節横断検証(Claude Project「パイプライン最新ステータス」11節)と同じ日付から、
# 実テストと同季節の1年前(秋)・年末商戦(冬)・COVID後の回復期(夏)を選ぶ。短期トレンド特徴量は前年同週を参照しないので
# 2019-09-13より前という制約はない。学習週(2020年8〜9月)より前なのでラベルの重なりもない。
SEASON_WEEKS = {"season1": "2019-09-21", "season2": "2019-12-21", "season3": "2020-06-20"}
BUILD_FLAGS = ["--trend", "--extra-sources", "age_trend"]


# ---------------------------------------------------------------------------
# 共通
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def md_table(df: pd.DataFrame, index: bool = True) -> str:
    """tabulateに依存しない簡易Markdown表(Mac/Colabどちらでも追加インストール不要にするため)。"""
    d = df.reset_index() if index else df
    def fmt(v):
        if isinstance(v, float):
            return "" if v != v else f"{v:.6g}"
        return str(v)
    header = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join([header, sep] + body)


def paths(work_dir: str) -> dict:
    p = {
        "root": work_dir,
        "features": os.path.join(work_dir, "features"),
        "models": os.path.join(work_dir, "models"),
        "results": os.path.join(work_dir, "results"),
        "submission": os.path.join(work_dir, "submission"),
        "manifest": os.path.join(work_dir, "manifest.json"),
    }
    for k, v in p.items():
        if k != "manifest":
            os.makedirs(v, exist_ok=True)
    return p


def load_manifest(p: dict, holdout_weeks: int = 3) -> dict:
    """週の構成(cutoff)を決めて保存する。一度決めたら再実行しても変えない。"""
    if os.path.exists(p["manifest"]):
        with open(p["manifest"]) as f:
            return json.load(f)
    con = get_con()
    max_dat = con.execute("SELECT MAX(t_dat) FROM trans").fetchone()[0]
    con.close()

    def day(n):
        return (pd.Timestamp(max_dat) - pd.Timedelta(days=n)).strftime("%Y-%m-%d")

    weeks = {"val": day(7), "train1": day(14), "train2": day(21), "train3": day(28), "train4": day(35)}
    # multi_week_eval.holdout_cutoffs と同じ週(最終日-42/-49/-56日)。どの学習週のラベル期間とも重ならない
    for i in range(holdout_weeks):
        weeks[f"holdout{i + 1}"] = day(42 + 7 * i)
    weeks.update(SEASON_WEEKS)
    manifest = {"max_dat": pd.Timestamp(max_dat).strftime("%Y-%m-%d"), "final": pd.Timestamp(max_dat).strftime("%Y-%m-%d"),
                "weeks": weeks, "created": datetime.now().isoformat(timespec="seconds")}
    with open(p["manifest"], "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def week_path(p: dict, name: str, cutoff: str) -> str:
    return os.path.join(p["features"], f"{name}_{cutoff}.parquet")


def report_weeks(manifest: dict) -> list:
    extra = sorted(w for w in manifest["weeks"] if w.startswith("holdout"))
    extra += sorted(w for w in manifest["weeks"] if w.startswith("season"))
    return REPORT_WEEKS_FIXED + extra


def sync(p: dict, sync_dir) -> None:
    """モデル・結果・提出ファイル・manifestだけを同期する(特徴量parquetは大きいので同期しない)。"""
    if not sync_dir:
        return
    for sub in ["models", "results", "submission"]:
        src = p[sub]
        dst = os.path.join(sync_dir, sub)
        os.makedirs(dst, exist_ok=True)
        for f in glob.glob(os.path.join(src, "*")):
            target = os.path.join(dst, os.path.basename(f))
            if not os.path.exists(target) or os.path.getmtime(f) > os.path.getmtime(target):
                shutil.copy2(f, target)
    shutil.copy2(p["manifest"], os.path.join(sync_dir, "manifest.json"))
    log(f"同期しました -> {sync_dir}")


def run_build_subprocess(name: str, cutoff: str, out: str, with_label: bool = True) -> None:
    cmd = [sys.executable, os.path.join(HERE, "build_one_cutoff.py"), "--name", name, "--cutoff", cutoff,
           "--out", out] + BUILD_FLAGS + ([] if with_label else ["--no-label"])
    t0 = time.time()
    result = subprocess.run(cmd, cwd=HERE)
    if result.returncode != 0:
        hint = ""
        if result.returncode in (-9, 137):
            hint = (" → OSにSIGKILLされました(メモリ不足の可能性が高い)。HM_DUCKDB_MEMORYを下げるか、"
                    "High-RAMランタイムを使ってください")
        raise RuntimeError(f"{name}(cutoff={cutoff})の構築に失敗しました returncode={result.returncode}{hint}")
    log(f"{name} 完了 ({(time.time() - t0) / 60:.1f}分)")


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def cmd_build(args) -> None:
    p = paths(args.work_dir)
    manifest = load_manifest(p, args.holdout_weeks)
    # 既定ではseason週(任意の追加評価)は作らない。作る場合は --only season1,season2,season3
    names = args.only.split(",") if args.only else [w for w in manifest["weeks"] if not w.startswith("season")]
    for name in names:
        cutoff = manifest["weeks"][name]
        out = week_path(p, name, cutoff)
        if os.path.exists(out):
            log(f"[skip] {name} ({cutoff}) は作成済み")
            continue
        log(f"=== {name} (cutoff={cutoff}) を構築 ===")
        run_build_subprocess(name, cutoff, out)
    sync(p, args.sync_dir)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def week_context(p: dict, name: str, cutoff: str, con) -> dict:
    """評価週ごとの正解・アクティブ顧客・年齢層・フォールバック用リスト(提出時と同じ作り方)をキャッシュする。"""
    cache = os.path.join(p["features"], f"context_{name}.pkl")
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    from candidates import ensure_cust_age
    from common import active_customers
    actual = label_window(con, cutoff, 7)
    active = active_customers(con, cutoff, 56)
    ensure_cust_age(con)
    con.register("_ctx_customers", pd.DataFrame({"customer_idx": np.array(list(actual.keys()), dtype=np.int32)}))
    bdf = con.execute("SELECT customer_idx, age_bucket FROM cust_age JOIN _ctx_customers USING (customer_idx)").df()
    con.unregister("_ctx_customers")
    age_top, global_top = tp.fallback_lists(con, cutoff)
    ctx = {"actual": actual, "active": {int(c) for c in active},
           "buckets": {int(c): (int(b) if b == b else -1) for c, b in zip(bdf.customer_idx, bdf.age_bucket)},
           "age_top": age_top, "global_top": global_top}
    pd.to_pickle(ctx, cache)
    return ctx


COLD_START_MODES = {"global": "", "age_trend": "+agecold"}


def clear_cached_scores(p: dict, label: str) -> None:
    """モデルを作り直したときに、古いモデルで計算した顧客別APのキャッシュを消す(レポートに新旧が混ざらないように)。"""
    for suffix in COLD_START_MODES.values():
        for f in glob.glob(os.path.join(p["results"], f"ap_{label}{suffix}_*.npy")):
            os.remove(f)


def evaluate_payload(p: dict, manifest: dict, label: str, payload: dict, vocab: dict, con) -> list:
    """報告週ごとに、提出ファイルと同じ作り方の予測でMAP@12を出す(フォールバック2通り)。

    label        : 従来どおり全体人気でフォールバック(06と同じ)。モデル同士の比較はこちらで行う
    label+agecold: コールドスタートを同年齢層の直近人気にした場合(コールドスタート部分の効果を見る)
    """
    rows = []
    for name in report_weeks(manifest):
        cutoff = manifest["weeks"][name]
        path = week_path(p, name, cutoff)
        if not os.path.exists(path):
            log(f"  [{label}] {name}: 特徴量テーブルが無いのでスキップ")
            continue
        ap_paths = {mode: os.path.join(p["results"], f"ap_{label}{suffix}_{name}.npy")
                    for mode, suffix in COLD_START_MODES.items()}
        if not all(os.path.exists(x) for x in ap_paths.values()):
            ctx = week_context(p, name, cutoff, con)
            model_map = tp.top_k_map(tp.score_week(path, payload, vocab))
            for mode, ap_path in ap_paths.items():
                customers, ap = tp.submission_like_ap(ctx["actual"], model_map, ctx["active"], ctx["buckets"],
                                                      ctx["age_top"], ctx["global_top"], cold_start=mode)
                np.save(ap_path, ap)
            cust_path = os.path.join(p["results"], f"customers_{name}.npy")
            if not os.path.exists(cust_path):
                np.save(cust_path, customers)
        for mode, suffix in COLD_START_MODES.items():
            ap = np.load(ap_paths[mode])
            rows.append({"variant": f"{label}{suffix}", "week": name, "cutoff": cutoff,
                         "map12": float(ap.mean()), "n_customers": len(ap)})
        log(f"  [{label}] {name} ({cutoff}) MAP@12={rows[-2]['map12']:.5f} "
            f"(コールドスタートを年齢層人気にすると {rows[-1]['map12']:.5f}, n={rows[-1]['n_customers']:,})")
    return rows


def _write_scores(p: dict, rows: list) -> None:
    path = os.path.join(p["results"], "scores.csv")
    new = pd.DataFrame(rows)
    if os.path.exists(path):
        old = pd.read_csv(path)
        old = old[~old.set_index(["variant", "week"]).index.isin(new.set_index(["variant", "week"]).index)]
        new = pd.concat([old, new], ignore_index=True)
    new.sort_values(["variant", "week"]).to_csv(path, index=False)


def cmd_train(args) -> None:
    p = paths(args.work_dir)
    manifest = load_manifest(p)
    weeks = manifest["weeks"]
    for name in TRAIN_WEEKS + [STOP_WEEK, "val"]:
        if not os.path.exists(week_path(p, name, weeks[name])):
            raise FileNotFoundError(f"{name}の特徴量テーブルがありません。先に build を実行してください")
    vocab = tp.category_vocab()
    con = get_con()

    for variant in args.variants.split(","):
        spec = tp.VARIANTS[variant]
        model_path = os.path.join(p["models"], f"{variant}.pkl")
        if os.path.exists(model_path):
            log(f"[skip] {variant} は学習済み({model_path})。評価だけ行います")
            payload = tp.load_payload(model_path)
        else:
            log(f"=== {variant}: 学習 (pool={spec['pool']}, 特徴量{len(spec['features'])}列) ===")
            clear_cached_scores(p, variant)
            t0 = time.time()
            train_df = tp.load_training_frame(
                [week_path(p, w, weeks[w]) for w in TRAIN_WEEKS], spec["features"], spec["pool"], vocab,
                neg_frac=args.neg_frac, seed=42,
            )
            stop_df = tp.load_training_frame(
                [week_path(p, STOP_WEEK, weeks[STOP_WEEK])], spec["features"], spec["pool"], vocab,
                neg_frac=0.0, seed=0, positive_only=True,
            )
            log(f"  学習行数 {len(train_df):,} / early stopping用 {len(stop_df):,}")
            model, best_iter = tp.fit_ranker(train_df, spec["features"], eval_df=stop_df)
            del train_df, stop_df
            payload = dict(model=model, feature_cols=spec["features"], cat_cols=[c for c in tp.CAT_COLS],
                           categories=vocab, pool=spec["pool"], variant=variant, best_iter=best_iter,
                           trained_on=[weeks[w] for w in TRAIN_WEEKS], early_stop_week=weeks[STOP_WEEK],
                           protocol="train2-4 / early stop train1 / report val+holdout")
            tp.save_payload(model_path, **payload)
            tp.feature_importance(model, spec["features"]).to_csv(
                os.path.join(p["results"], f"importance_{variant}.csv"), index=False)
            log(f"  学習完了 best_iter={best_iter} ({(time.time() - t0) / 60:.1f}分)")

        rows = evaluate_payload(p, manifest, variant, payload, vocab, con)
        _write_scores(p, rows)
        val_row = [r for r in rows if r["week"] == "val"]
        if val_row and not args.no_log:
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            append_experiment_log(
                run_id=f"run_{ts}_trend_{variant}", timestamp=ts, notebook="run_trend_experiment.py",
                description=f"短期トレンド仮説 variant={variant} (pool={spec['pool']})",
                features=", ".join(spec["features"]),
                params=f"{tp.LGBM_PARAMS}, best_iter={payload['best_iter']}, neg_frac={args.neg_frac}",
                cv_score=round(val_row[0]["map12"], 5), public_lb="", submission_file="",
                model_file=os.path.basename(model_path),
                notes="train2-4で学習/train1でearly stopping/valは報告専用(05の楽観バイアスなし)",
            )
        sync(p, args.sync_dir)
    con.close()
    cmd_report(args)


def cmd_score_ref(args) -> None:
    """既存モデル(例: 0906版pickle + train_categories)を同じ週で採点し、report に並べる。"""
    p = paths(args.work_dir)
    manifest = load_manifest(p)
    vocab = tp.category_vocab()
    raw = tp.load_payload(args.model)
    with open(args.categories, "rb") as f:
        train_categories = __import__("pickle").load(f)
    # 旧モデルは学習時のカテゴリ語彙(コード対応)で学習しているので、その語彙で符号化する
    ref_vocab = dict(vocab)
    for c, cats in train_categories.items():
        ref_vocab[c] = list(cats)
    payload = dict(model=raw["model"], feature_cols=raw["feature_cols"], pool="base", best_iter=None)
    clear_cached_scores(p, args.label)
    con = get_con()
    rows = evaluate_payload(p, manifest, args.label, payload, ref_vocab, con)
    _write_scores(p, rows)
    con.close()
    cmd_report(args)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def cmd_report(args) -> None:
    p = paths(args.work_dir)
    manifest = load_manifest(p)
    scores_path = os.path.join(p["results"], "scores.csv")
    if not os.path.exists(scores_path):
        log("まだ結果がありません")
        return
    scores = pd.read_csv(scores_path)
    weeks = [w for w in report_weeks(manifest) if w in set(scores["week"])]
    pivot = scores.pivot(index="week", columns="variant", values="map12").reindex(weeks)
    lines = ["# 短期トレンド仮説 検証レポート", "",
             f"作成: {datetime.now().isoformat(timespec='minutes')} / データ最終日: {manifest['max_dat']}", "",
             "プロトコル: train2〜4で学習、train1でearly stopping、val・holdout・seasonは報告専用(学習にも停止判定にも不使用)。",
             "MAP@12は提出ファイルと同じ作り方で計算: 直近56日アクティブな顧客はモデル上位12件、それ以外はフォールバック。",
             "列名の `+agecold` はフォールバックを「同年齢層の直近人気」にした場合(無印は従来どおり全体人気)。",
             "", "## 週別 MAP@12", "", md_table(pivot.round(5)), ""]
    lines += ["平均: " + ", ".join(f"{c}={pivot[c].mean():.5f}" for c in pivot.columns), ""]

    comparisons = []
    if "base" in pivot.columns:
        for variant in [c for c in pivot.columns if c != "base"]:
            diffs = []
            for w in weeks:
                a_path = os.path.join(p["results"], f"ap_base_{w}.npy")
                b_path = os.path.join(p["results"], f"ap_{variant}_{w}.npy")
                if os.path.exists(a_path) and os.path.exists(b_path):
                    d = np.load(b_path) - np.load(a_path)
                    stat = tp.paired_bootstrap(d, seed=1)
                    comparisons.append({"variant": variant, "week": w, **stat})
                    diffs.append(d)
            if diffs:
                pooled = tp.paired_bootstrap(np.concatenate(diffs), seed=2)
                n_win = sum(1 for w in weeks if pivot.loc[w, variant] > pivot.loc[w, "base"])
                comparisons.append({"variant": variant, "week": "ALL(pooled)", **pooled,
                                    "weeks_better": f"{n_win}/{len(weeks)}"})
    if comparisons:
        comp = pd.DataFrame(comparisons)
        comp.to_csv(os.path.join(p["results"], "bootstrap.csv"), index=False)
        lines += ["## base(0906と同じ列・プール)との差: 顧客単位ペアブートストラップ(2000回)", "",
                  md_table(comp.round(6), index=False), "",
                  "判定の目安: pooledの95%CI下限 > 0 かつ 4分の3以上の週で改善 → 採用候補 / CIが0をまたぐ → ノイズと区別できない / "
                  "CI上限 < 0 → 悪化。",
                  "注意: pooledのCIは顧客だけをリサンプリングしており週ごとの揺れを含まないので狭めに出る。"
                  "週ごとの勝敗(weeks_better)と各週のCIも必ず合わせて見ること。", ""]
        for variant in comp["variant"].unique():
            row = comp[(comp.variant == variant) & (comp.week == "ALL(pooled)")]
            if row.empty:
                continue
            r = row.iloc[0]
            n_win, n_weeks = (int(x) for x in str(r.get("weeks_better", "0/1")).split("/"))
            if r.ci_low > 0 and n_win >= int(np.ceil(0.75 * n_weeks)):
                verdict = "採用候補(差がノイズの範囲を超えて改善し、週ごとにも一貫)"
            elif r.ci_low > 0:
                verdict = "判断保留(平均では改善だが週ごとに一貫しない)"
            elif r.ci_high < 0:
                verdict = "悪化"
            else:
                verdict = "判断保留(差がノイズの範囲内)"
            lines.append(f"- **{variant}**: {verdict} (平均差 {r.mean_diff:+.5f}, 95%CI [{r.ci_low:+.5f}, {r.ci_high:+.5f}], "
                         f"改善した週 {r.get('weeks_better', '-')})")
        lines.append("")

    for variant in ["trend_feat", "trend_full"]:
        imp_path = os.path.join(p["results"], f"importance_{variant}.csv")
        if os.path.exists(imp_path):
            imp = pd.read_csv(imp_path)
            imp["rank"] = np.arange(1, len(imp) + 1)
            imp["is_trend"] = imp["feature"].isin(TREND_FEATURES + ["from_age_trend"])
            share = imp.loc[imp.is_trend, "gain"].sum() / max(imp["gain"].sum(), 1e-12)
            lines += [f"## 特徴量重要度(gain) {variant}: トレンド系の合計シェア {share:.1%}", "",
                      md_table(imp.head(20)[["rank", "feature", "gain", "is_trend"]].round(1), index=False), ""]

    report = "\n".join(lines)
    with open(os.path.join(p["results"], "report.md"), "w") as f:
        f.write(report)
    print(report)
    sync(p, getattr(args, "sync_dir", None))


# ---------------------------------------------------------------------------
# refit / predict
# ---------------------------------------------------------------------------

def cmd_refit(args) -> None:
    """提出用: 評価で決めた木の本数(best_iter)を固定し、最新の4週で学習し直す(early stoppingなし)。

    短期トレンド仮説のもとでは直近の週ほど情報価値が高いので、評価時は報告専用だったval週も学習に入れる。
    """
    p = paths(args.work_dir)
    manifest = load_manifest(p)
    weeks = manifest["weeks"]
    base_payload = tp.load_payload(os.path.join(p["models"], f"{args.variant}.pkl"))
    spec = tp.VARIANTS[args.variant]
    refit_weeks = args.weeks.split(",")
    vocab = tp.category_vocab()
    log(f"=== refit {args.variant}: weeks={refit_weeks}, n_estimators={base_payload['best_iter']} ===")
    train_df = tp.load_training_frame([week_path(p, w, weeks[w]) for w in refit_weeks], spec["features"],
                                      spec["pool"], vocab, neg_frac=args.neg_frac, seed=7)
    log(f"  学習行数 {len(train_df):,}")
    model, _ = tp.fit_ranker(train_df, spec["features"], n_estimators=base_payload["best_iter"])
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out = os.path.join(p["models"], f"{args.variant}_refit.pkl")
    payload = dict(model=model, feature_cols=spec["features"], cat_cols=list(tp.CAT_COLS), categories=vocab,
                   pool=spec["pool"], variant=args.variant, best_iter=base_payload["best_iter"],
                   trained_on=[weeks[w] for w in refit_weeks], protocol="refit for submission")
    tp.save_payload(out, **payload)
    os.makedirs(MODELS_DIR, exist_ok=True)
    shutil.copy2(out, os.path.join(MODELS_DIR, f"model_{ts}_trend_{args.variant}_ranker.pkl"))
    log(f"保存しました: {out}")
    sync(p, args.sync_dir)


def cmd_predict(args) -> None:
    p = paths(args.work_dir)
    manifest = load_manifest(p)
    final_cutoff = manifest["final"]
    model_path = args.model or os.path.join(p["models"], f"{args.variant}_refit.pkl")
    payload = tp.load_payload(model_path)
    vocab = payload.get("categories") or tp.category_vocab()

    final_path = week_path(p, "final", final_cutoff)
    if not os.path.exists(final_path):
        log(f"=== 提出用テーブル(cutoff={final_cutoff}, ラベルなし)を構築 ===")
        run_build_subprocess("final", final_cutoff, final_path, with_label=False)

    log("スコアリング中...")
    pred_map = tp.top_k_map(tp.score_week(final_path, payload, vocab))
    log(f"Ranker予測を持つ顧客: {len(pred_map):,}")

    from candidates import age_trend_lists, build_popularity_candidates, ensure_cust_age
    con = get_con()
    ensure_cust_age(con)
    global_top = build_popularity_candidates(con, final_cutoff, top_n=12)
    age_top = age_trend_lists(con, final_cutoff, k=12) if args.cold_start == "age_trend" else {}
    customers = con.execute("""
        SELECT cm.customer_id, cm.customer_idx, ca.age_bucket
        FROM cust_map cm LEFT JOIN cust_age ca USING (customer_idx)
        ORDER BY cm.customer_idx
    """).df()
    con.close()

    buckets = customers["age_bucket"].fillna(-1).astype(int).to_numpy()
    preds = [tp.fill_to_k(pred_map.get(int(c), []), age_top.get(int(b), []), global_top)
             for c, b in zip(customers["customer_idx"].to_numpy(), buckets)]
    customers["prediction"] = [format_prediction(x) for x in preds]
    n_cold = sum(1 for c in customers["customer_idx"] if int(c) not in pred_map)
    log(f"コールドスタート顧客: {n_cold:,}人 → {args.cold_start} でフォールバック")

    assert customers["prediction"].str.split().str.len().eq(12).all(), "12件に満たない顧客がいます"
    sample_path = os.path.join(HERE, "..", "01_input", "sample_submission.csv")
    if os.path.exists(sample_path):
        sample_ids = pd.read_csv(sample_path, usecols=["customer_id"])["customer_id"]
        assert set(sample_ids) == set(customers["customer_id"]), "sample_submission.csvの顧客と一致しません"
        log("フォーマット検証OK(sample_submission.csvと顧客集合が一致)")
    else:
        log("注意: sample_submission.csv が無いので顧客集合の検証をスキップしました")

    scores_path = os.path.join(p["results"], "scores.csv")
    cv = np.nan
    if os.path.exists(scores_path):
        s = pd.read_csv(scores_path)
        label = payload["variant"] + COLD_START_MODES[args.cold_start]
        hit = s[(s.variant == label) & (s.week == "val")]
        cv = float(hit["map12"].iloc[0]) if len(hit) else np.nan
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    fname = f"submission_{ts}_cv{cv:.4f}.csv"
    out = os.path.join(p["submission"], fname)
    customers[["customer_id", "prediction"]].to_csv(out, index=False)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    shutil.copy2(out, os.path.join(SUBMISSION_DIR, fname))
    log(f"保存しました: {out}")
    if not args.no_log:
        append_experiment_log(
            run_id=f"run_{ts}_trend_{payload['variant']}_submit", timestamp=ts, notebook="run_trend_experiment.py",
            description=f"短期トレンド仮説 {payload['variant']} refit + cold_start={args.cold_start}",
            features=", ".join(payload["feature_cols"]), params=f"best_iter={payload['best_iter']}",
            cv_score=round(cv, 5) if cv == cv else "", public_lb="", submission_file=fname,
            model_file=os.path.basename(model_path), notes=f"trained_on={payload.get('trained_on')}",
        )
    print(f"\nKaggleへの提出:\n  kaggle competitions submit -c h-and-m-personalized-fashion-recommendations "
          f"-f {out} -m \"trend {payload['variant']} cold_start={args.cold_start}\"")
    sync(p, args.sync_dir)


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", default=os.environ.get("HM_WORK_DIR", DEFAULT_WORK_DIR))
    parser.add_argument("--sync-dir", default=os.environ.get("HM_SYNC_DIR"))
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--holdout-weeks", type=int, default=3, help="初回のみ有効(manifest作成時に週構成を固定)")
    b.add_argument("--only", default="", help="カンマ区切りの週名だけ作る(例: val,train1)")
    b.set_defaults(func=cmd_build)

    t = sub.add_parser("train")
    t.add_argument("--variants", default="base,trend_feat,trend_full")
    t.add_argument("--neg-frac", type=float, default=0.15)
    t.add_argument("--no-log", action="store_true", help="experiment_log.csvに書かない")
    t.set_defaults(func=cmd_train)

    r = sub.add_parser("report")
    r.set_defaults(func=cmd_report)

    s = sub.add_parser("score-ref")
    s.add_argument("--model", required=True)
    s.add_argument("--categories", required=True, help="local_extract_categories.pyで作ったtrain_categories_*.pkl")
    s.add_argument("--label", default="ref_0906")
    s.set_defaults(func=cmd_score_ref)

    f = sub.add_parser("refit")
    f.add_argument("--variant", required=True, choices=list(tp.VARIANTS))
    f.add_argument("--weeks", default="val,train1,train2,train3")
    f.add_argument("--neg-frac", type=float, default=0.15)
    f.set_defaults(func=cmd_refit)

    pr = sub.add_parser("predict")
    pr.add_argument("--variant", default="trend_full", choices=list(tp.VARIANTS))
    pr.add_argument("--model", default=None, help="省略時は <work-dir>/models/<variant>_refit.pkl")
    pr.add_argument("--cold-start", default="age_trend", choices=["age_trend", "global"])
    pr.add_argument("--no-log", action="store_true")
    pr.set_defaults(func=cmd_predict)

    args = parser.parse_args()
    args.work_dir = os.path.abspath(args.work_dir)
    args.func(args)


if __name__ == "__main__":
    main()
