"""Side-by-side comparison of two learners run on identical splits (results/{learner}/metrics.csv).

Folds, seeds, cells, buffer and in-fold feature selection are the same for both learners, so the
differences are paired. Writes results/learner_comparison.md.

Usage:
  uv run python scripts/compare_learners.py [--a xgboost] [--b tabicl]
"""
import argparse

import pandas as pd

from hgt.io import results_dir
from hgt.splits import FEATURE_SETS


def pooled(m: pd.DataFrame, exp: str) -> pd.DataFrame:
    d = m[(m["experiment"] == exp) & (m["fold"] == -1)]
    keys = ["train", "test", "feature_set", "repeat", "k", "draw", "variant"]
    return d.pivot_table(index=keys, columns="metric", values="value").reset_index()


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="xgboost")
    ap.add_argument("--b", default="tabicl")
    args = ap.parse_args()
    A, B = args.a, args.b
    m = {name: pd.read_csv(results_dir() / name / "metrics.csv") for name in (A, B)}
    lines = [f"# Learner comparison: {A} vs {B}", "",
             "Identical splits, seeds, cells, buffer and in-fold feature selection; only the final learner "
             f"differs. Δ = {B} − {A} (pooled R², mean over repeats).", ""]

    # E1
    e1 = {n: pooled(d, "E1").groupby(["test", "feature_set"])["r2"].mean() for n, d in m.items()}
    j = pd.concat(e1, axis=1).dropna().reset_index()
    j["Δ"] = j[B] - j[A]
    j["feature_set"] = pd.Categorical(j["feature_set"], FEATURE_SETS, ordered=True)
    lines += ["## E1 in-domain R²", "", md_table(j.sort_values(["test", "feature_set"])), ""]

    # E2
    e2 = {n: pooled(d, "E2").groupby("feature_set")[["r2", "bias"]].mean() for n, d in m.items()}
    j = pd.concat(e2, axis=1).dropna()
    j.columns = [f"{n} {c}" for n, c in j.columns]
    j[f"Δ r2"] = j[f"{B} r2"] - j[f"{A} r2"]
    lines += ["## E2 cross-country (Tanzania 2022 → Colombia 2022)", "", md_table(j.reset_index()), ""]

    # E3 at the largest k, tz+k
    e3 = {}
    for n, d in m.items():
        p = pooled(d, "E3")
        if not p.empty:
            e3[n] = p.groupby(["variant", "k", "feature_set", "draw"])["r2"].mean()
    if len(e3) == 2:
        # paired: only (variant, k, FS, draw) combinations run by both learners (same shot blocks)
        j = pd.concat(e3, axis=1).dropna()
        n_draws = j.reset_index()["draw"].nunique()
        j = j.groupby(level=["variant", "k", "feature_set"]).mean().reset_index()
        j["Δ"] = j[B] - j[A]
        lines += [f"## E3 few-shot R² (mean over the {n_draws} draw(s) run by both learners)", "", md_table(j), ""]

    # E4 temporal loss = baseline (B->B) - cross-year (A->B)
    rows = []
    for n, d in m.items():
        p = pooled(d, "E4")
        for (test, variant, fs), g in p.groupby(["test", "variant", "feature_set"]):
            base = g[g["train"] == test]["r2"].mean()
            for train in set(g["train"]) - {test}:
                cross = g[g["train"] == train]["r2"].mean()
                rows.append({"learner": n, "direction": f"{train} → {test}", "variant": variant,
                             "feature_set": fs, "baseline": base, "cross": cross, "loss": base - cross})
    e4 = pd.DataFrame(rows)
    if not e4.empty:
        w = e4.pivot_table(index=["direction", "variant", "feature_set"], columns="learner",
                           values=["cross", "loss"]).dropna()
        w.columns = [f"{n} {c}" for c, n in w.columns]
        lines += ["## E4 cross-year: cross-year R² and temporal loss", "", md_table(w.reset_index()), ""]

    out = results_dir() / "learner_comparison.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
