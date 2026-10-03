"""Stage 3 report: figures (results/figures/) and results/summary.md from results/metrics.csv.

Tables give mean ± sd over repeats (E1, E2, E4) or over draws (E3, repeat 0 only).

Usage:
  uv run python -m hgt.report
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hgt.config import load_config
from hgt.io import experiment_dir, table_path
from hgt.splits import FEATURE_SETS

FS_COLORS = {"HC": "#2a78d6", "AEF": "#eb6834", "HC+AEF": "#1baf7a", "TES": "#8e44c9", "HC+TES": "#c99a06"}
FS_PRESENT = list(FEATURE_SETS)   # feature sets with results; set in main()
PAIRS = [("AEF", "HC"), ("TES", "HC"), ("TES", "AEF"), ("HC+AEF", "AEF"), ("HC+TES", "TES")]   # (a, b): a − b


def paired_diffs(r2: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Paired R² differences (a − b, same partition per repeat) for the pairs whose sets both ran."""
    return {f"{a} − {b}": ((r2[a] - r2[b]).mean(), (r2[a] - r2[b]).std())
            for a, b in PAIRS if a in r2.columns and b in r2.columns}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CO22 = "colombia_2022"
TZ22 = "tanzania_2022"
ATTRIBUTION = "The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind."

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
})


def fig_dir():
    d = experiment_dir() / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def pm(mean, sd, nd=3):
    if pd.isna(sd):                      # single repeat / draw: no spread to report
        return f"{mean:.{nd}f}"
    return f"{mean:.{nd}f} ± {sd:.{nd}f}"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def pretty(sy: str) -> str:
    site, year = sy.split("_")
    return f"{site.capitalize()} {year}"


def pooled(m: pd.DataFrame, exp: str) -> pd.DataFrame:
    """Pooled (fold = -1) rows of one experiment, metrics as columns."""
    d = m[(m["experiment"] == exp) & (m["fold"] == -1)]
    keys = ["train", "test", "feature_set", "repeat", "k", "draw", "variant"]
    return d.pivot_table(index=keys, columns="metric", values="value").reset_index()


# --- E1 ------------------------------------------------------------------------
def e1_section(m, lines):
    p = pooled(m, "E1")
    site_years = list(dict.fromkeys(p["test"]))
    agg = p.groupby(["test", "feature_set"]).agg(
        r2=("r2", "mean"), r2_sd=("r2", "std"), rmse=("rmse", "mean"), rmse_sd=("rmse", "std"),
        rrmse=("rrmse", "mean"), rrmse_sd=("rrmse", "std"), bias=("bias", "mean"),
        bias_sd=("bias", "std"), n=("n", "first"), reps=("repeat", "nunique")).reset_index()

    # figure: grouped bars
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(len(site_years))
    w = 0.8 / len(FS_PRESENT)
    for ax, (col, label) in zip(axes, [("r2", "Pooled OOF R²"), ("rmse", "Pooled OOF RMSE (m)")]):
        for i, fs in enumerate(FS_PRESENT):
            a = agg[agg["feature_set"] == fs].set_index("test").loc[site_years]
            ax.bar(x + (i - (len(FS_PRESENT) - 1) / 2) * w, a[col], w - 0.02, yerr=a[col + "_sd"], color=FS_COLORS[fs],
                   label=fs, capsize=2, error_kw={"elinewidth": 1, "ecolor": INK2})
        ax.set_xticks(x, [pretty(s) for s in site_years])
        ax.set_ylim(top=ax.get_ylim()[1] * 1.15)      # headroom for the legend
        ax.set_ylabel(label)
        ax.grid(axis="x", visible=False)
    axes[0].legend(ncol=len(FS_PRESENT), loc="upper left", fontsize=8)
    fig.suptitle("E1 in-domain blocked CV (mean ± sd over repeats)")
    fig.tight_layout()
    fig.savefig(fig_dir() / "e1_r2_rmse.png", dpi=150)
    plt.close(fig)

    # table
    tab = []
    for sy in site_years:
        for fs in FS_PRESENT:
            a = agg[(agg["test"] == sy) & (agg["feature_set"] == fs)].iloc[0]
            tab.append({"site-year": pretty(sy), "n cells": int(a["n"]), "FS": fs,
                        "R²": pm(a["r2"], a["r2_sd"]), "RMSE (m)": pm(a["rmse"], a["rmse_sd"], 2),
                        "rRMSE": pm(a["rrmse"], a["rrmse_sd"]), "bias (m)": pm(a["bias"], a["bias_sd"], 2)})
    reps = int(agg["reps"].max())
    lines += ["## E1: in-domain blocked CV", "",
              f"Pooled out-of-fold metrics per repeat, mean ± sd over {reps} repeats "
              "(5 folds of 1 km blocks, folds balanced by cell count).", "", md_table(pd.DataFrame(tab)), ""]

    # paired differences (same folds and seed within a repeat)
    lines += ["Paired difference in pooled R² (same partition within each repeat), mean ± sd over repeats:", ""]
    diff_tab = []
    for sy in site_years:
        r2 = p[p["test"] == sy].pivot_table(index="repeat", columns="feature_set", values="r2")
        diff_tab.append({"site-year": pretty(sy),
                         **{k: pm(mu, sd) for k, (mu, sd) in paired_diffs(r2).items()}})
    lines += [md_table(pd.DataFrame(diff_tab)), ""]

    # spatial buffer: training cells removed per fold
    nb = m[(m["experiment"] == "E1") & (m["metric"].isin(["n_buffer_removed", "n_train"]))]
    if not nb.empty:
        nbw = nb[nb["feature_set"] == nb["feature_set"].iloc[0]].pivot_table(
            index="test", columns="metric", values="value", aggfunc="mean")
        buf_tab = [{"site-year": pretty(sy), "training cells / fold (after buffer)": f"{nbw.loc[sy, 'n_train']:.0f}",
                    "removed by buffer / fold": f"{nbw.loc[sy, 'n_buffer_removed']:.0f}",
                    "removed %": f"{100 * nbw.loc[sy, 'n_buffer_removed'] / (nbw.loc[sy, 'n_train'] + nbw.loc[sy, 'n_buffer_removed']):.1f}"}
                   for sy in site_years if sy in nbw.index]
        lines += ["Spatial buffer (training cells adjacent to the fold's test blocks dropped), mean over folds:", "",
                  md_table(pd.DataFrame(buf_tab)), ""]

    # in-fold feature selection: number of features kept
    ns = m[(m["experiment"] == "E1") & (m["metric"] == "n_selected")]
    if not ns.empty:
        nt = ns.pivot_table(index="test", columns="feature_set", values="value", aggfunc=["mean", "min", "max"])
        sel_tab = [{"site-year": pretty(sy), **{fs: f"{nt[('mean', fs)][sy]:.0f} ({nt[('min', fs)][sy]:.0f}–"
                                                   f"{nt[('max', fs)][sy]:.0f})"
                                                for fs in FS_PRESENT if ("mean", fs) in nt.columns}}
                   for sy in site_years]
        lines += ["Features kept by the in-fold selection, mean (min–max) over folds and repeats:", "",
                  md_table(pd.DataFrame(sel_tab)), ""]

    # per-fold spread
    f = m[(m["experiment"] == "E1") & (m["fold"] >= 0) & (m["metric"] == "r2")]
    spread = f.groupby(["test", "feature_set"])["value"].agg(["min", "max"]).reset_index()
    worst = spread.loc[spread["min"].idxmin()]
    lines += [f"Per-fold R² ranges from {spread['min'].min():.2f} to {spread['max'].max():.2f}; the lowest single "
              f"fold is {worst['min']:.2f} ({pretty(worst['test'])}, {worst['feature_set']}).", ""]
    return agg


def e1_scatter():
    path = experiment_dir() / "predictions_E1.parquet"
    if not path.exists():
        return
    pr = pd.read_parquet(path)
    pr = pr[pr["repeat"] == 0]
    pr["sy"] = pr["site"] + "_" + pr["year"].astype(str)
    site_years = list(dict.fromkeys(pr["sy"]))
    fig, axes = plt.subplots(len(site_years), len(FS_PRESENT), figsize=(3.2 * len(FS_PRESENT), 3.0 * len(site_years)),
                             squeeze=False)
    for i, sy in enumerate(site_years):
        d = pr[pr["sy"] == sy]
        lim = (0, float(max(d["obs"].max(), d["pred"].max())) * 1.03)
        for j, fs in enumerate(FS_PRESENT):
            ax = axes[i, j]
            s = d[d["fs"] == fs]
            ax.hexbin(s["obs"], s["pred"], gridsize=40, extent=lim + lim, mincnt=1, cmap="Blues", bins="log")
            ax.plot(lim, lim, color=INK, lw=1, ls="--")
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_aspect("equal")
            ax.grid(False)
            r2 = 1 - np.sum((s["obs"] - s["pred"]) ** 2) / np.sum((s["obs"] - s["obs"].mean()) ** 2)
            rmse = np.sqrt(np.mean((s["obs"] - s["pred"]) ** 2))
            ax.text(0.04, 0.96, f"R² = {r2:.2f}\nRMSE = {rmse:.2f} m", transform=ax.transAxes,
                    va="top", fontsize=8)
            if i == 0:
                ax.set_title(fs)
            if j == 0:
                ax.set_ylabel(f"{pretty(sy)}\npredicted (m)")
            if i == len(site_years) - 1:
                ax.set_xlabel("observed P95 height (m)")
    fig.suptitle("E1 out-of-fold predictions, repeat 0 (colour = log count)")
    fig.tight_layout()
    fig.savefig(fig_dir() / "e1_scatter.png", dpi=130)
    plt.close(fig)


# --- E2 ------------------------------------------------------------------------
def e2_section(m, lines):
    p1 = pooled(m, "E1")
    p1 = p1[p1["test"] == CO22].set_index(["feature_set", "repeat"])
    p2 = pooled(m, "E2").set_index(["feature_set", "repeat"])
    reps = sorted(set(p2.index.get_level_values("repeat")) & set(p1.index.get_level_values("repeat")))

    # per-fold matched differences (same folds, same repeat)
    f1 = m[(m["experiment"] == "E1") & (m["test"] == CO22) & (m["fold"] >= 0)]
    f2 = m[(m["experiment"] == "E2") & (m["fold"] >= 0)]
    fk = ["feature_set", "repeat", "fold", "metric"]
    ff = f1.set_index(fk)["value"].to_frame("e1").join(f2.set_index(fk)["value"].to_frame("e2"), how="inner")
    ff = ff.reset_index()

    tab, plot = [], []
    for fs in FS_PRESENT:
        a = p1.loc[fs].loc[reps]
        b = p2.loc[fs].loc[reps]
        loss_r2 = a["r2"] - b["r2"]
        loss_rmse = b["rmse"] - a["rmse"]
        fr = ff[(ff["feature_set"] == fs) & (ff["metric"] == "r2")]
        tab.append({"FS": fs, "E1 R² (CO22 in-domain)": pm(a["r2"].mean(), a["r2"].std()),
                    "E2 R² (TZ22 → CO22)": pm(b["r2"].mean(), b["r2"].std()),
                    "transfer loss ΔR² (E1 − E2)": pm(loss_r2.mean(), loss_r2.std()),
                    "per-fold ΔR²": pm((fr["e1"] - fr["e2"]).mean(), (fr["e1"] - fr["e2"]).std()),
                    "E1 RMSE": pm(a["rmse"].mean(), a["rmse"].std(), 2),
                    "E2 RMSE": pm(b["rmse"].mean(), b["rmse"].std(), 2),
                    "ΔRMSE (E2 − E1)": pm(loss_rmse.mean(), loss_rmse.std(), 2),
                    "E2 bias (m)": pm(b["bias"].mean(), b["bias"].std(), 2)})
        plot.append((fs, a, b))

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    x = np.arange(len(FS_PRESENT))
    w = 0.38
    for ax, (col, label) in zip(axes, [("r2", "Pooled R², Colombia 2022"), ("rmse", "Pooled RMSE (m), Colombia 2022")]):
        for i, (fs, a, b) in enumerate(plot):
            ax.bar(i - w / 2, a[col].mean(), w - 0.03, yerr=a[col].std(), color=FS_COLORS[fs], capsize=2,
                   error_kw={"ecolor": INK2})
            ax.bar(i + w / 2, b[col].mean(), w - 0.03, yerr=b[col].std(), color=FS_COLORS[fs], alpha=0.45,
                   hatch="///", edgecolor=FS_COLORS[fs], capsize=2, error_kw={"ecolor": INK2})
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(x, FS_PRESENT)
        ax.set_ylabel(label)
        ax.grid(axis="x", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#8a8984"),
               plt.Rectangle((0, 0), 1, 1, facecolor="#8a8984", alpha=0.45, hatch="///", edgecolor="#8a8984")]
    axes[0].legend(handles, ["E1 in-domain (CV)", "E2 cross-country (TZ 2022 → CO 2022)"], loc="lower left",
                   fontsize=8)
    fig.suptitle("E2 cross-country transfer vs in-domain (mean ± sd over repeats)")
    fig.tight_layout()
    fig.savefig(fig_dir() / "e2_transfer.png", dpi=150)
    plt.close(fig)

    h_tz = pd.read_parquet(table_path("tanzania", 2022), columns=["height"])["height"]
    h_co = pd.read_parquet(table_path("colombia", 2022), columns=["height"])["height"]
    lines += ["## E2: cross-country transfer (train Tanzania 2022 → test Colombia 2022)", "",
              f"One model per FS and repeat fitted on all Tanzania 2022 cells, evaluated on all Colombia 2022 cells "
              f"(mean ± sd over {len(reps)} repeats). Transfer loss = E1 − E2 pooled R² within the same repeat; "
              "'per-fold ΔR²' uses the matched E1 test folds (mean ± sd over repeats × folds).", "",
              md_table(pd.DataFrame(tab)), "",
              f"Target ranges differ strongly: Tanzania 2022 height mean {h_tz.mean():.1f} m (P95 {h_tz.quantile(.95):.1f}, "
              f"max {h_tz.max():.1f}); Colombia 2022 mean {h_co.mean():.1f} m (P95 {h_co.quantile(.95):.1f}, "
              f"max {h_co.max():.1f}). {100 * (h_co > h_tz.max()).mean():.0f}% of Colombia cells are taller than "
              f"the tallest Tanzania cell and {100 * (h_co > h_tz.quantile(.95)).mean():.0f}% exceed the Tanzania P95. "
              "Tree ensembles (including XGBoost) cannot extrapolate beyond the training target range, so E2 is dominated by a large "
              "negative bias; negative R² here means worse than predicting the Colombia mean.", ""]
    return plot


# --- E3 ------------------------------------------------------------------------
def e3_section(m, lines):
    p3 = pooled(m, "E3")
    if p3.empty:
        lines += ["## E3: few-shot recovery", "", "Not run.", ""]
        return
    p1 = pooled(m, "E1")
    p1 = p1[(p1["test"] == CO22) & (p1["repeat"] == 0)].set_index("feature_set")
    p2 = pooled(m, "E2")
    p2 = p2[p2["repeat"] == 0].set_index("feature_set")
    p3 = p3.copy()
    p3["recovery"] = [(r - p2.loc[fs, "r2"]) / (p1.loc[fs, "r2"] - p2.loc[fs, "r2"])
                      for r, fs in zip(p3["r2"], p3["feature_set"])]
    agg = p3.groupby(["feature_set", "variant", "k"]).agg(
        r2=("r2", "mean"), r2_sd=("r2", "std"), rmse=("rmse", "mean"), rmse_sd=("rmse", "std"),
        rec=("recovery", "mean"), rec_sd=("recovery", "std"), shots=("n_shot_cells", "mean"),
        shots_sd=("n_shot_cells", "std"), bias=("bias", "mean"), draws=("draw", "nunique")).reset_index()
    n_draws = int(agg["draws"].max())
    ks = sorted(agg["k"].unique())

    fig = plt.figure(figsize=(3.7 * len(FS_PRESENT), 7))
    gs = fig.add_gridspec(2, len(FS_PRESENT), height_ratios=[1, 0.9])
    for j, fs in enumerate(FS_PRESENT):
        ax = fig.add_subplot(gs[0, j])
        e1, e2 = p1.loc[fs, "r2"], p2.loc[fs, "r2"]
        lo_data = []
        for variant, ls, mk in [("tz+k", "-", "o"), ("k-only", "--", "s")]:
            a = agg[(agg["feature_set"] == fs) & (agg["variant"] == variant)].sort_values("k")
            if a.empty:
                continue
            ax.errorbar(a["k"], a["r2"], yerr=a["r2_sd"].fillna(0), color=FS_COLORS[fs], ls=ls, marker=mk, ms=5,
                        lw=2, capsize=2, label=variant)
            lo_data.append((a["r2"] - a["r2_sd"].fillna(0)).min())
        ax.axhline(e1, color=INK, lw=1, ls=":")
        ax.text(ks[0], e1, f" E1 ceiling {e1:.2f}", va="bottom", fontsize=7.5, color=INK2)
        lo = min(min(lo_data), 0) - 0.05
        lo = max(lo, -1.0)
        if e2 >= lo:
            ax.axhline(e2, color=INK2, lw=1, ls="-.")
            ax.text(ks[0], e2, f" E2 floor {e2:.2f}", va="bottom", fontsize=7.5, color=INK2)
        else:
            ax.text(0.02, 0.02, f"E2 floor = {e2:.2f} (below axis)", transform=ax.transAxes, fontsize=7.5,
                    color=INK2)
        ax.set_ylim(lo, max(e1, agg["r2"].max()) + 0.08)
        ax.set_xscale("log")
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.minorticks_off()
        ax.set_title(fs)
        ax.set_xlabel("k shot blocks (Colombia 2022)")
        if j == 0:
            ax.set_ylabel("Pooled R², Colombia 2022")
        ax.legend(loc="lower right", fontsize=8)
    ax = fig.add_subplot(gs[1, :])
    for fs in FS_PRESENT:
        for variant, ls, mk in [("tz+k", "-", "o"), ("k-only", "--", "s")]:
            a = agg[(agg["feature_set"] == fs) & (agg["variant"] == variant)].sort_values("k")
            if a.empty:
                continue
            ax.errorbar(a["k"], a["rec"], yerr=a["rec_sd"].fillna(0), color=FS_COLORS[fs], ls=ls, marker=mk, ms=5,
                        lw=2, capsize=2, label=f"{fs} {variant}")
    ax.axhline(1, color=INK, lw=1, ls=":")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.minorticks_off()
    ax.set_xlabel("k shot blocks")
    ax.set_ylabel("recovery = (R²k − R²E2) / (R²E1 − R²E2)")
    ax.legend(ncol=3, fontsize=8, loc="lower right")
    fig.suptitle(f"E3 few-shot recovery, Colombia 2022 (repeat 0, mean ± sd over {n_draws} draws)")
    fig.tight_layout()
    fig.savefig(fig_dir() / "e3_recovery.png", dpi=150)
    plt.close(fig)

    # paired tz+k - k-only (same shot blocks and model seed for a given fold, k, draw)
    wide = p3.pivot_table(index=["feature_set", "k", "draw"], columns="variant", values="r2").reset_index()
    both = {"tz+k", "k-only"} <= set(wide.columns)
    if both:
        wide["gain"] = wide["tz+k"] - wide["k-only"]
        gain = wide.groupby(["feature_set", "k"])["gain"].agg(["mean", "std"])
    variants_run = [v for v in ("tz+k", "k-only") if v in set(agg["variant"])]

    tab = []
    for fs in FS_PRESENT:
        for k in ks:
            row = {"FS": fs, "k": k}
            for variant in variants_run:
                a = agg[(agg["feature_set"] == fs) & (agg["variant"] == variant) & (agg["k"] == k)].iloc[0]
                row[f"{variant} R²"] = pm(a["r2"], a["r2_sd"])
                row[f"{variant} recovery"] = pm(a["rec"], a["rec_sd"], 2)
                row["shot cells/fold"] = f"{a['shots']:.0f} ± {a['shots_sd']:.0f}"
            if both:
                row["tz+k − k-only ΔR²"] = pm(gain.loc[(fs, k), "mean"], gain.loc[(fs, k), "std"])
            tab.append(row)
    ref = pd.DataFrame([{"FS": fs, "E1 R² (r0)": f"{p1.loc[fs, 'r2']:.3f}", "E2 R² (r0)": f"{p2.loc[fs, 'r2']:.3f}"}
                        for fs in FS_PRESENT])
    lines += ["## E3: few-shot recovery in Colombia 2022", "",
              f"Repeat 0 only (the E1 colombia-2022 partition of repeat 0). For each of the 5 folds, k shot blocks are "
              f"drawn from that fold's training blocks ({n_draws} draws per k); models are tested on the fold's test "
              "blocks and pooled over the 5 folds. Values: mean ± sd over draws. 'shot cells/fold' = mean number of "
              "Colombia cells in the k shot blocks (blocks hold 1–100 cells).", "",
              "Reference levels (pooled, repeat 0):", "", md_table(ref), "", md_table(pd.DataFrame(tab)), ""]
    return agg


# --- E4 ------------------------------------------------------------------------
E4_PANELS = [  # (site, A, B, variant)
    ("colombia", 2021, 2022, "overlap"), ("colombia", 2021, 2022, "full"), ("colombia", 2022, 2021, "main"),
    ("colombia", 2021, 2022, "matched"), ("colombia", 2022, 2021, "matched"),
    ("tanzania", 2022, 2024, "main"), ("tanzania", 2024, 2022, "main"),
]


def e4_section(m, lines):
    p = pooled(m, "E4")
    if p.empty:
        lines += ["## E4: cross-year transfer", "", "Not run.", ""]
        return
    tab, panels = [], []
    present = set(p["variant"])
    for site, ya, yb, variant in [q for q in E4_PANELS if q[3] in present]:
        A, B = f"{site}_{ya}", f"{site}_{yb}"
        rows = {}
        for fs in FS_PRESENT:
            sel = p[(p["test"] == B) & (p["variant"] == variant) & (p["feature_set"] == fs)]
            cross = sel[sel["train"] == A].set_index("repeat")
            base = sel[sel["train"] == B].set_index("repeat")
            loss = (base["r2"] - cross["r2"]).dropna()
            tab.append({"direction": f"{pretty(A)} → {yb}" + ("" if variant == "main" else f" ({variant})"),
                        "FS": fs, "baseline B→B R²": pm(base["r2"].mean(), base["r2"].std()),
                        "cross-year A→B R²": pm(cross["r2"].mean(), cross["r2"].std()),
                        "temporal loss ΔR²": pm(loss.mean(), loss.std()),
                        "baseline RMSE": pm(base["rmse"].mean(), base["rmse"].std(), 2),
                        "cross RMSE": pm(cross["rmse"].mean(), cross["rmse"].std(), 2),
                        "cross bias (m)": pm(cross["bias"].mean(), cross["bias"].std(), 2),
                        "n test": int(base["n"].iloc[0])})
            rows[fs] = (base, cross)
        panels.append((site, ya, yb, variant, rows))

    ncol = 3
    nrow = (len(panels) + 1 + ncol - 1) // ncol          # panels + one legend slot
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol * len(FS_PRESENT) / 3, 3.2 * nrow), sharey=True,
                             squeeze=False)
    order = [(i // ncol, i % ncol) for i in range(len(panels))]
    w = 0.38
    for (r, c), (site, ya, yb, variant, rows) in zip(order, panels):
        ax = axes[r, c]
        for i, fs in enumerate(FS_PRESENT):
            base, cross = rows[fs]
            ax.bar(i - w / 2, base["r2"].mean(), w - 0.03, yerr=base["r2"].std(), color=FS_COLORS[fs], capsize=2,
                   error_kw={"ecolor": INK2})
            ax.bar(i + w / 2, cross["r2"].mean(), w - 0.03, yerr=cross["r2"].std(), color=FS_COLORS[fs],
                   alpha=0.45, hatch="///", edgecolor=FS_COLORS[fs], capsize=2, error_kw={"ecolor": INK2})
        ax.set_xticks(range(len(FS_PRESENT)), FS_PRESENT)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.grid(axis="x", visible=False)
        title = f"{site.capitalize()}: {ya} → {yb}" + ("" if variant == "main" else f" ({variant} cells)")
        ax.set_title(title)
        if c == 0:
            ax.set_ylabel("Pooled R²")
    for i in range(len(panels), nrow * ncol):
        axes[i // ncol, i % ncol].axis("off")
    leg_ax = axes[len(panels) // ncol, len(panels) % ncol]
    handles = [plt.Rectangle((0, 0), 1, 1, color="#8a8984"),
               plt.Rectangle((0, 0), 1, 1, facecolor="#8a8984", alpha=0.45, hatch="///", edgecolor="#8a8984")]
    handles += [plt.Rectangle((0, 0), 1, 1, color=FS_COLORS[fs]) for fs in FS_PRESENT]
    leg_ax.legend(handles, ["baseline B → B (same test cells)", "cross-year A → B"] + list(FS_PRESENT),
                  loc="center", fontsize=9)
    fig.suptitle("E4 cross-year transfer (mean ± sd over repeats)")
    fig.tight_layout()
    fig.savefig(fig_dir() / "e4_temporal.png", dpi=150)
    plt.close(fig)

    reps = int(p["repeat"].nunique())
    lines += ["## E4: cross-year transfer", "",
              f"One partition per site over the union of both years' blocks; A→B trains on year A's training-block "
              f"cells and tests on year B's test-block cells; the baseline B→B trains on year B's training-block "
              f"cells and tests on the same cells. Temporal loss = (B→B) − (A→B) pooled R² within a repeat; "
              f"mean ± sd over {reps} repeats. Colombia 2021→2022: 'overlap' (main) tests only on 2022 cells that also "
              "exist in 2021, 'full' on all 2022 test cells. 'matched' (Colombia, both directions): both years "
              "train only on cells present in both years inside the training blocks and are tested on the same "
              "shared test cells, so cross-year and baseline differ only in the year (equal training-set size and "
              "footprint). Use 'matched' for the Colombia temporal loss; 'overlap'/'full' confound year with the "
              "3–4× larger 2022 training set.", "", md_table(pd.DataFrame(tab)), ""]


# --- importances -----------------------------------------------------------------
def importance_figure():
    path = experiment_dir() / "importances_E1.csv"
    if not path.exists():
        return
    imp = pd.read_csv(path)
    sys_ = [s for s in (CO22, TZ22) if s in set(imp["site_year"])]
    fig, axes = plt.subplots(len(sys_), len(FS_PRESENT), figsize=(4 * len(FS_PRESENT), 4.2 * len(sys_)))
    axes = np.atleast_2d(axes)
    for i, sy in enumerate(sys_):
        for j, fs in enumerate(FS_PRESENT):
            ax = axes[i, j]
            d = imp[(imp["site_year"] == sy) & (imp["feature_set"] == fs)].sort_values("rank").head(15)
            colors = [FS_COLORS["HC"] if f.startswith("hc_") else FS_COLORS["TES"] if f.startswith("tes_")
                      else FS_COLORS["AEF"] for f in d["feature"]]
            ax.barh(np.arange(len(d))[::-1], d["importance"], color=colors, height=0.75)
            ax.set_yticks(np.arange(len(d))[::-1], d["feature"], fontsize=7)
            ax.grid(axis="y", visible=False)
            ax.set_title(f"{pretty(sy)} — {fs}")
            if i == len(sys_) - 1:
                ax.set_xlabel("gain importance (mean over folds, repeat 0)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=FS_COLORS[k]) for k in ("HC", "AEF", "TES")]
    fig.legend(handles, ["handcrafted (hc_*)", "AlphaEarth (aef_*)", "Tessera (tes_*)"], loc="upper right", ncol=3)
    fig.suptitle("E1 top-15 XGBoost gain importances")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(fig_dir() / "importances_E1.png", dpi=130)
    plt.close(fig)


def key_results(m) -> list[str]:
    """Short data-driven bullet list of the main comparisons (no interpretation beyond the numbers)."""
    out = ["## Key results", ""]
    has = set(m["experiment"])
    if "E1" in has:
        p = pooled(m, "E1")
        for sy in dict.fromkeys(p["test"]):
            r2 = p[p["test"] == sy].pivot_table(index="repeat", columns="feature_set", values="r2")
            diffs = paired_diffs(r2)
            out.append(f"- E1 {pretty(sy)}: R² " + ", ".join(f"{fs} {r2[fs].mean():.2f}" for fs in FS_PRESENT) +
                       ("; " + ", ".join(f"{k} = {pm(mu, sd)}" for k, (mu, sd) in diffs.items()) if diffs else "")
                       + ".")
    if {"E1", "E2"} <= has:
        p2 = pooled(m, "E2").groupby("feature_set")[["r2", "bias"]].mean()
        out.append("- E2 (Tanzania 2022 → Colombia 2022) fails for every FS: R² " +
                   ", ".join(f"{fs} {p2.loc[fs, 'r2']:.2f}" for fs in FS_PRESENT) +
                   "; mean bias " + ", ".join(f"{fs} {p2.loc[fs, 'bias']:.1f} m" for fs in FS_PRESENT) + ".")
    if "E3" in has:
        p3 = pooled(m, "E3")
        kmax = int(p3["k"].max())
        a = p3[p3["k"] == kmax].groupby(["feature_set", "variant"])["r2"].mean()
        wide = p3.pivot_table(index=["feature_set", "k", "draw"], columns="variant", values="r2")
        text = f"- E3 at k = {kmax} blocks: tz+k R² " + ", ".join(f"{fs} {a.loc[(fs, 'tz+k')]:.2f}" for fs in FS_PRESENT)
        if "k-only" in wide.columns:
            gain = (wide["tz+k"] - wide["k-only"]).groupby(level=["feature_set", "k"]).mean()
            text += ("; k-only R² " + ", ".join(f"{fs} {a.loc[(fs, 'k-only')]:.2f}" for fs in FS_PRESENT) +
                     f". Mean tz+k − k-only ΔR² over all k and FS ranges from {gain.min():.3f} to {gain.max():.3f}")
        else:
            text += " (tz+k only; k-only not run)"
        out.append(text + ".")
    if "E4" in has:
        p = pooled(m, "E4")
        for site, ya, yb, variant in [q for q in E4_PANELS if q[3] in set(p["variant"])]:
            sel = p[(p["test"] == f"{site}_{yb}") & (p["variant"] == variant)]
            loss = {}
            for fs in FS_PRESENT:
                s = sel[sel["feature_set"] == fs]
                base = s[s["train"] == f"{site}_{yb}"].set_index("repeat")["r2"]
                cross = s[s["train"] == f"{site}_{ya}"].set_index("repeat")["r2"]
                loss[fs] = (base - cross).mean()
            out.append(f"- E4 {site.capitalize()} {ya} → {yb}" + ("" if variant == "main" else f" ({variant})") +
                       ": temporal loss ΔR² " + ", ".join(f"{fs} {v:.3f}" for fs, v in loss.items()) + ".")
    out.append("")
    return out


def caveats(m) -> list[str]:
    cfg = load_config()
    n_draws_run = int(m.loc[m["experiment"] == "E3", "draw"].max() + 1) if (m["experiment"] == "E3").any() else 0
    out = ["## Caveats", "",
           "- **Tessera coverage:** Tessera leaves pixels with fewer than 15 cloud-free observations per year "
           "empty; about 21 % (2021) and 18 % (2022) of Colombia cells have no Tessera data. With "
           "`pipeline.cells = covered` these cells are excluded for every feature set, so results describe the "
           "less cloudy part of the Colombia site; Tanzania is fully covered. Earlier all-cell results (no "
           "Tessera, no selection) are archived in `results/archive_xgb_allcells_noselect/`.",
           "- Tessera (TESSERA, Feng et al. 2025, arXiv:2506.20380) was accessed on 2026-10-02 from "
           "https://registry.opendata.aws/tessera (dClimate v1.1 store, CC-BY-SA).",
           "- **E3 uses repeat 0 only** (one block partition); its sd is over draws of shot blocks, not over "
           "partitions." + (f" {n_draws_run} draws per k were run (config: {cfg['fewshot']['n_draws']})."
                            if n_draws_run else ""),
           "- **Recovery is sensitive to the E2 floor.** Where E2 R² is strongly negative (systematic bias from "
           "transferring a model trained on much lower Tanzanian canopies), even a weak model trained on a few "
           "Colombia blocks removes most of the bias and scores a high 'recovery'. Compare tz+k against k-only, "
           "and both against the E1 ceiling, rather than reading recovery alone.",
           "- **Cross-year gaps differ:** 1 year for Colombia (2021→2022) and 2 years for Tanzania (2022→2024). "
           "Real canopy change, lidar acquisition differences and image-composite differences are all mixed into "
           "the temporal loss.",
           "- **S1 acquisition density differs between site-years** (S1A descending only; Colombia 2021 has only "
           "22 S1A scenes), so the S1 part of HC is noisier in some years. AEF annual embeddings are built from "
           "their own sensor stack and are not affected the same way.",
           ("- **Spatial buffer:** training cells within {b} cell(s) ({m} m) of a fold's test blocks are dropped "
            "(E1, E3 shot cells, E4), so no test cell has an adjacent training cell; see the E1 table of removed "
            "cells. Spatial autocorrelation beyond {m} m can still make in-domain scores somewhat optimistic."
            if cfg["pipeline"]["buffer_cells"] > 0 else
            "- **Spatial autocorrelation across block edges:** test blocks touch training blocks without a buffer, "
            "so cells near block edges have correlated neighbours in training. This affects every FS equally but "
            "makes all in-domain scores somewhat optimistic.").format(
               b=cfg["pipeline"]["buffer_cells"], m=100 * cfg["pipeline"]["buffer_cells"]),
           "- **Uneven block sizes** (1–100 cells): folds are balanced by cell count, but the number of blocks per "
           "fold differs, and few-shot k blocks contain a variable number of cells (reported as shot cells).",
           "- Colombia 2021 is small (2212 cells, 37 blocks) and lies inside the 2022 footprint; E1 numbers for it "
           "are noisier.",
           "- Gain importances are biased toward correlated / continuous features and are descriptive only.",
           "- No hyper-parameter tuning; one fixed XGBoost configuration for all feature sets. Rankings could change with tuning or "
           "another learner.", ""]
    return out


def main() -> None:
    m = pd.read_csv(experiment_dir() / "metrics.csv")
    FS_PRESENT[:] = [fs for fs in FEATURE_SETS if fs in set(m["feature_set"])]
    lines = ["# Canopy height (P95, 100 m): handcrafted S1/S2 vs AlphaEarth and Tessera embeddings", "",
             {"xgboost": "XGBoost (n_estimators = {n}, learning_rate = {lr}, max_depth = {md}, subsample = {ss}, "
                         "colsample_bytree = {cs}, no tuning)",
              "tabicl": "TabICLv2 (TabICLRegressor, n_estimators = {ti}, CPU; no tuning)",
              "plsr": "PLS regression ({pc} components, standardised features, predictions clipped at 0 m; linear baseline)",
              }[load_config()["model"]["learner"]] +
             " on 1 km spatial block splits. "
             "FS: HC = 98 handcrafted S1/S2 features (incl. 24 GLCM), AEF = 128 AlphaEarth features (64 mean + 64 "
             "std), TES = 256 Tessera features (128 mean + 128 std), HC+AEF / HC+TES = combinations. Metrics: R², RMSE (m), rRMSE = RMSE / mean(obs), bias = mean(pred − obs). "
             f"Figures in `results/{load_config()['model']['learner']}/figures/`.".format(
                                                     ti=load_config()["tabicl"]["n_estimators"],
                                                     pc=load_config()["plsr"]["n_components"],
                                                     n=load_config()["model"]["n_estimators"],
                                                     lr=load_config()["model"]["learning_rate"],
                                                     md=load_config()["model"]["max_depth"],
                                                     ss=load_config()["model"]["subsample"],
                                                     cs=load_config()["model"]["colsample_bytree"]), ""]
    pl = load_config()["pipeline"]
    sel = pl["feature_selection"]
    lines += ["**Pipeline.** Cells: " + ("only cells with Tessera data (`covered`), identical for every feature set; "
              "block partitions are built on all cells first, so folds match earlier all-cell runs."
              if pl["cells"] == "covered" else "all cells.") +
              " Feature selection: " + ("none." if sel["method"] == "none" else
              ("features whose mean |SHAP| is above the mean over all features (threshold = mean)"
               if sel["method"] == "shap_mean" else f"top-{sel['k']} features by mean |SHAP|") +
              f" of a selector XGBoost (same hyper-parameters), fitted inside every training fold (SHAP on at most "
              f"{sel['max_rows']} training rows); the final XGBoost is refitted on the selected features." +
              (" The number of selected features therefore varies by feature set and fold (see the selection table)."
               if sel["method"] == "shap_mean" else "")), ""]
    lines += key_results(m)
    run_log = experiment_dir() / "run_log.csv"
    if run_log.exists():
        runs = pd.read_csv(run_log).groupby("experiment").tail(1)       # latest run per experiment
        lines += ["## Run settings", "", "Latest run per experiment (from `results/run_log.csv`; n_draws only for E3).",
                  "", md_table(runs[["experiment", "time", "n_estimators", "n_repeats", "n_draws", "quick"]]), ""]
    if (m["experiment"] == "E1").any():
        e1_section(m, lines)
        e1_scatter()
        importance_figure()
    if (m["experiment"] == "E2").any() and (m["experiment"] == "E1").any():
        e2_section(m, lines)
    if (m["experiment"] == "E3").any():
        e3_section(m, lines)
    if (m["experiment"] == "E4").any():
        e4_section(m, lines)
    lines += caveats(m)
    lines += ["## Attribution", "", ATTRIBUTION, ""]
    out = experiment_dir() / "summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out} and figures in {fig_dir()}")


if __name__ == "__main__":
    main()
