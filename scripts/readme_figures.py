"""Figures for README.md (HC vs AEF only): docs/figures/*.png

Reads results/{learner}/metrics.csv (E1, E2, E2rev, E4) and data/tables/*.parquet (height distributions).

Usage:
  uv run python scripts/readme_figures.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from hgt.config import ROOT
from hgt.io import experiment_dir, table_path
from hgt.report import FS_COLORS, INK, INK2, pretty

FS = ["HC", "AEF"]
OUT = ROOT / "docs" / "figures"


def pooled(m: pd.DataFrame, exp: str, metric: str) -> pd.DataFrame:
    d = m[(m["experiment"] == exp) & (m["fold"] == -1) & (m["metric"] == metric) & m["feature_set"].isin(FS)]
    return d.groupby(["train", "test", "variant", "feature_set"], as_index=False)["value"].mean()


def val(d: pd.DataFrame, train: str, test: str, fs: str, variant: str = "main") -> float:
    return float(d[(d["train"] == train) & (d["test"] == test) & (d["feature_set"] == fs)
                   & (d["variant"] == variant)]["value"].iloc[0])


def fig_in_domain_vs_temporal(m: pd.DataFrame) -> None:
    e1, e4 = pooled(m, "E1", "r2"), pooled(m, "E4", "r2")
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1, 1.25]})

    # left: in-domain R2
    sys_ = ["colombia_2021", "colombia_2022", "tanzania_2022", "tanzania_2024"]
    x = np.arange(len(sys_))
    for i, fs in enumerate(FS):
        a.bar(x + (i - 0.5) * 0.38, [val(e1, s, s, fs) for s in sys_], 0.36, color=FS_COLORS[fs], label=fs)
    a.set_xticks(x, [pretty(s).replace(" ", "\n") for s in sys_])
    a.set_ylabel("Pooled out-of-fold R²")
    a.set_title("In-domain (blocked CV): AEF is better", loc="left")
    a.legend(loc="upper left", ncol=2)
    a.set_ylim(0, 0.85)
    a.grid(axis="x", visible=False)

    # right: dumbbell, trained on target year (hollow) -> trained on the other year (filled)
    dirs = [("colombia_2021", "colombia_2022", "matched"), ("colombia_2022", "colombia_2021", "matched"),
            ("tanzania_2022", "tanzania_2024", "main"), ("tanzania_2024", "tanzania_2022", "main")]
    y = 0
    ticks = []
    for src, dst, var in dirs:
        for j, fs in enumerate(FS):
            yy = y + (0.17 if j == 0 else -0.17)
            base, cross = val(e4, dst, dst, fs, var), val(e4, src, dst, fs, var)
            b.plot([cross, base], [yy, yy], color=FS_COLORS[fs], lw=2, zorder=2)
            b.scatter([base], [yy], s=55, facecolor="#fcfcfb", edgecolor=FS_COLORS[fs], linewidth=2, zorder=3)
            b.scatter([cross], [yy], s=55, color=FS_COLORS[fs], zorder=3)
            b.text(0.835, yy, f"−{base - cross:.2f}", va="center", ha="right", fontsize=8, color=INK2)
        ticks.append(y)
        y += 1
    b.set_yticks(ticks, [f"{pretty(s).split()[0]} {s.split('_')[1]} → {d.split('_')[1]}" for s, d, _ in dirs])
    b.invert_yaxis()
    b.set_xlim(0.2, 0.85)
    b.set_xlabel("R² on the target year")
    b.set_title("Temporal transfer: AEF loses more (ΔR² at right)", loc="left")
    b.grid(axis="y", visible=False)
    handles = [Line2D([], [], marker="o", color=FS_COLORS[fs], lw=2, label=fs) for fs in FS] + [
        Line2D([], [], marker="o", mfc="#fcfcfb", mec=INK2, mew=2, color="none", label="trained on target year"),
        Line2D([], [], marker="o", color=INK2, ls="none", label="trained on the other year")]
    b.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.17), fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(OUT / "in_domain_vs_temporal.png", dpi=150)
    plt.close(fig)


def fig_geographic(m: pd.DataFrame) -> None:
    bias = pd.concat([pooled(m, "E2", "bias").assign(exp="E2"), pooled(m, "E2rev", "bias").assign(exp="E2rev")])
    rmse = pd.concat([pooled(m, "E2", "rmse").assign(exp="E2"), pooled(m, "E2rev", "rmse").assign(exp="E2rev")])
    conds = [("E2", "Tanzania → Colombia"), ("E2rev", "Colombia → Tanzania")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1, 1.1]})

    x = np.arange(len(conds))
    for i, fs in enumerate(FS):
        v = [float(bias[(bias.exp == e) & (bias.feature_set == fs)]["value"].iloc[0]) for e, _ in conds]
        bars = a.bar(x + (i - 0.5) * 0.36, v, 0.34, color=FS_COLORS[fs], label=fs)
        for bar, val_ in zip(bars, v):
            a.text(bar.get_x() + bar.get_width() / 2, val_ + (0.5 if val_ > 0 else -0.5), f"{val_:+.1f}".replace("-", "−"),
                   ha="center", va="bottom" if val_ > 0 else "top", fontsize=9, color=INK)
    a.axhline(0, color=INK2, lw=1)
    a.set_xticks(x, [c for _, c in conds])
    a.set_ylabel("Mean bias: prediction − observation (m)")
    a.set_ylim(-14, 14)
    a.set_title("Cross-country transfer: the bias flips sign", loc="left")
    a.legend(loc="upper left", ncol=2)
    a.grid(axis="x", visible=False)

    tz = pd.read_parquet(table_path("tanzania", 2022))["height"]
    co = pd.read_parquet(table_path("colombia", 2022))["height"]
    bins = np.arange(0, 66, 2)
    b.hist(co, bins=bins, color="#9b9a96", alpha=0.55, label="Colombia 2022", density=True)
    b.hist(tz, bins=bins, color=INK, histtype="step", linewidth=1.6, label="Tanzania 2022", density=True)
    b.axvline(co.mean(), color="#9b9a96", ls="--", lw=1)
    b.axvline(tz.mean(), color=INK, ls="--", lw=1)
    b.text(tz.mean() + 0.8, b.get_ylim()[1] * 0.92, f"Tanzania\nmean {tz.mean():.1f} m", fontsize=8, va="top")
    b.text(co.mean() + 0.8, b.get_ylim()[1] * 0.55, f"Colombia\nmean {co.mean():.1f} m", fontsize=8, va="top",
           color=INK2)
    b.set_xlabel("Cell P95 canopy height (m)")
    b.set_ylabel("Density")
    b.set_title("The two sites have very different heights", loc="left")
    b.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "geographic_transfer.png", dpi=150)
    plt.close(fig)
    print(rmse.pivot_table(index="exp", columns="feature_set", values="value").round(2))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(experiment_dir() / "metrics.csv")
    fig_in_domain_vs_temporal(m)
    fig_geographic(m)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
