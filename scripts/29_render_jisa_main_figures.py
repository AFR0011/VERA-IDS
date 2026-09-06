#!/usr/bin/env python3
"""Render JISA main-text numerical Figures 2-11 from frozen source tables.

Run scripts/28_build_jisa_main_figure_sources.py first. This renderer performs no
scientific re-analysis and writes SVG, PDF, and 300-dpi PNG versions of each figure.
Figure 1 is the non-numerical methodology overview and is handled separately.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs/12_jisa_finalization/20_main_figure_sources"
OUT = ROOT / "outputs/12_jisa_finalization/21_main_figures"


def setup() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8.5, "axes.labelsize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.6,
        "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "figure.facecolor": "white", "axes.facecolor": "white",
    })


def read(name: str) -> pd.DataFrame:
    path = SRC / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run scripts/28_build_jisa_main_figure_sources.py first")
    return pd.read_csv(path)


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("svg", {}), ("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)


def clean(ax: plt.Axes, grid: str | None = None) -> None:
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis=grid, linewidth=0.35, alpha=0.25); ax.set_axisbelow(True)


def panel(ax: plt.Axes, text: str) -> None:
    ax.text(-0.08, 1.03, text, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom")


def ref_name(row: pd.Series, include_task: bool = False) -> str:
    paper = str(row["paper"])
    name = "Adewole-derived XGB" if paper.startswith("adewole") else "Neto-derived RF" if paper.startswith("neto") else str(row.get("model_profile", paper))
    task = str(row.get("closed_set_task_used", "")).strip()
    suffix = f" ({task} reference)" if include_task and task and task.lower() != "nan" else ""
    return f"{name} / {row['dataset']}{suffix}"


def figure2() -> None:
    df = read("fig2_closed_set_composition.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.3), sharey=True)
    for i, (ax, ds) in enumerate(zip(axes, ["CICIDS2017", "CICIoT2023"])):
        sub = df[df.dataset.astype(str) == ds].copy()
        sub["ord"] = sub.model_family.map({"rf": 0, "xgb": 1}); sub = sub.sort_values("ord")
        x = np.arange(len(sub))
        for xi, (_, r) in zip(x, sub.iterrows()):
            ax.plot([xi, xi], [r.final_strict_macro_f1, r.stage2_macro_f1], color="0.55", lw=0.9)
        ax.scatter(x, sub.stage2_macro_f1, marker="o", s=34, facecolors="white", edgecolors="black", lw=0.9)
        ax.scatter(x, sub.final_strict_macro_f1, marker="s", s=30, color="black")
        direct = float(sub.direct_multiclass_macro_f1.iloc[0])
        ax.axhline(direct, color="0.25", ls="--", lw=0.8)
        ax.text(.98, direct + .006, f"Direct MC {direct:.3f}", ha="right", va="bottom", fontsize=7.2)
        ax.set(xticks=x, xticklabels=["RF", "XGB"], ylim=(.64, .98), xlabel="Model family")
        clean(ax, "y"); panel(ax, f"({'ab'[i]}) {ds}")
    axes[0].set_ylabel("Macro-F1")
    axes[1].legend(handles=[
        Line2D([0], [0], marker="o", ls="none", mfc="white", mec="black", label="Stage 2 conditional"),
        Line2D([0], [0], marker="s", ls="none", color="black", label="Final two-stage"),
        Line2D([0], [0], ls="--", color="0.25", label="Direct multiclass"),
    ], loc="lower right", frameon=False)
    fig.tight_layout(w_pad=2); save(fig, "figure2_closed_set_composition")


def figure3() -> None:
    df = read("fig3_reference_profiles_closed_set.csv").copy()
    df["label"] = df.apply(lambda r: ref_name(r, True), axis=1); df = df.sort_values("closed_set_macro_f1").reset_index(drop=True)
    y = np.arange(len(df)); fig, ax = plt.subplots(figsize=(6.4, 3.1))
    for i, r in df.iterrows():
        ax.plot([r.full_framework_macro_f1_supported_labels, r.closed_set_macro_f1], [i, i], color="0.55", lw=1)
    ax.scatter(df.closed_set_macro_f1, y, marker="o", s=34, facecolors="white", edgecolors="black", lw=.9)
    ax.scatter(df.full_framework_macro_f1_supported_labels, y, marker="s", s=30, color="black")
    ax.set(yticks=y, yticklabels=df.label, xlim=(.62, .97), xlabel="Macro-F1"); clean(ax, "x")
    fig.legend(handles=[
        Line2D([0], [0], marker="o", ls="none", mfc="white", mec="black", label="Published closed-set reference"),
        Line2D([0], [0], marker="s", ls="none", color="black", label="Protocol A composed system"),
    ], loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.5, .01))
    fig.tight_layout(rect=(0, .14, 1, 1)); save(fig, "figure3_reference_profiles_closed_set")


def figure4() -> None:
    df = read("fig4_reference_profiles_protocol_b.csv").copy()
    keys = sorted([tuple(x) for x in df[["paper", "model_profile", "dataset"]].drop_duplicates().to_numpy()], key=lambda x: (x[2], x[0], x[1]))
    fig, axes = plt.subplots(len(keys), 1, figsize=(7, 5.8), sharex=True); axes = np.atleast_1d(axes)
    letters = "abcdefghijklmnopqrstuvwxyz"
    for i, (ax, (paper, profile, ds)) in enumerate(zip(axes, keys)):
        sub = df[(df.paper.astype(str) == str(paper)) & (df.model_profile.astype(str) == str(profile)) & (df.dataset.astype(str) == str(ds))].copy()
        sub = sub.sort_values("unknown_detection_rate").reset_index(drop=True); y = np.arange(len(sub))
        ax.scatter(sub.unknown_detection_rate, y, s=28, color="black"); ax.set(yticks=y, yticklabels=sub.holdout_family, xlim=(0, 1))
        clean(ax, "x"); panel(ax, f"({letters[i]}) {ref_name(sub.iloc[0])}")
        ax.text(.99, .08, f"Mean UDR = {sub.unknown_detection_rate.mean():.3f}", transform=ax.transAxes, ha="right", fontsize=7.4)
    axes[-1].set_xlabel("Held-out-family unknown-detection rate")
    fig.tight_layout(h_pad=1.3); save(fig, "figure4_reference_profiles_protocol_b")


def figure5() -> None:
    df = read("fig5_recovered_cross_split_overlap.csv"); fig, ax = plt.subplots(figsize=(6.2, 3.4)); ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    nodes = {"train": (.5, .82), "val": (.2, .24), "test": (.8, .24)}; w, h = .18, .12
    for name, (x, y) in nodes.items():
        ax.add_patch(Rectangle((x-w/2, y-h/2), w, h, fill=False, lw=.9)); ax.text(x, y, name.capitalize(), ha="center", va="center", fontsize=9)
    labels = {frozenset({"train", "val"}): (.31, .55), frozenset({"train", "test"}): (.69, .55), frozenset({"val", "test"}): (.5, .19)}
    for _, r in df.iterrows():
        left, right = str(r.split_left), str(r.split_right); x0, y0 = nodes[left]; x1, y1 = nodes[right]
        ax.plot([x0, x1], [y0, y1], color="0.45", lw=.9, zorder=0); lx, ly = labels[frozenset({left, right})]
        ax.text(lx, ly, f"{int(r.exact_feature_row_hash_overlap):,}\n({100*float(r.fraction_of_smaller_split):.2f}% of smaller split)", ha="center", va="center", fontsize=8, bbox={"facecolor":"white","edgecolor":"none","pad":1.5})
    fig.tight_layout(); save(fig, "figure5_recovered_cross_split_overlap")


def figure6() -> None:
    df = read("fig6_cicids_partition_sensitivity.csv").sort_values("historical_unknown_detection_rate").reset_index(drop=True); y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for i, r in df.iterrows(): ax.plot([r.historical_unknown_detection_rate, r.new_unknown_detection_rate], [i, i], color="0.55", lw=1)
    ax.scatter(df.historical_unknown_detection_rate, y, marker="o", s=32, facecolors="white", edgecolors="black", lw=.9)
    ax.scatter(df.new_unknown_detection_rate, y, marker="s", s=28, color="black")
    ax.set(yticks=y, yticklabels=df.holdout_family, xlim=(0, 1.02), xlabel="Unknown-detection rate"); clean(ax, "x")
    ax.text(.98, .06, f"Mean UDR: {df.historical_unknown_detection_rate.mean():.3f} → {df.new_unknown_detection_rate.mean():.3f}\nMean |family shift|: {df.abs_delta_unknown_detection_rate.mean():.3f}", transform=ax.transAxes, ha="right", fontsize=7.4)
    ax.legend(handles=[Line2D([0],[0],marker="o",ls="none",mfc="white",mec="black",label="Recovered within-day"), Line2D([0],[0],marker="s",ls="none",color="black",label="Grouped sensitivity")], loc="lower right", frameon=False)
    fig.tight_layout(); save(fig, "figure6_cicids_partition_sensitivity")


def known_scatter(source: str, stem: str, offsets: dict[str, tuple[int,int]]) -> None:
    df = read(source); fig, ax = plt.subplots(figsize=(4.5, 4.1))
    ax.scatter(df.known_recall, df.heldout_udr_mean, s=34, facecolors="white", edgecolors="black", lw=.9)
    for _, r in df.iterrows():
        dx, dy = offsets.get(str(r.family), (4,5)); ax.annotate(str(r.family), (r.known_recall, r.heldout_udr_mean), xytext=(dx,dy), textcoords="offset points", fontsize=7.2)
    ax.plot([0,1],[0,1], ls=":", lw=.7, color="0.65"); ax.set(xlim=(0,1.03), ylim=(0,1.03), xlabel="Known-family recall", ylabel="Held-out-family UDR"); clean(ax,"both")
    rho = df.known_recall.corr(df.heldout_udr_mean, method="spearman"); ax.text(.04,.95,rf"$\rho_s$ = {rho:.3f}", transform=ax.transAxes, ha="left", va="top", fontsize=8)
    fig.tight_layout(); save(fig, stem)


def figure7() -> None:
    known_scatter("fig7_known_vs_heldout_cicids2017.csv", "figure7_known_vs_heldout_cicids2017", {"Botnet":(-45,6),"BruteForce":(5,4),"DDoS":(-34,8),"DoS":(-28,-13),"Scan/Recon":(-56,6),"Web/App":(5,5)})


def figure8() -> None:
    known_scatter("fig8_known_vs_heldout_ciciot2023.csv", "figure8_known_vs_heldout_ciciot2023", {"Botnet":(-48,8),"BruteForce":(5,5),"DDoS":(-38,18),"DoS":(-30,6),"Other":(5,5),"Scan/Recon":(5,-13)})


def heatmap(ax: plt.Axes, df: pd.DataFrame, ds: str, label: str):
    sub = df[df.dataset.astype(str) == ds]; rows = sorted(sub.holdout_family.astype(str).unique()); cols = sorted(sub.destination.astype(str).unique())
    arr = sub.pivot_table(index="holdout_family", columns="destination", values="residual_destination_share_mean").reindex(index=rows, columns=cols).fillna(0).to_numpy(float)
    im = ax.imshow(arr, vmin=0, vmax=1, cmap="Greys", aspect="auto", interpolation="nearest")
    ax.set(xticks=np.arange(len(cols)), xticklabels=cols, yticks=np.arange(len(rows)), yticklabels=rows, xlabel="Residual predicted label", ylabel="Held-out family")
    ax.tick_params(axis="x", labelrotation=45); plt.setp(ax.get_xticklabels(), ha="right")
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            if arr[r,c] >= .10: ax.text(c,r,f"{arr[r,c]:.2f}",ha="center",va="center",fontsize=6.7,color="white" if arr[r,c]>=.58 else "black")
    panel(ax,label); return im


def figure9() -> None:
    df = read("fig9_failure_destination_matrix.csv"); fig, axes = plt.subplots(1,2,figsize=(7.6,4))
    im = heatmap(axes[0],df,"CICIDS2017","(a) CICIDS2017"); heatmap(axes[1],df,"CICIoT2023","(b) CICIoT2023")
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=.026, pad=.02); cb.set_label("Share of residual non-Unknown failures")
    fig.subplots_adjust(left=.10,right=.91,bottom=.24,top=.91,wspace=.42); save(fig,"figure9_failure_destinations")


def figure10() -> None:
    df = read("fig10_rejector_delta_udr.csv"); methods=["margin","entropy","family_conditional"]; markers={"margin":"o","entropy":"s","family_conditional":"^"}; labels={"margin":"Margin","entropy":"Entropy","family_conditional":"Family-conditional"}
    delta=100*pd.to_numeric(df.delta_unknown_detection_rate_vs_max_confidence,errors="coerce"); xmin=min(-5,float(delta.min())-5); xmax=max(5,float(delta.max())+5)
    fig, axes=plt.subplots(1,2,figsize=(7.3,3.9),sharex=True)
    for i,(ax,ds) in enumerate(zip(axes,["CICIDS2017","CICIoT2023"])):
        fams=sorted(df.loc[df.dataset.astype(str)==ds,"holdout_family"].astype(str).unique()); fmap={f:j for j,f in enumerate(fams)}; offs={"margin":-.16,"entropy":0,"family_conditional":.16}
        for m in methods:
            sub=df[(df.dataset.astype(str)==ds)&(df.method.astype(str)==m)]; x=100*sub.delta_unknown_detection_rate_vs_max_confidence.astype(float); y=np.array([fmap[str(f)]+offs[m] for f in sub.holdout_family])
            ax.scatter(x,y,marker=markers[m],s=30,facecolors="black" if m=="family_conditional" else "white",edgecolors="black",lw=.8,label=labels[m])
        ax.axvline(0,color="0.35",lw=.8); ax.set(yticks=np.arange(len(fams)),yticklabels=fams,xlim=(xmin,xmax),xlabel="Δ held-out UDR vs max-confidence (percentage points)"); clean(ax,"x"); panel(ax,f"({'ab'[i]}) {ds}")
    h,l=axes[1].get_legend_handles_labels(); uniq=dict(zip(l,h)); fig.legend(uniq.values(),uniq.keys(),loc="lower center",ncol=3,frameon=False,bbox_to_anchor=(.5,.005))
    fig.tight_layout(rect=(0,.11,1,1),w_pad=2); save(fig,"figure10_rejector_delta_udr")


def figure11() -> None:
    df=read("fig11_validation_test_rejection.csv"); markers={"max_confidence":"o","margin":"s","entropy":"^","family_conditional":"D"}; labels={"max_confidence":"Max-confidence","margin":"Margin","entropy":"Entropy","family_conditional":"Family-conditional"}
    fig,ax=plt.subplots(figsize=(6.5,4.2))
    for m,mk in markers.items():
        sub=df[df.method.astype(str)==m]
        for ds,face in (("CICIDS2017","white"),("CICIoT2023","black")):
            s=sub[sub.dataset.astype(str)==ds]
            if not s.empty: ax.scatter(s.validation_overall_reject_rate,s.overall_reject_rate,marker=mk,s=30,facecolors=face,edgecolors="black",lw=.8)
    maxv=max(.16,float(df.overall_reject_rate.max())*1.08); ax.plot([0,.04],[0,.04],ls=":",color="0.55",lw=.8); ax.axvline(.03,ls="--",color="0.35",lw=.8); ax.axhline(.03,ls="--",color="0.35",lw=.8)
    ax.set(xlim=(0,.035),ylim=(0,maxv),xlabel="Validation rejection rate",ylabel="Test rejection rate"); clean(ax,"both")
    o=df.loc[df.overall_reject_rate.astype(float).idxmax()]; ax.annotate(f"{o.dataset} {o.holdout_family}\n{labels.get(str(o.method),o.method)}",(o.validation_overall_reject_rate,o.overall_reject_rate),xytext=(-78,-4),textcoords="offset points",fontsize=7.1,arrowprops={"arrowstyle":"-","lw":.6,"color":"0.35"})
    mh=[Line2D([0],[0],marker=mk,ls="none",mfc="white",mec="black",label=labels[m]) for m,mk in markers.items()]; dh=[Line2D([0],[0],marker="o",ls="none",mfc=f,mec="black",label=ds) for ds,f in (("CICIDS2017","white"),("CICIoT2023","black"))]
    leg=ax.legend(handles=mh,loc="upper left",bbox_to_anchor=(1.02,1),frameon=False,borderaxespad=0); ax.add_artist(leg); ax.legend(handles=dh,loc="upper left",bbox_to_anchor=(1.02,.58),frameon=False,borderaxespad=0)
    fig.tight_layout(rect=(0,0,.77,1)); save(fig,"figure11_validation_test_rejection")


def main() -> None:
    setup()
    if not (SRC / "main_figure_source_manifest.json").exists():
        raise FileNotFoundError("Missing main figure source manifest; run scripts/28_build_jisa_main_figure_sources.py first")
    for fn in (figure2,figure3,figure4,figure5,figure6,figure7,figure8,figure9,figure10,figure11): fn()
    payload={"status":"JISA_MAIN_NUMERICAL_FIGURES_RENDERED","figure_numbers":list(range(2,12)),"formats":["svg","pdf","png"],"source_dir":str(SRC.relative_to(ROOT)),"output_dir":str(OUT.relative_to(ROOT)),"figure_1":"handled separately"}
    (OUT/"main_figure_render_summary.json").write_text(json.dumps(payload,indent=2),encoding="utf-8"); print(json.dumps(payload,indent=2))


if __name__ == "__main__": main()
