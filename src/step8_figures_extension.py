"""Step 8: figures for steps 6 and 7.

Figure 4: hazard categorization and prediction of serious risk decisions (step 6).
Figure 5: pyrrolizidine alkaloid case study (step 7).

Run:  python step8_figures_extension.py   (after steps 6 and 7)
"""
import matplotlib.pyplot as plt
import pandas as pd
from config import RESULTS, FIGURES


def fig_categorization():
    m = pd.read_csv(RESULTS / "hazard_categorization_models.csv")
    c = pd.read_csv(RESULTS / "sgl_coefficients.csv")
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1, 1.25]})
    labels = ["Class level\nlasso", "Name level\nlasso", "Name level\ngroup lasso", "Name level\nsparse group lasso"]
    ax[0].bar(labels, m["test_auc"], color=["#999999", "#4C72B0", "#4C72B0", "#C44E52"])
    for i, r in m.iterrows():
        ax[0].text(i, r["test_auc"] + 0.004, f"{r['test_auc']:.3f}", ha="center", fontsize=9)
    ax[0].set_ylim(0.5, 0.85); ax[0].set_ylabel("AUC on 2026 notifications (out of sample)")
    ax[0].set_title("Finer hazard categories predict serious decisions better")
    ax[0].tick_params(axis="x", labelsize=8)
    sel = c[c["group"].isin(["pathogenic micro-organisms", "mycotoxins"]) & c["feature"].str.startswith("name_")].copy()
    sel = sel.sort_values(["group", "coef"])
    sel["label"] = sel["feature"].str.replace("name_", "", regex=False)
    col = sel["group"].map({"pathogenic micro-organisms": "#DD8452", "mycotoxins": "#55A868"})
    ax[1].barh(sel["label"], sel["coef"], color=col)
    ax[1].axvline(0, color="black", lw=0.8)
    ax[1].set_xlabel("Sparse group lasso coefficient (log-odds of a serious decision)")
    ax[1].set_title("Within a hazard class, hazards differ (orange: pathogens, green: mycotoxins)")
    ax[1].tick_params(axis="y", labelsize=8); ax[1].title.set_fontsize(10)
    plt.tight_layout(); plt.savefig(FIGURES / "fig4_hazard_categorization.png", dpi=150); plt.close()


def fig_pa():
    yp = pd.read_csv(RESULTS / "pa_by_year_product.csv", index_col=0)
    yr = pd.read_csv(RESULTS / "pa_by_year.csv", index_col=0)
    nr = pd.read_csv(RESULTS / "pa_notifying_overrepresentation.csv").sort_values("ratio")
    main = ["herbs and spices", "cocoa and cocoa preparations, coffee and tea"]
    yp["other"] = yp.drop(columns=[c for c in main if c in yp]).sum(axis=1)
    yp = yp[[c for c in main if c in yp] + ["other"]].div(yr["months"], axis=0)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    yp.rename(columns={main[1]: "cocoa, coffee and tea"}).plot.bar(stacked=True, ax=ax[0], color=["#55A868", "#8172B2", "#BBBBBB"])
    ax[0].set_ylabel("PA notifications per month"); ax[0].set_xlabel("")
    ax[0].set_title("Pyrrolizidine alkaloid notifications per month (2026: to 21 October)"); ax[0].tick_params(axis="x", rotation=0)
    ax[0].legend(frameon=False, fontsize=8)
    colors = ["#C44E52" if q < 0.05 else "#BBBBBB" for q in nr["q"]]
    ax[1].barh(nr["notifying"], nr["ratio"], color=colors)
    ax[1].axvline(1, color="black", lw=0.8)
    ax[1].set_xlabel("PA notifications / expected from the country's overall notifications")
    ax[1].set_title("Which countries notify PAs more than their volume implies (red: q < 0.05)")
    ax[1].title.set_fontsize(10)
    plt.tight_layout(); plt.savefig(FIGURES / "fig5_pyrrolizidine_alkaloids.png", dpi=150); plt.close()


if __name__ == "__main__":
    fig_categorization(); fig_pa()
    print("Saved figures/fig4_hazard_categorization.png and figures/fig5_pyrrolizidine_alkaloids.png")
