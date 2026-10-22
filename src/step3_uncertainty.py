"""Step 3: how stable are the centrality rankings?

(a) Bootstrap: resample notifications with replacement, rebuild the network, recompute PageRank.
    Report a 90% interval for each node and how often it appears in the top K.
(b) Temporal stability: Spearman correlation of PageRank rankings between consecutive years.

Run:  python src/step3_uncertainty.py
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from config import CLEAN_FILE, RESULTS, N_BOOTSTRAP, SEED, TOP_K
from step2_network import build_layers, centrality_table


def bootstrap(df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.reset_index(drop=True)
    groups = df.groupby("notification_id").indices          # notification -> row positions
    ids = np.array(list(groups.keys()))
    records = []
    for b in range(n_boot):
        sample_ids = rng.choice(ids, size=len(ids), replace=True)
        rows = np.concatenate([groups[i] for i in sample_ids])
        new_ids = np.repeat(np.arange(len(sample_ids)), [len(groups[i]) for i in sample_ids])
        # repeated notifications get new ids so they count as separate draws
        sample = df.iloc[rows].assign(notification_id=new_ids)
        _, _, g = build_layers(sample)
        tab = centrality_table(g)
        tab["rank_within_kind"] = tab.groupby("kind")["pagerank"].rank(ascending=False, method="min")
        tab["boot"] = b
        records.append(tab[["node", "kind", "name", "pagerank", "rank_within_kind", "boot"]])
        if (b + 1) % 100 == 0:
            print(f"  bootstrap {b + 1}/{n_boot}")
    return pd.concat(records, ignore_index=True)


def summarise(boot: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    s = boot.groupby(["node", "kind", "name"]).agg(
        pagerank_median=("pagerank", "median"),
        pagerank_p05=("pagerank", lambda x: np.quantile(x, 0.05)),
        pagerank_p95=("pagerank", lambda x: np.quantile(x, 0.95)),
        share_in_top_k=("rank_within_kind", lambda r: (r <= TOP_K).sum() / n_boot),
    ).reset_index()
    return s.sort_values(["kind", "pagerank_median"], ascending=[True, False])


def temporal_stability(df: pd.DataFrame) -> pd.DataFrame:
    years = sorted(df["year"].dropna().unique())
    ranks = {}
    for y in years:
        _, _, g = build_layers(df[df["year"] == y])
        ranks[y] = centrality_table(g).set_index("node")["pagerank"]
    rows = []
    for y0, y1 in zip(years[:-1], years[1:]):
        common = ranks[y0].index.intersection(ranks[y1].index)
        rho, _ = spearmanr(ranks[y0][common], ranks[y1][common])
        rows.append({"from": int(y0), "to": int(y1), "common_nodes": len(common), "spearman_rho": round(rho, 3)})
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(CLEAN_FILE, parse_dates=["date"])
    print(f"Running {N_BOOTSTRAP} bootstrap replications...")
    boot = bootstrap(df, N_BOOTSTRAP, SEED)
    summary = summarise(boot, N_BOOTSTRAP)
    summary.to_csv(RESULTS / "centrality_bootstrap.csv", index=False)
    for kind in ["origin", "product", "hazard"]:
        print(f"\n{kind.upper()}: median PageRank [90% interval], share of replications in top {TOP_K}")
        print(summary[summary["kind"] == kind].head(TOP_K)
              [["name", "pagerank_median", "pagerank_p05", "pagerank_p95", "share_in_top_k"]].round(4).to_string(index=False))

    if "year" in df and df["year"].nunique() > 1:
        ts = temporal_stability(df)
        ts.to_csv(RESULTS / "temporal_stability.csv", index=False)
        print("\nRank correlation between consecutive years:\n", ts.to_string(index=False))


if __name__ == "__main__":
    main()
