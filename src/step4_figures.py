"""Step 4: two figures for the README.

Figure 1: the two-layer network (top nodes only, for readability).
Figure 2: top products and hazards with bootstrap 90% intervals.

Run:  python src/step4_figures.py
"""
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from config import CLEAN_FILE, RESULTS, FIGURES, TOP_K
from step2_network import build_layers

COLORS = {"origin": "#4C72B0", "product": "#DD8452", "hazard": "#C44E52"}


def figure_network(df, n_per_kind=12):
    _, _, g = build_layers(df)
    tab = pd.read_csv(RESULTS / "centrality_combined.csv")
    keep = pd.concat([tab[tab["kind"] == k].head(n_per_kind) for k in COLORS])["node"]
    h = g.subgraph(keep)
    x = {"origin": 0, "product": 1, "hazard": 2}
    pos = {}
    for k in COLORS:
        nodes = [n for n in keep if n in h and h.nodes[n]["kind"] == k]   # ordered by PageRank (top = most central)
        for i, n in enumerate(nodes):
            pos[n] = (x[k], -i)
    widths = [0.3 + 3 * d["weight"] / max(1, max(dd["weight"] for *_, dd in h.edges(data=True))) for *_, d in h.edges(data=True)]
    plt.figure(figsize=(14, 8))
    nx.draw_networkx_edges(h, pos, width=widths, alpha=0.35)
    for k, c in COLORS.items():
        nodes = [n for n in h if h.nodes[n]["kind"] == k]
        nx.draw_networkx_nodes(h, pos, nodelist=nodes, node_color=c, node_size=120, label=k)
    nx.draw_networkx_labels(h, {n: (p[0] + 0.03, p[1]) for n, p in pos.items()},
                            labels={n: n.split(":", 1)[1][:40] for n in h}, font_size=7, horizontalalignment="left")
    plt.title("Origin country – product category – hazard network (top nodes, EU RASFF notifications)")
    plt.legend(loc="lower center", ncol=3, frameon=False)
    plt.xlim(-0.1, 2.75)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIGURES / "fig1_network.png", dpi=200)
    plt.close()


def figure_intervals():
    s = pd.read_csv(RESULTS / "centrality_bootstrap.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, kind in zip(axes, ["product", "hazard"]):
        t = s[s["kind"] == kind].head(TOP_K).iloc[::-1]
        ax.errorbar(t["pagerank_median"], range(len(t)),
                    xerr=[t["pagerank_median"] - t["pagerank_p05"], t["pagerank_p95"] - t["pagerank_median"]],
                    fmt="o", color=COLORS[kind], capsize=3)
        ax.set_yticks(range(len(t)))
        ax.set_yticklabels([n[:48] for n in t["name"]], fontsize=8)
        ax.set_xlabel("PageRank (median and 90% bootstrap interval)")
        ax.set_title(f"Top {TOP_K} {kind}s")
    plt.tight_layout()
    plt.savefig(FIGURES / "fig2_centrality_uncertainty.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(CLEAN_FILE, parse_dates=["date"])
    figure_network(df)
    figure_intervals()
    print("Saved figures/fig1_network.png and figures/fig2_centrality_uncertainty.png")
