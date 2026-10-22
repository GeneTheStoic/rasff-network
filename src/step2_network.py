"""Step 2: build a two-layer network and compute centrality.

Layer 1 (trade/origin):  origin country -- product category
Layer 2 (hazard):        product category -- hazard
Edge weight = number of notifications linking the two nodes.
Product categories are the nodes shared by both layers.

Run:  python src/step2_network.py
"""
import networkx as nx
import pandas as pd
from config import CLEAN_FILE, RESULTS, TOP_K


def build_layers(df: pd.DataFrame):
    """Return (G_origin_product, G_product_hazard, G_combined)."""
    def weighted_graph(pairs, kind_a, kind_b):
        g = nx.Graph()
        counts = pairs.value_counts()
        for (a, b), w in counts.items():
            g.add_node(f"{kind_a}:{a}", kind=kind_a)
            g.add_node(f"{kind_b}:{b}", kind=kind_b)
            g.add_edge(f"{kind_a}:{a}", f"{kind_b}:{b}", weight=int(w))
        return g

    l1 = df.drop_duplicates(["notification_id", "origin"])   # each notification counts once per origin
    l2 = df.drop_duplicates(["notification_id", "hazard"])   # each notification counts once per hazard
    g1 = weighted_graph(l1[["origin", "product_category"]].apply(tuple, axis=1), "origin", "product")
    g2 = weighted_graph(l2[["product_category", "hazard"]].apply(tuple, axis=1), "product", "hazard")
    combined = nx.compose(g1, g2)                              # shared product nodes connect the layers
    return g1, g2, combined


def centrality_table(g: nx.Graph) -> pd.DataFrame:
    strength = dict(g.degree(weight="weight"))
    pagerank = nx.pagerank(g, weight="weight")
    tab = pd.DataFrame({"node": list(g.nodes)})
    tab["kind"] = tab["node"].map(nx.get_node_attributes(g, "kind"))
    tab["name"] = tab["node"].str.split(":", n=1).str[1]
    tab["strength"] = tab["node"].map(strength)
    tab["pagerank"] = tab["node"].map(pagerank)
    return tab.sort_values("pagerank", ascending=False)


def product_communities(g2: nx.Graph, seed=42) -> pd.DataFrame:
    """Project the product-hazard layer onto products (products linked if they share hazards) and find communities."""
    products = [n for n, d in g2.nodes(data=True) if d["kind"] == "product"]
    proj = nx.bipartite.weighted_projected_graph(g2, products)
    comms = nx.community.louvain_communities(proj, weight="weight", seed=seed)
    rows = [{"product": p.split(":", 1)[1], "community": i} for i, c in enumerate(comms) for p in c]
    return pd.DataFrame(rows).sort_values(["community", "product"])


def main():
    df = pd.read_csv(CLEAN_FILE, parse_dates=["date"])
    g1, g2, g = build_layers(df)
    print(f"Layer origin-product: {g1.number_of_nodes()} nodes, {g1.number_of_edges()} edges")
    print(f"Layer product-hazard: {g2.number_of_nodes()} nodes, {g2.number_of_edges()} edges")
    print(f"Combined network:     {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

    RESULTS.mkdir(exist_ok=True)
    tab = centrality_table(g)
    tab.to_csv(RESULTS / "centrality_combined.csv", index=False)
    for kind in ["origin", "product", "hazard"]:
        print(f"\nTop {TOP_K} {kind} nodes by PageRank:")
        print(tab[tab["kind"] == kind].head(TOP_K)[["name", "strength", "pagerank"]].to_string(index=False))

    comm = product_communities(g2)
    comm.to_csv(RESULTS / "product_communities.csv", index=False)
    print(f"\n{comm['community'].nunique()} product communities found (see results/product_communities.csv)")


if __name__ == "__main__":
    main()
