"""Step 5: which links are stronger than notification volume alone would predict?

Raw edge counts mix two things: genuine product-specific risk and sheer volume
(how often an origin or a product is notified overall, which also reflects trade
and inspection intensity). This step separates them with a null model and then
checks the result out of sample.

(a) Pairwise layers (origin-product, product-hazard). Null model: the two margins
    are independent. For each link, a hypergeometric test asks whether the observed
    count exceeds what the margins imply (statistically validated networks,
    Tumminello et al., 2011, PLoS ONE 6(3): e17994).
(b) Coupling between layers (origin-product-hazard). Null model: given the product,
    origin and hazard are independent, so the layers interact only through the
    shared product node. A triple that rejects this null is an origin-specific
    hazard profile that neither layer shows on its own.
(c) Multiple testing: Benjamini-Hochberg false discovery rate across all links.
(d) Out-of-sample validation: links are validated on 2023-2025 and then checked
    on 2026 notifications, with expected counts recomputed from 2026 margins.

Run:  python src/step5_validated_links.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from config import CLEAN_FILE, RESULTS, FIGURES

FDR = 0.01          # false discovery rate for validation
TRAIN_END = 2025    # validate on years <= TRAIN_END
TEST_YEAR = 2026    # check out of sample on this year


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def pair_table(d: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Observed and expected counts for each (a, b) link; unit = notification."""
    x = d.drop_duplicates(["notification_id", a, b])[[a, b]]
    t = x.value_counts().rename("n").reset_index()
    t["n_a"] = t[a].map(x[a].value_counts())
    t["n_b"] = t[b].map(x[b].value_counts())
    total = len(x)
    t["expected"] = t["n_a"] * t["n_b"] / total
    t["p"] = hypergeom.sf(t["n"] - 1, total, t["n_a"], t["n_b"])
    return t


def triple_table(d: pd.DataFrame) -> pd.DataFrame:
    """Origin-product-hazard counts against conditional independence given the product."""
    x = d.drop_duplicates(["notification_id", "origin", "hazard_class"])[["origin", "product_category", "hazard_class"]]
    t = x.value_counts().rename("n").reset_index()
    op = x.groupby(["origin", "product_category"]).size().rename("n_op")
    ph = x.groupby(["product_category", "hazard_class"]).size().rename("n_ph")
    pp = x.groupby("product_category").size().rename("n_p")
    t = t.join(op, on=["origin", "product_category"]).join(ph, on=["product_category", "hazard_class"]).join(pp, on="product_category")
    t["expected"] = t["n_op"] * t["n_ph"] / t["n_p"]
    t["p"] = hypergeom.sf(t["n"] - 1, t["n_p"], t["n_op"], t["n_ph"])
    return t


def out_of_sample(train: pd.DataFrame, test: pd.DataFrame, keys: list) -> pd.DataFrame:
    """Attach test-year observed and expected counts to the training links."""
    test = test.set_index(keys)[["n", "expected"]].rename(columns={"n": "n_test", "expected": "exp_test_obs"})
    return train.join(test, on=keys)


def expected_in_test(train: pd.DataFrame, test_raw: pd.DataFrame, kind: str, keys: list) -> pd.Series:
    """Expected test-year count under the null, including links absent from the test year."""
    if kind == "triple":
        x = test_raw.drop_duplicates(["notification_id", "origin", "hazard_class"])
        op = x.groupby(["origin", "product_category"]).size()
        ph = x.groupby(["product_category", "hazard_class"]).size()
        pp = x.groupby("product_category").size()
        e = [op.get((o, p), 0) * ph.get((p, h), 0) / pp.get(p, np.inf)
             for o, p, h in zip(train["origin"], train["product_category"], train["hazard_class"])]
        return pd.Series(e, index=train.index)
    a, b = keys
    x = test_raw.drop_duplicates(["notification_id", a, b])
    na, nb, total = x[a].value_counts(), x[b].value_counts(), len(x)
    return train[a].map(na).fillna(0) * train[b].map(nb).fillna(0) / total


def run(df: pd.DataFrame, kind: str, keys: list) -> tuple:
    train_raw, test_raw = df[df["year"] <= TRAIN_END], df[df["year"] == TEST_YEAR]
    if kind == "triple":
        train, test = triple_table(train_raw), triple_table(test_raw)
    else:
        train, test = pair_table(train_raw, *keys), pair_table(test_raw, *keys)
    train["q"] = bh(train["p"].values)
    train["validated"] = train["q"] < FDR
    train["ratio"] = train["n"] / train["expected"]
    train = out_of_sample(train, test, keys)
    train["n_test"] = train["n_test"].fillna(0)
    train["exp_test"] = expected_in_test(train, test_raw, kind, keys)

    rows = []
    for label, mask in [("validated", train["validated"]), ("not validated", ~train["validated"])]:
        g = train[mask & (train["exp_test"] > 0)]
        rows.append({"layer": kind, "group": label, "links": int(mask.sum()),
                     "share_over_expected_in_test": round((g["n_test"] > g["exp_test"]).mean(), 3),
                     "observed_over_expected_in_test": round(g["n_test"].sum() / g["exp_test"].sum(), 2)})
    return train.sort_values(["validated", "n"], ascending=[False, False]), pd.DataFrame(rows)


def figure(summary: pd.DataFrame):
    labels = {"origin-product": "Origin–product", "product-hazard": "Product–hazard", "triple": "Origin–product–hazard\n(layer coupling)"}
    fig, ax = plt.subplots(figsize=(9, 4.5))
    layers = list(labels)
    xpos = np.arange(len(layers))
    for i, (grp, color) in enumerate([("validated", "#C44E52"), ("not validated", "#8C8C8C")]):
        vals = [summary[(summary.layer == l) & (summary.group == grp)]["observed_over_expected_in_test"].iloc[0] for l in layers]
        bars = ax.bar(xpos + (i - 0.5) * 0.38, vals, 0.38, color=color, label=f"{grp} on {2023}–{TRAIN_END}")
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.axhline(1, color="black", lw=0.8, ls="--")
    ax.set_xticks(xpos, [labels[l] for l in layers])
    ax.set_ylabel(f"Observed / expected in {TEST_YEAR} (out of sample)")
    ax.set_title(f"Links validated on {2023}–{TRAIN_END} stay over-represented in {TEST_YEAR}")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURES / "fig3_validated_links_out_of_sample.png", dpi=200)
    plt.close()


def main():
    df = pd.read_csv(CLEAN_FILE, parse_dates=["date"])
    specs = [("origin-product", ["origin", "product_category"]),
             ("product-hazard", ["product_category", "hazard_class"]),
             ("triple", ["origin", "product_category", "hazard_class"])]
    summaries = []
    for kind, keys in specs:
        table, summary = run(df, kind, keys)
        table.to_csv(RESULTS / f"validated_links_{kind}.csv", index=False)
        summaries.append(summary)
        v = table[table["validated"]]
        print(f"\n{kind}: {len(table)} links, {len(v)} validated at FDR {FDR}")
        print(summary.to_string(index=False))
        cols = keys + ["n", "expected", "ratio"]
        print("Largest raw links that are NOT validated (explained by volume):")
        top = table.sort_values("n", ascending=False).head(15)
        print(top[~top["validated"]][cols].round(2).to_string(index=False))
        print("Strongest validated links (n >= 20, by observed/expected):")
        print(v[v["n"] >= 20].sort_values("ratio", ascending=False).head(8)[cols].round(1).to_string(index=False))
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(RESULTS / "validated_links_out_of_sample.csv", index=False)
    FIGURES.mkdir(exist_ok=True)
    figure(summary)
    print("\nSaved results/validated_links_*.csv and figures/fig3_validated_links_out_of_sample.png")


if __name__ == "__main__":
    main()
