"""Step 7: case study on pyrrolizidine alkaloids (PAs).

PAs are plant toxins found mainly in herbs, spices, teas and herbal infusions. EU maximum
levels were set by Commission Regulation (EU) 2020/2040, applicable from 1 July 2022, with
products lawfully placed on the market before that date allowed until 31 December 2023.

Questions:
  1. Where do PA notifications come from (products, origins) and who notifies them?
  2. Which origins and notifying countries are over-represented, given their overall
     volume of notifications (hypergeometric test, Benjamini-Hochberg)?
  3. Are PA notifications classified as serious more or less often than other hazards
     in the same product category?
  4. Did the rate change after the end of the transition period (2023 vs 2024-2026), and by year?

Notifications are counted once (a notification with two origin countries counts once
for each origin in the origin table only).

Run:  python step7_pyrrolizidine_alkaloids.py
"""
import numpy as np
import pandas as pd
from scipy.stats import hypergeom, fisher_exact, binomtest
from config import CLEAN_FILE, RESULTS

PA = "pyrrolizidine alkaloids"
MIN_PA = 5          # minimum PA notifications for a country to be tested


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def over_representation(notif, col, is_pa, universe_mask=None):
    """For each value of `col`, is its share of PA notifications larger than its share of all
    notifications in the universe? Hypergeometric upper tail."""
    u = notif if universe_mask is None else notif[universe_mask]
    pa = u[is_pa.loc[u.index]]
    N, K = len(u), len(pa)
    rows = []
    for v, k in pa[col].value_counts().items():
        if k < MIN_PA:
            continue
        n = (u[col] == v).sum()
        rows.append({col: v, "pa_notifications": k, "all_notifications": n,
                     "expected": K * n / N, "ratio": k / (K * n / N), "p": hypergeom.sf(k - 1, N, n, K)})
    out = pd.DataFrame(rows)
    out["q"] = bh(out["p"])
    return out.sort_values("ratio", ascending=False)


def main():
    df = pd.read_csv(CLEAN_FILE, parse_dates=["date"])
    df["is_pa"] = df["hazard_name"].str.lower().str.strip() == PA
    notif = df.groupby("notification_id").agg(date=("date", "first"), year=("year", "first"),
                                              product=("product_category", "first"), notifying=("notifying", "first"),
                                              risk=("risk", "first"), is_pa=("is_pa", "max"))
    pa = notif[notif["is_pa"]]
    print(f"PA notifications: {len(pa)} of {len(notif)} food notifications with a coded hazard")

    # 1. composition
    print("\nBy product category:"); print(pa["product"].value_counts().to_string())
    origins = df[df["is_pa"]].drop_duplicates(["notification_id", "origin"])
    print("\nBy origin (top 8):"); print(origins["origin"].value_counts().head(8).to_string())
    print("\nBy notifying country (top 8):"); print(pa["notifying"].value_counts().head(8).to_string())
    pa.groupby(["year", "product"]).size().unstack(fill_value=0).to_csv(RESULTS / "pa_by_year_product.csv")

    # 2. over-representation
    is_pa = notif["is_pa"]
    notif_rep = over_representation(notif, "notifying", is_pa)
    print("\nNotifying countries, PA share vs all notifications:")
    print(notif_rep.round(4).to_string(index=False))
    notif_rep.to_csv(RESULTS / "pa_notifying_overrepresentation.csv", index=False)
    o = df.drop_duplicates(["notification_id", "origin"]).set_index("notification_id")
    o_is_pa = o["is_pa"] | o.index.isin(pa.index)
    o = o.assign(is_pa=o_is_pa).reset_index()
    herbs = o["product_category"] == "herbs and spices"
    orig_rep = over_representation(o, "origin", o["is_pa"], universe_mask=herbs)
    print("\nOrigins, within herbs and spices: PA share vs all herbs-and-spices notifications:")
    print(orig_rep.round(4).to_string(index=False))
    orig_rep.to_csv(RESULTS / "pa_origin_overrepresentation_herbs.csv", index=False)

    # 3. risk decisions, within the same product categories
    rows = []
    for prod in ["herbs and spices", "cocoa and cocoa preparations, coffee and tea"]:
        sub = notif[notif["product"] == prod]
        a = sub[sub["is_pa"]]; b = sub[~sub["is_pa"]]
        s_a, s_b = (a["risk"] == "serious").sum(), (b["risk"] == "serious").sum()
        odds, p = fisher_exact([[s_a, len(a) - s_a], [s_b, len(b) - s_b]])
        rows.append({"product": prod, "pa_n": len(a), "pa_serious_share": s_a / len(a),
                     "other_n": len(b), "other_serious_share": s_b / len(b), "odds_ratio": odds, "p": p})
    risk = pd.DataFrame(rows)
    print("\nShare classified as serious, PA vs other hazards in the same product category:")
    print(risk.round(3).to_string(index=False))
    risk.to_csv(RESULTS / "pa_risk_decisions.csv", index=False)

    # 4. monthly rate after the end of the transition period
    last = notif["date"].max()
    months_2023 = 12
    months_after = (last.year - 2024) * 12 + last.month - 1 + last.day / last.days_in_month
    n_2023 = (pa["year"] == 2023).sum(); n_after = (pa["year"] >= 2024).sum()
    all_2023 = (notif["year"] == 2023).sum(); all_after = (notif["year"] >= 2024).sum()
    # conditional test: given the total PA count, is the split 2023 vs later proportional to
    # all food notifications? (controls for changes in overall notification volume)
    test = binomtest(n_2023, n_2023 + n_after, all_2023 / (all_2023 + all_after))
    rate = pd.DataFrame([{"period": "2023 (transition)", "pa": n_2023, "months": months_2023, "pa_per_month": n_2023 / months_2023,
                          "pa_share_of_all": n_2023 / all_2023},
                         {"period": f"2024 to {last:%d %b %Y}", "pa": n_after, "months": round(months_after, 1),
                          "pa_per_month": n_after / months_after, "pa_share_of_all": n_after / all_after}])
    print("\nMonthly rate before and after the end of the transition period:")
    print(rate.round(4).to_string(index=False))
    print(f"Share of PA among all notifications, 2023 vs later: binomial test p = {test.pvalue:.3f}")
    yr = pd.DataFrame({"pa": pa.groupby("year").size(), "all": notif.groupby("year").size()})
    yr["months"] = [12.0] * (len(yr) - 1) + [round(last.month - 1 + last.day / last.days_in_month, 1)]
    yr["pa_per_month"] = yr["pa"] / yr["months"]; yr["pa_share_of_all"] = yr["pa"] / yr["all"]
    print("By year:"); print(yr.round(4).to_string())
    yr.to_csv(RESULTS / "pa_by_year.csv")
    k26, n26 = yr.loc[2026, "pa"], yr.loc[2026, "all"]
    p_before = yr.loc[:2025, "pa"].sum() / yr.loc[:2025, "all"].sum()
    t26 = binomtest(int(k26), int(n26), p_before, alternative="less")
    print(f"2026 vs 2023-2025: PA share {k26 / n26:.4f} vs {p_before:.4f}, one-sided binomial p = {t26.pvalue:.2g}")
    rate.to_csv(RESULTS / "pa_rate_transition.csv", index=False)


if __name__ == "__main__":
    main()
