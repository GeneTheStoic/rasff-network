"""Step 6: does the way hazards are categorized change what we can predict?

Inspired by Stevens, Almeida et al. (2025, Clinical Nutrition), who compare diet
categorizations for flare prediction with the sparse group lasso, this step compares
hazard categorizations for predicting which EU food notifications receive a "serious"
risk decision.

Outcome: risk decision "serious" (vs. any other decision) for each notification.
Predictors: product category and origin country, plus hazards at one of two levels:
  - class level: 28 hazard classes;
  - name level: individual hazards (at least 20 notifications in 2023-2025), grouped by class.
Models (logistic, penalized):
  M1 class level, lasso;  M2 name level, lasso;  M3 name level, group lasso (groups = hazard
  classes; product and origin indicators are singleton groups);  M4 name level, sparse group lasso (alpha = 0.5), which selects classes and,
  inside each selected class, the hazards that matter.
Tuning: penalty chosen on 2025 (models fitted on 2023-2024). Final models fitted on
2023-2025 and evaluated out of sample on 2026: AUC, log loss, and a paired bootstrap
interval for the AUC difference between models.

Run:  python step6_hazard_categorization.py
"""
import re
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from config import CLEAN_FILE, RESULTS, SEED
import sgl

MIN_COUNT = 20
N_ORIGINS = 25
N_LAMBDA = 20
N_BOOT = 1000


SPELLING = {"chlorpyriphos-ethyl": "chlorpyrifos", "chlorpyrifos-ethyl": "chlorpyrifos",
            "aflatoxins b1": "aflatoxin b1", "e.coli": "escherichia coli", "salmonellen": "salmonella", "salmonella spp": "salmonella"}


def norm_name(s):
    """Spelling-only normalization: case, punctuation, and known spelling variants."""
    s = re.sub(r"\s+", " ", str(s).lower().replace("spp.", "spp")).strip(" .,")
    for a, b in SPELLING.items():
        s = re.sub(rf"^{re.escape(a)}\b", b, s)
    return s


def auc(y, p):
    r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def logloss(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def build(df):
    df = df.copy()
    df["hname"] = df["hazard_name"].map(norm_name)
    notif = df.groupby("notification_id").agg(year=("year", "first"), risk=("risk", "first"),
                                              product=("product_category", "first"), origin=("origin", "first"))
    notif["y"] = (notif["risk"] == "serious").astype(int)
    train_mask = notif["year"] <= 2025
    top_origins = notif.loc[train_mask, "origin"].value_counts().head(N_ORIGINS).index
    notif["origin_g"] = np.where(notif["origin"].isin(top_origins), notif["origin"], "other")
    P = pd.get_dummies(notif["product"], prefix="product").astype(float)
    O = pd.get_dummies(notif["origin_g"], prefix="origin").astype(float)
    C = pd.crosstab(df["notification_id"], df["hazard_class"]).clip(upper=1).reindex(notif.index, fill_value=0).astype(float)
    C.columns = [f"class_{c}" for c in C.columns]
    counts = df[df["year"] <= 2025].drop_duplicates(["notification_id", "hname"])["hname"].value_counts()
    keep = counts[counts >= MIN_COUNT].index
    hn = df[df["hname"].isin(keep)]
    N = pd.crosstab(hn["notification_id"], hn["hname"]).clip(upper=1).reindex(notif.index, fill_value=0).astype(float)
    name_class = hn.drop_duplicates("hname").set_index("hname")["hazard_class"]
    # residual "other hazards in class" indicator so that rare hazards are not lost
    other = df[~df["hname"].isin(keep)]
    R = pd.crosstab(other["notification_id"], other["hazard_class"]).clip(upper=1).reindex(notif.index, fill_value=0).astype(float)
    R.columns = [f"other_{c}" for c in R.columns]
    name_cols = list(N.columns)
    N.columns = [f"name_{c}" for c in N.columns]
    # product and origin indicators are singleton groups (penalized as in the lasso);
    # hazards are grouped by hazard class
    groups_names = list(P.columns) + list(O.columns) + [name_class[c] for c in name_cols] + [c.replace("other_", "") for c in R.columns]
    X_class = pd.concat([P, O, C], axis=1)
    X_name = pd.concat([P, O, N, R], axis=1)
    return notif, X_class, X_name, np.array(groups_names)


def tune_and_fit(X, y, years, groups, alpha):
    tr, va = years <= 2024, years == 2025
    lmax = sgl.lambda_max(X[tr], y[tr], groups, alpha)
    grid = lmax * np.logspace(0, -3, N_LAMBDA)
    best, beta = None, None
    for lam in grid:
        beta = sgl.fit(X[tr], y[tr], groups, lam, alpha, b0=beta)
        ll = logloss(y[va], sgl.predict(beta, X[va]))
        if best is None or ll < best[0]:
            best = (ll, lam)
    full = years <= 2025
    return sgl.fit(X[full], y[full], groups, best[1], alpha), best[1]


def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(CLEAN_FILE)
    notif, Xc, Xn, groups_n = build(df)
    y = notif["y"].values; years = notif["year"].values; te = years == 2026
    models = {
        "M1 class level, lasso": (Xc, np.arange(Xc.shape[1]), 1.0),
        "M2 name level, lasso": (Xn, np.arange(Xn.shape[1]), 1.0),
        "M3 name level, group lasso": (Xn, groups_n, 0.0),
        "M4 name level, sparse group lasso": (Xn, groups_n, 0.5),
    }
    rows, preds, betas = [], {}, {}
    for name, (X, groups, alpha) in models.items():
        Xv = X.values
        beta, lam = tune_and_fit(Xv, y, years, groups, alpha)
        p = sgl.predict(beta, Xv[te]); preds[name] = p; betas[name] = (beta, X.columns, groups)
        nz = np.abs(beta[1:]) > 1e-8
        hazard_cols = [i for i, c in enumerate(X.columns) if c.startswith(("class_", "name_", "other_"))]
        rows.append({"model": name, "features": X.shape[1], "selected": int(nz.sum()),
                     "hazard_features_selected": int(nz[hazard_cols].sum()),
                     "test_auc": auc(y[te], p), "test_logloss": logloss(y[te], p)})
        print(f"  {name}: AUC {rows[-1]['test_auc']:.3f}, selected {rows[-1]['selected']} of {X.shape[1]}", flush=True)
    res = pd.DataFrame(rows)
    # paired bootstrap of AUC differences against M1, on the 2026 test set
    yt = y[te]; n = len(yt); diffs = {k: [] for k in preds if not k.startswith("M1")}
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if yt[i].min() == yt[i].max():
            continue
        a1 = auc(yt[i], preds["M1 class level, lasso"][i])
        for k in diffs:
            diffs[k].append(auc(yt[i], preds[k][i]) - a1)
    res["auc_diff_vs_M1"] = [np.nan] + [np.mean(diffs[k]) for k in diffs]
    res["diff_ci05"] = [np.nan] + [np.quantile(diffs[k], 0.05) for k in diffs]
    res["diff_ci95"] = [np.nan] + [np.quantile(diffs[k], 0.95) for k in diffs]
    res.to_csv(RESULTS / "hazard_categorization_models.csv", index=False)
    print("\nOut-of-sample comparison (trained 2023-2025, tested on 2026):")
    print(res.round(3).to_string(index=False))
    # coefficients of the sparse group lasso, by hazard class
    beta, cols, groups = betas["M4 name level, sparse group lasso"]
    coef = pd.DataFrame({"feature": cols, "group": groups, "coef": beta[1:]})
    coef.to_csv(RESULTS / "sgl_coefficients.csv", index=False)
    hz = coef[coef["feature"].str.startswith(("name_", "other_"))]
    by_group = hz.assign(nonzero=hz["coef"].abs() > 1e-8).groupby("group").agg(hazards=("feature", "size"), selected=("nonzero", "sum"))
    print("\nSparse group lasso: hazard classes and hazards selected inside each class:")
    print(by_group[by_group["selected"] > 0].sort_values("selected", ascending=False).to_string())
    for g in ["pathogenic micro-organisms", "mycotoxins", "pesticide residues", "heavy metals"]:
        sub = hz[(hz["group"] == g) & (hz["coef"].abs() > 1e-8)].sort_values("coef", ascending=False)
        print(f"\n{g}:"); print(sub[["feature", "coef"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
