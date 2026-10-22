"""Step 1: load the RASFF Window export(s), clean them and save one tidy table.

Output: one row per (notification, origin country, hazard), with the product category.

Run:  python step1_load_clean.py
"""
import re
import pandas as pd
import config as c


def read_raw() -> pd.DataFrame:
    files = sorted(list(c.RAW_DIR.glob("*.xlsx")) + list(c.RAW_DIR.glob("*.csv")))
    if not files:
        raise FileNotFoundError(f"No .xlsx or .csv files found in {c.RAW_DIR}")
    frames = [pd.read_excel(f) if f.suffix == ".xlsx" else pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=[c.COL_REFERENCE])
    print(f"Loaded {len(df):,} unique notifications from {len(files)} file(s).")
    return df


def tidy(text):
    if pd.isna(text):
        return None
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text or None


def parse_hazards(cell):
    """'Aflatoxin B1 - {mycotoxins},aflatoxin total - {mycotoxins}' -> [('aflatoxin b1','mycotoxins'), ...]"""
    if pd.isna(cell):
        return []
    out = []
    for item in re.split(r"(?<=\}),", str(cell)):
        m = re.match(r"\s*(.*?)\s*-\s*\{(.*?)\}\s*$", item)
        if m:
            name, cls = tidy(m.group(1)), tidy(m.group(2))
            out.append(((name or cls).lower(), cls.lower()))
    return out


def main():
    raw = read_raw()
    df = raw[raw[c.COL_TYPE].isin(c.KEEP_TYPES)].copy()
    print(f"Kept {len(df):,} notifications of type {c.KEEP_TYPES}.")

    df["date"] = pd.to_datetime(df[c.COL_DATE], dayfirst=True, errors="coerce")
    df["year"] = df["date"].dt.year
    df["product_category"] = df[c.COL_CATEGORY].map(tidy).str.lower()
    df["notifying"] = df[c.COL_NOTIFYING].map(tidy)
    df["risk"] = df[c.COL_RISK].map(tidy)
    df["origin"] = df[c.COL_ORIGIN].map(tidy).str.split(",")
    df["hazard_pairs"] = df[c.COL_HAZARDS].map(parse_hazards)

    no_hazard = (df["hazard_pairs"].str.len() == 0).sum()
    df = df[df["hazard_pairs"].str.len() > 0]
    print(f"Dropped {no_hazard:,} notifications without a coded hazard (e.g. labelling-only cases).")

    df = df.explode("origin").explode("hazard_pairs")
    df["origin"] = df["origin"].str.strip()
    df["hazard_name"] = df["hazard_pairs"].str[0]
    df["hazard_class"] = df["hazard_pairs"].str[1]
    df["hazard"] = df[c.HAZARD_LEVEL]
    df = df.dropna(subset=["origin", "product_category", "hazard"])

    df["notification_id"] = df[c.COL_REFERENCE].astype(str)
    cols = ["notification_id", "date", "year", "origin", "notifying", "product_category",
            "hazard", "hazard_name", "hazard_class", "risk"]
    df = df[cols].drop_duplicates()
    c.CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(c.CLEAN_FILE, index=False)

    print(f"\nSaved {len(df):,} rows covering {df['notification_id'].nunique():,} notifications "
          f"({df['date'].min():%d/%m/%Y} to {df['date'].max():%d/%m/%Y}) to {c.CLEAN_FILE.name}")
    print("Distinct origin countries:", df["origin"].nunique(),
          "| product categories:", df["product_category"].nunique(),
          "| hazard classes:", df["hazard_class"].nunique(),
          "| hazard names:", df["hazard_name"].nunique())
    print(f"Hazard level used for the network: {c.HAZARD_LEVEL}")


if __name__ == "__main__":
    main()
