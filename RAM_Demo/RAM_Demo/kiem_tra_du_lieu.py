from pathlib import Path
import re
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
b = pd.read_csv(DATA / "ram_pricing_intelligence_2026.csv")
b["capacity_gb"] = pd.to_numeric(b["capacity_gb"], errors="coerce")

# 1) Với tin kit nhiều thanh (vd "2x 16GB"), capacity_gb là cả bộ hay mỗi thanh?
pat = re.compile(r"(\d+)\s*[xX]\s*(\d+)\s*gb", re.I)
kinds = []
for title, cap in zip(b["title"], b["capacity_gb"]):
    m = pat.search(str(title))
    if m and pd.notna(cap) and int(m.group(1)) > 1:
        n, per = int(m.group(1)), int(m.group(2))
        if cap == n * per:
            kinds.append("capacity = cả bộ (n x mỗi thanh)")
        elif cap == per:
            kinds.append("capacity = mỗi thanh")
        else:
            kinds.append("khác")
print("=== 1) Ý nghĩa capacity_gb với tin kit nhiều thanh ===")
print(pd.Series(kinds).value_counts().to_string())

# 2) bus_speed_mhz thiếu: bao nhiêu dòng trích được từ tiêu đề?
def speed_from_title(t):
    t = str(t)
    m = re.search(r"\b(\d{4,5})\s*(?:MHz|MT/s)\b", t, re.I) or re.search(r"DDR[2-5]\s*-?\s*(\d{3,5})\b", t, re.I)
    return float(m.group(1)) if m else np.nan

miss = b[pd.to_numeric(b["bus_speed_mhz"], errors="coerce").isna()]
filled = miss["title"].apply(speed_from_title)
print("\n=== 2) bus_speed_mhz ===")
print(f"Thiếu: {len(miss)} dòng; trích được từ tiêu đề: {int(filled.notna().sum())} dòng")
print(miss.assign(speed_title=filled)[["title", "speed_title"]].head(8).to_string())

# 3) D1 và D3: số dòng so với số sản phẩm duy nhất
c = pd.read_csv(DATA / "costinflation-ddr5-ram-retail-prices-raw-2026-07-13-to-2026-08-10.csv")
a = pd.read_csv(DATA / "ram-price-observations.csv")
print("\n=== 3) Số dòng và số sản phẩm duy nhất ===")
print(f"D1: {len(c)} dòng, {c['product_name'].nunique()} tên sản phẩm duy nhất")
print(f"D3: {len(a)} dòng, {a['title'].nunique()} tiêu đề duy nhất, dung lượng kit: {sorted(a['kit_capacity_gb'].unique())}")

# 4) wasPrice
print("\n=== 4) wasPrice ===")
print(f"Có wasPrice: {int(b['wasPrice'].notna().sum())}/{len(b)} dòng")