from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA, OUT = ROOT / "data", ROOT / "models"
OUT.mkdir(exist_ok=True)


def trend(df, product_col, date_col, price_col, min_days=4):
    """Tốc độ thay đổi giá mỗi tháng, hồi quy có hiệu ứng cố định theo sản phẩm.
    Sai số chuẩn tính theo cụm sản phẩm."""
    d = df[[product_col, date_col, price_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d = d[d[price_col] > 0]
    # giá trung vị của từng sản phẩm theo từng ngày
    d = d.groupby([product_col, date_col], as_index=False)[price_col].median()
    d = d[d.groupby(product_col)[date_col].transform("nunique") >= min_days]
    if d.empty:
        raise ValueError("Không có sản phẩm nào xuất hiện đủ số ngày tối thiểu.")

    d["y"] = np.log(d[price_col])
    d["t"] = (d[date_col] - d[date_col].min()).dt.days.astype(float)

    g = d.groupby(product_col)
    d["y_dm"] = d["y"] - g["y"].transform("mean")
    d["t_dm"] = d["t"] - g["t"].transform("mean")

    sxx = float((d.t_dm ** 2).sum())
    slope = float((d.t_dm * d.y_dm).sum() / sxx)
    d["res"] = d.y_dm - slope * d.t_dm
    scores = (d.t_dm * d.res).groupby(d[product_col]).sum()
    se = float(np.sqrt((scores ** 2).sum()) / sxx)

    f = lambda s: float(np.expm1(s * 30))
    return {
        "monthly_change": f(slope),
        "monthly_ci_low": f(slope - 1.96 * se),
        "monthly_ci_high": f(slope + 1.96 * se),
        "n_products": int(d[product_col].nunique()),
        "n_days": int(d[date_col].nunique()),
        "start": str(d[date_col].min().date()),
        "end": str(d[date_col].max().date()),
    }


d1 = pd.read_csv(DATA / "ram-price-observations.csv")
d3 = pd.read_csv(DATA / "costinflation-ddr5-ram-retail-prices-raw-2026-07-13-to-2026-08-10.csv")
d3 = d3[d3.normalized_price_amount.notna()]

result = {}
for key, args in {
    "D1": (d1, "product_id", "capture_date_utc", "price_usd"),
    "D3": (d3, "product_name", "observed_date", "normalized_price_amount"),
}.items():
    try:
        result[key] = trend(*args)
    except Exception as e:
        print(f"Bỏ qua {key}: {e}")

if not result:
    raise RuntimeError("Không đo được xu hướng từ nguồn nào.")

(OUT / "market_trend.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
for k, v in result.items():
    print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()})
