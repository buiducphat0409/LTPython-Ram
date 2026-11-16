from pathlib import Path
import json, re, warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV, GroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "models"
OUT.mkdir(exist_ok=True)
RANDOM_STATE = 42
ALPHA = 0.10  # mức bao phủ mục tiêu = 90%
MAX_ROWS_FOR_SVR = 6000  # SVR RBF chậm khi nhiều dòng (giống notebook của bạn Dương)
warnings.filterwarnings("ignore")

# Nhóm dung lượng dùng cho độ chính xác theo nhóm
CAP_BINS = [0, 8, 16, 32, 64, np.inf]
CAP_LABELS = ["≤8 GB", "9-16 GB", "17-32 GB", "33-64 GB", ">64 GB"]


def require(name):
    p = DATA / name
    if not p.exists():
        raise FileNotFoundError(f"Thiếu {p}. Xem README.md để biết danh sách CSV cần cung cấp.")
    return pd.read_csv(p)


def save_model(key, model, features, defaults, target, metrics, extra=None):
    joblib.dump(model, OUT / f"{key}.joblib")
    meta = {"name": key, "features": list(features), "defaults": defaults,
            "target": target, "metrics": metrics}
    if extra:
        meta.update(extra)
    (OUT / f"{key}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=float), encoding="utf-8")


def metrics(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ape = np.abs(y - pred) / np.maximum(np.abs(y), 1e-9)  # sai số tương đối từng mẫu
    return {"MAE": float(mean_absolute_error(y, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
            "R2": float(r2_score(y, pred)),
            "MAPE": float(np.mean(ape) * 100),
            "within_10": float(np.mean(ape <= 0.10) * 100),
            "within_20": float(np.mean(ape <= 0.20) * 100)}


def conformal_info(y_cal, pred_cal, y_test, pred_test, alpha=ALPHA):
    """Split Conformal Prediction với absolute residual.
    alpha=0.10 tương ứng mức bao phủ mục tiêu 90%.
    Residual/q có cùng đơn vị với y (USD/bộ hoặc USD/GB)."""
    y_cal = np.asarray(y_cal, dtype=float)
    pred_cal = np.asarray(pred_cal, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    pred_test = np.asarray(pred_test, dtype=float)

    residuals = np.abs(y_cal - pred_cal)
    n = len(residuals)
    if n == 0:
        raise ValueError("Calibration set đang rỗng.")

    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    q = float(np.quantile(residuals, level, method="higher"))

    lower, upper = pred_test - q, pred_test + q
    coverage = float(np.mean((y_test >= lower) & (y_test <= upper)))
    return {"interval_alpha": float(alpha), "interval_level": float(1 - alpha),
            "conformal_q": q, "calibration_n": int(n),
            "test_coverage": coverage, "test_n": int(len(y_test))}


def segment_stats(y, pred, gen, cap, titles=None):
    """Độ chính xác theo nhóm {thế hệ x nhóm dung lượng}, đo trên tập TEST.
    y, pred: giá thật / giá dự đoán (cùng đơn vị); gen: thế hệ RAM; cap: dung lượng dùng để chia nhóm.
    titles: tiêu đề sản phẩm của các mẫu test, để đếm số sản phẩm duy nhất trong nhóm (n_products)."""
    grp = pd.cut(pd.Series(np.asarray(cap, dtype=float)), bins=CAP_BINS, labels=CAP_LABELS)
    data = {
        "y": np.asarray(y, dtype=float),
        "p": np.asarray(pred, dtype=float),
        "gen": np.asarray(gen, dtype=object),
        "grp": grp.astype(object).values,
    }
    if titles is not None:
        data["title"] = (pd.Series(titles).astype(str).str.lower()
                         .str.replace(r"\s+", " ", regex=True).values)
    df = pd.DataFrame(data).dropna(subset=["gen", "grp"])
    df["ape"] = np.abs(df.y - df.p) / np.maximum(np.abs(df.y), 1e-9)
    out = []
    for (g, c), part in df.groupby(["gen", "grp"]):
        mape = float(part.ape.mean() * 100)
        seg = {"generation": str(g), "cap_group": str(c), "n": int(len(part)),
               "MAPE": mape, "accuracy": max(0.0, 100 - mape),
               "within_10": float((part.ape <= 0.10).mean() * 100),
               "within_20": float((part.ape <= 0.20).mean() * 100)}
        if titles is not None:
            seg["n_products"] = int(part.title.nunique())
        out.append(seg)
    return out


def split_3way_random(X, y, seed=RANDOM_STATE):
    """60% train / 20% calibration / 20% test (ngẫu nhiên)."""
    X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.20, random_state=seed)
    X_tr, X_cal, y_tr, y_cal = train_test_split(X_tmp, y_tmp, test_size=0.25, random_state=seed)
    return X_tr, X_cal, X_te, y_tr, y_cal, y_te


def split_3way_group(X, y, groups, seed=RANDOM_STATE):
    """60/20/20 nhưng không để cùng một nhóm (tên sản phẩm) xuất hiện ở 2 phần."""
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    i_trcal, i_te = next(gss1.split(X, y, groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    i_tr_rel, i_cal_rel = next(gss2.split(X.iloc[i_trcal], y.iloc[i_trcal], groups.iloc[i_trcal]))
    return i_trcal[i_tr_rel], i_trcal[i_cal_rel], i_te


def make_defaults(X, cols):
    return {c: (X[c].median() if pd.api.types.is_numeric_dtype(X[c]) else X[c].mode().iloc[0]) for c in cols}


# ---- Số thanh và tổng dung lượng của D2 -------------------------------------------
# Đã kiểm chứng trên D2: với 862/862 tin kit nhiều thanh dạng "NxYGB" (vd "2x 16GB"),
# capacity_gb là dung lượng CẢ BỘ (không có tin nào là mỗi thanh) -> KHÔNG nhân thêm số thanh.
# Tin dạng "lot of N" / "kit of N" (chưa kiểm chứng) vẫn nhân với N.
KIT_COUNT_PAT = r"(\d+)\s*[xX]\s*\d+\s*gb"
KIT_FLAG_PAT = r"\d+\s*[xX]\s*\d+\s*gb"


def count_sticks(title):
    t = str(title).lower()
    for pat in (KIT_COUNT_PAT, r"lot of\s*(\d+)", r"kit of\s*(\d+)"):
        m = re.search(pat, t)
        if m:
            return int(m.group(1))
    return 1


def kit_total_capacity(frame):
    """Trả về (sticks_count, total_capacity_gb) cho dataframe có cột title, capacity_gb."""
    cap = pd.to_numeric(frame["capacity_gb"], errors="coerce")
    sticks = frame["title"].apply(count_sticks)
    has_kit = frame["title"].astype(str).str.contains(KIT_FLAG_PAT, case=False, regex=True)
    total = np.where(has_kit, cap, cap * sticks)
    return sticks, total


def add_stick_features(frame):
    """Thêm sticks_count và total_capacity_gb từ tiêu đề. Dùng cho XGBoost."""
    d = frame.copy()
    d["sticks_count"], d["total_capacity_gb"] = kit_total_capacity(d)
    return d


# Dataset D2 dùng chung cho Linear và XGBoost
main = require("ram_pricing_intelligence_2026.csv")

# ------------------------------------------- 1) Linear Regression (cải tiến)
# Notebook: Linear Regression cơ sở -> cải tiến bằng Ridge (L2) + log-target + feature engineering

def extract_lr_features(frame):
    d = frame.copy()
    d["sticks_count"], d["total_capacity_gb"] = kit_total_capacity(d)
    d["cas_latency"] = d.title.apply(lambda t: float(m.group(1)) if (m := re.search(r"\bcl\s*(\d{2})\b", str(t).lower())) else np.nan)
    low = d.title.astype(str).str.lower()
    d["is_rgb"] = low.str.contains(r"rgb|argb").astype(int)
    d["is_ecc_rdimm"] = low.str.contains(r"ecc|rdimm|lrdimm|registered").astype(int)
    d["is_laptop"] = low.str.contains(r"sodimm|so-dimm|laptop").astype(int)
    d["bandwidth_index"] = d.total_capacity_gb * d.bus_speed_mhz / 1000.0
    d["log_capacity"] = np.log2(np.maximum(d.total_capacity_gb, 1))
    d["log_bus_speed"] = np.log2(np.maximum(d.bus_speed_mhz, 100))
    d["cap_speed_ratio"] = d.total_capacity_gb / np.maximum(d.bus_speed_mhz, 100)
    return d

lr_data = main.drop_duplicates().copy()
for c in ["price", "capacity_gb", "bus_speed_mhz", "is_bulk_server"]:
    lr_data[c] = pd.to_numeric(lr_data[c], errors="coerce")
lr_data = lr_data.dropna(subset=["price", "capacity_gb"])
lr_data = lr_data[(lr_data.price > 0) & (lr_data.capacity_gb > 0)]
lr_data = extract_lr_features(lr_data)
unit_p = lr_data.price / lr_data.total_capacity_gb
q99 = lr_data.price.quantile(.99)
lr_data = lr_data[(unit_p >= .3) & (unit_p <= 55.0) & (lr_data.price <= q99)].copy()
lr_num = ["total_capacity_gb", "bus_speed_mhz", "sticks_count", "cas_latency", "bandwidth_index", "log_capacity", "log_bus_speed", "cap_speed_ratio", "is_rgb", "is_ecc_rdimm", "is_laptop", "is_bulk_server"]
lr_cat = ["ram_generation", "brand", "condition", "unit_type", "type"]
lr_features = lr_num + lr_cat
Xl, yl = lr_data[lr_features], lr_data.price
Xl_tr, Xl_cal, Xl_te, yl_tr, yl_cal, yl_te = split_3way_random(Xl, yl)
lr_pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), lr_num),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), lr_cat)
])
lr_model = TransformedTargetRegressor(regressor=Pipeline([("prep", lr_pre), ("model", Ridge(alpha=1.0))]), func=np.log1p, inverse_func=np.expm1)
lr_model.fit(Xl_tr, yl_tr)
p_cal, p_te = lr_model.predict(Xl_cal), lr_model.predict(Xl_te)
save_model("linear_regression", lr_model, lr_features, make_defaults(Xl, lr_features), "price", metrics(yl_te, p_te),
           extra={**conformal_info(yl_cal, p_cal, yl_te, p_te), "dataset": "D1 + D2 (Ridge cải tiến) + D3",
                  "segments": segment_stats(yl_te, p_te, Xl_te["ram_generation"], Xl_te["total_capacity_gb"],
                                            titles=lr_data.loc[Xl_te.index, "title"]),
                  "segment_cap_col": "total_capacity_gb"})
print("[1/5] Linear Regression (cải tiến: Ridge + log-target) exported")

# ------------------------------------------ 2 & 3) Random Forest + Gradient Boosting
# Bộ dữ liệu gộp D1+D2+D3 (chỉ DDR5, không bulk), target usd_per_gb.
# SVR (mục 4) dùng lại đúng bộ dữ liệu này.
# Lưu ý: capacity_gb ở cả ba nguồn đều là dung lượng CẢ BỘ (D2 đã kiểm chứng),
# nên usd_per_gb = price / capacity_gb của nguồn B là đúng.
a = require("ram-price-observations.csv")
b = require("ram_pricing_intelligence_2026.csv")
c = require("costinflation-ddr5-ram-retail-prices-raw-2026-07-13-to-2026-08-10.csv")
df_a = pd.DataFrame({"source":"A_retail", "title":a.title, "capacity_gb":a.kit_capacity_gb,
                     "price":a.price_usd, "usd_per_gb":a.usd_per_gb, "condition":"New", "speed_hint":np.nan})
b5 = b[(b.ram_generation == "DDR5") & (b.is_bulk_server == 0) & (b.unit_type != "Bulk")].copy().dropna(subset=["capacity_gb", "price"])
df_b = pd.DataFrame({"source":"B_ebay", "title":b5.title, "capacity_gb":b5.capacity_gb,
                     "price":b5.price, "usd_per_gb":b5.price / b5.capacity_gb, "condition":b5.condition, "speed_hint":b5.bus_speed_mhz})
c = c[(c.quantity_value > 0) & c.normalized_price_amount.notna()]
df_c = pd.DataFrame({"source":"C_costinflation", "title":c.product_name, "capacity_gb":c.quantity_value,
                     "price":c.price_amount, "usd_per_gb":c.normalized_price_amount, "condition":"New", "speed_hint":np.nan})
df = pd.concat([df_a, df_b, df_c], ignore_index=True)
BRANDS = ["Corsair", "G.Skill", "Kingston", "Crucial", "Micron", "TEAMGROUP", "T-Force", "Patriot", "Samsung", "SK hynix", "A-Tech", "Acer", "Biwin", "ADATA", "XPG", "Lexar", "PNY", "OLOy", "Mushkin", "Silicon Power", "Netac", "Thermaltake", "Gigabyte", "ASUS", "MSI"]
def find_brand(t):
    z = str(t).lower().replace(" ","").replace("-","").replace(".","")
    for br in BRANDS:
        if br.lower().replace(" ","").replace("-","").replace(".","") in z: return br
    if "hynix" in z: return "SK hynix"
    if "gskill" in z or "tridentz" in z: return "G.Skill"
    return "Other"
def find_speed(row):
    if pd.notna(row.speed_hint): return row.speed_hint
    t = str(row.title)
    m = re.search(r"\b([3-9]\d{3})\s*(?:MHz|MT/s|MTs)?\b", t, re.I)
    if m: return float(m.group(1))
    m = re.search(r"B(\d{4})C\d{2}", t)
    return float(m.group(1)) if m else np.nan
def find_cl(t):
    t = str(t); m = re.search(r"\bCL\s?-?(\d{2})\b", t, re.I)
    if m: return float(m.group(1))
    m = re.search(r"\b(\d{2})-\d{2}-\d{2}-\d{2,3}\b", t)
    if m: return float(m.group(1))
    m = re.search(r"B\d{4}C(\d{2})", t)
    return float(m.group(1)) if m else np.nan
def find_sticks(t):
    m = re.search(r"\(?\b(\d)\s*[xX]\s*\d+\s*GB", str(t))
    return int(m.group(1)) if m else 1
df["brand"] = df.title.apply(find_brand)
df["speed_mhz"] = df.apply(find_speed, axis=1)
df["cas_latency"] = df.title.apply(find_cl)
df["num_sticks"] = df.title.apply(find_sticks)
df["is_sodimm"] = df.title.str.contains(r"SO-?DIMM|laptop|notebook", case=False, regex=True).astype(int)
df["has_rgb"] = df.title.str.contains("RGB", case=False).astype(int)
df["is_ecc_server"] = df.title.str.contains(r"ECC|RDIMM|registered|server", case=False, regex=True).astype(int)
df = df[~df.condition.isin(["For parts or not working"])]
qlo, qhi = df.usd_per_gb.quantile([.005, .995])
df = df[df.usd_per_gb.between(qlo, qhi)].reset_index(drop=True)
rf_num = ["capacity_gb", "speed_mhz", "cas_latency", "num_sticks", "is_sodimm", "has_rgb", "is_ecc_server"]
rf_cat = ["brand", "source", "condition"]
rf_features = rf_num + rf_cat


def total_capacity(X):
    """Tổng dung lượng của bộ = capacity_gb (cả ba nguồn đều là dung lượng cả bộ)."""
    return np.asarray(X["capacity_gb"], dtype=float)


Xr, yr = df[rf_features], df.usd_per_gb
groups = df.title.str.lower().str.replace(r"\s+", " ", regex=True)
itr, ical, ite = split_3way_group(Xr, yr, groups)
Xr_tr, Xr_cal, Xr_te = Xr.iloc[itr], Xr.iloc[ical], Xr.iloc[ite]
yr_tr, yr_cal, yr_te = yr.iloc[itr], yr.iloc[ical], yr.iloc[ite]
groups_tr = groups.iloc[itr]

tot_te = total_capacity(Xr_te)
gen_te = ["DDR5"] * len(Xr_te)  # RF/GB chỉ huấn luyện trên DDR5
titles_te = df.title.iloc[ite]

def make_tree_pre():
    return ColumnTransformer([("num", SimpleImputer(strategy="median", add_indicator=True), rf_num),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), rf_cat)])
rf_base = Pipeline([("prep", make_tree_pre()), ("rf", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))])
rf_params = {"rf__n_estimators":[200,300,500], "rf__max_depth":[None,10,20,30], "rf__min_samples_split":[2,5,10], "rf__min_samples_leaf":[1,2,4], "rf__max_features":["sqrt",.5,1.0]}
rf_search = RandomizedSearchCV(rf_base, rf_params, n_iter=15, cv=GroupKFold(n_splits=3), scoring="neg_mean_absolute_error", random_state=42, n_jobs=-1)
rf_search.fit(Xr_tr, yr_tr, groups=groups_tr)
rf = rf_search.best_estimator_
p_cal, p_te = rf.predict(Xr_cal), rf.predict(Xr_te)
save_model("random_forest", rf, rf_features, make_defaults(Xr, rf_features), "usd_per_gb", metrics(yr_te, p_te),
           extra={**conformal_info(yr_cal, p_cal, yr_te, p_te), "dataset": "D1+D2+D3",
                  "segments": segment_stats(yr_te, p_te, gen_te, tot_te, titles=titles_te),
                  "segment_cap_col": "__total__"})
print("[2/5] Random Forest exported")

gb_base = Pipeline([("prep", make_tree_pre()), ("gbr", GradientBoostingRegressor(n_estimators=300, learning_rate=.1, max_depth=3, random_state=42))])
gb_params = {"gbr__n_estimators":[200,300,500], "gbr__learning_rate":[.03,.05,.1], "gbr__max_depth":[3,4,5,6], "gbr__min_samples_leaf":[1,5,10], "gbr__subsample":[.7,.85,1.0], "gbr__max_features":[None,"sqrt",.5], "gbr__loss":["squared_error","huber"]}
gb_search = RandomizedSearchCV(gb_base, gb_params, n_iter=15, cv=GroupKFold(n_splits=3), scoring="neg_mean_absolute_error", random_state=42, n_jobs=-1)
gb_search.fit(Xr_tr, yr_tr, groups=groups_tr)
gb = gb_search.best_estimator_
p_cal, p_te = gb.predict(Xr_cal), gb.predict(Xr_te)
save_model("gradient_boosting", gb, rf_features, make_defaults(Xr, rf_features), "usd_per_gb", metrics(yr_te, p_te),
           extra={**conformal_info(yr_cal, p_cal, yr_te, p_te), "dataset": "D1+D2+D3",
                  "segments": segment_stats(yr_te, p_te, gen_te, tot_te, titles=titles_te),
                  "segment_cap_col": "__total__"})
print("[3/5] Gradient Boosting exported")

# ---------------------------------------------------------------- 4) SVR
# Bản cập nhật của bạn Dương (svr_ram_3_data_duong.ipynb): D1+D2+D3, target usd_per_gb,
# đặc trưng/tham số giữ nguyên; giới hạn 6000 dòng lấy mẫu theo tỷ lệ nguồn.
# Khác notebook: demo chia 60/20/20 (thêm tập calibration cho khoảng dự đoán 90%).
svr_num = ["capacity_gb", "speed_mhz", "cas_latency", "num_sticks", "is_sodimm", "has_rgb", "is_ecc_server"]
svr_cat = ["brand", "source", "condition"]
svr_features = svr_num + svr_cat

df_svr = df.copy()
df_svr["capacity_gb"] = pd.to_numeric(df_svr["capacity_gb"], errors="coerce")
df_svr["usd_per_gb"] = pd.to_numeric(df_svr["usd_per_gb"], errors="coerce")
df_svr = df_svr.dropna(subset=["capacity_gb", "usd_per_gb"])
df_svr = df_svr[(df_svr.capacity_gb > 0) & (df_svr.usd_per_gb > 0)]
if len(df_svr) > MAX_ROWS_FOR_SVR:
    parts = []
    for _, part in df_svr.groupby("source"):
        n = max(1, round(MAX_ROWS_FOR_SVR * len(part) / len(df_svr)))
        parts.append(part.sample(n=min(n, len(part)), random_state=RANDOM_STATE))
    df_svr = pd.concat(parts, ignore_index=True)
    if len(df_svr) > MAX_ROWS_FOR_SVR:
        df_svr = df_svr.sample(MAX_ROWS_FOR_SVR, random_state=RANDOM_STATE)
df_svr = df_svr.reset_index(drop=True)

Xs, ys = df_svr[svr_features], df_svr.usd_per_gb
groups_s = df_svr.title.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
s_tr, s_cal, s_te = split_3way_group(Xs, ys, groups_s)
Xs_tr, Xs_cal, Xs_te = Xs.iloc[s_tr], Xs.iloc[s_cal], Xs.iloc[s_te]
ys_tr, ys_cal, ys_te = ys.iloc[s_tr], ys.iloc[s_cal], ys.iloc[s_te]

svr_pre = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scaler", StandardScaler())]), svr_num),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), svr_cat)
])
svr_pipe = Pipeline([("preprocessor", svr_pre), ("svr", SVR(kernel="rbf", C=100, epsilon=.1, gamma="scale"))])
svr = TransformedTargetRegressor(regressor=svr_pipe, func=np.log1p, inverse_func=np.expm1)
svr.fit(Xs_tr, ys_tr)
p_cal, p_te = np.maximum(svr.predict(Xs_cal), 0), np.maximum(svr.predict(Xs_te), 0)
save_model("svr", svr, svr_features, make_defaults(Xs, svr_features), "usd_per_gb", metrics(ys_te, p_te),
           extra={**conformal_info(ys_cal, p_cal, ys_te, p_te), "dataset": "D1+D2+D3",
                  "segments": segment_stats(ys_te, p_te, ["DDR5"] * len(Xs_te), total_capacity(Xs_te),
                                            titles=df_svr.title.iloc[s_te]),
                  "segment_cap_col": "__total__"})
print(f"[4/5] SVR exported (D1+D2+D3, {len(df_svr)} dòng, train {len(Xs_tr)} / cal {len(Xs_cal)} / test {len(Xs_te)})")

# ---------------------------------------------------------------- 5) XGBoost
xg = main.drop_duplicates().copy()
for col in ["capacity_gb", "bus_speed_mhz", "price", "wasPrice", "available", "sold", "is_bulk_server"]:
    if col in xg.columns: xg[col] = pd.to_numeric(xg[col], errors="coerce")
xg = xg[xg.price.notna() & (xg.price > 0)].copy()
xg = add_stick_features(xg)  # thêm sticks_count, total_capacity_gb (trước khi bỏ cột title)
drop_cols = ["price", "title", "seller_id", "mpn", "itemLocation", "lastUpdated"]
Xx = xg.drop(columns=[c for c in drop_cols if c in xg.columns])
yx = xg.price
Xx = Xx.loc[:, Xx.nunique(dropna=False) > 1]
num = Xx.select_dtypes(include=["number", "bool"]).columns.tolist()
cat = Xx.select_dtypes(exclude=["number", "bool"]).columns.tolist()
Xx_tr, Xx_cal, Xx_te, yx_tr, yx_cal, yx_te = split_3way_random(Xx, yx)
xg_pre = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat)
])
xg_model = Pipeline([("preprocessor", xg_pre), ("regressor", XGBRegressor(n_estimators=500, learning_rate=.04, max_depth=5, min_child_weight=2, subsample=.85, colsample_bytree=.85, objective="reg:squarederror", eval_metric="rmse", random_state=42, n_jobs=-1))])
xg_model.fit(Xx_tr, np.log1p(yx_tr))
# Residual conformal tính SAU khi đưa về USD bằng expm1
p_cal = np.expm1(xg_model.predict(Xx_cal))
p_te = np.expm1(xg_model.predict(Xx_te))
save_model("xgboost", xg_model, list(Xx.columns), make_defaults(Xx, Xx.columns), "price", metrics(yx_te, p_te),
           extra={**conformal_info(yx_cal, p_cal, yx_te, p_te), "dataset": "D2", "inverse_log1p": True,
                  "segments": segment_stats(yx_te, p_te, Xx_te["ram_generation"], Xx_te["total_capacity_gb"],
                                            titles=xg.loc[Xx_te.index, "title"]),
                  "segment_cap_col": "total_capacity_gb"})
print("[5/5] XGBoost exported")
print("Hoàn tất: models/*.joblib và models/*.json")