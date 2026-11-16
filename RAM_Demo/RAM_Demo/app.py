from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"

st.set_page_config(page_title="Dự đoán giá RAM", page_icon="🖥️", layout="wide")
st.title("Dự đoán giá RAM bằng 5 thuật toán học máy")
st.caption("Demo nghiên cứu — kết quả phụ thuộc dữ liệu huấn luyện, không phải báo giá thị trường trực tiếp.")

LR_NAME = "Linear Regression (cải tiến: Ridge + log-target)"
XGB_NAME = "XGBoost"
MODEL_FILES = {
    LR_NAME: "linear_regression.joblib",
    "Random Forest": "random_forest.joblib",
    "Gradient Boosting": "gradient_boosting.joblib",
    "Support Vector Regression (SVR)": "svr.joblib",
    XGB_NAME: "xgboost.joblib",
}

@st.cache_resource
def load_artifacts():
    models, meta = {}, {}
    for name, filename in MODEL_FILES.items():
        p = MODEL_DIR / filename
        mp = MODEL_DIR / (p.stem + ".json")
        if p.exists() and mp.exists():
            models[name] = joblib.load(p)
            meta[name] = json.loads(mp.read_text(encoding="utf-8"))
    return models, meta

models, metadata = load_artifacts()


def cap_label(x):
    """Phải khớp với CAP_BINS / CAP_LABELS trong train_models.py."""
    for hi, lab in [(8, "≤8 GB"), (16, "9-16 GB"), (32, "17-32 GB"), (64, "33-64 GB")]:
        if x <= hi:
            return lab
    return ">64 GB"


def lookup_segment(info, generation, cap_value, min_n=20):
    """Tìm nhóm (thế hệ x dung lượng) khớp đầu vào; trả None nếu không có hoặc n < min_n."""
    if cap_value is None or pd.isna(cap_value):
        return None
    for s in info.get("segments", []):
        if s["generation"] == generation and s["cap_group"] == cap_label(float(cap_value)):
            return s if s["n"] >= min_n else None
    return None


with st.sidebar:
    st.header("Thông tin đầu vào")
    st.caption("Nhập thông số của một bộ RAM. Các trường nâng cao chỉ dùng khi mô hình cần thêm thuộc tính listing.")
    with st.form("ram_form"):
        brand = st.selectbox("Thương hiệu", ["Kingston", "Corsair", "G.Skill", "Crucial", "Samsung", "SK hynix", "Micron", "TEAMGROUP", "ADATA", "Other"])
        generation = st.selectbox("Thế hệ RAM", ["DDR5", "DDR4", "DDR3", "DDR2", "Khác"])
        capacity = st.number_input("Tổng dung lượng (GB)", min_value=1.0, max_value=4096.0, value=32.0, step=1.0)
        speed = st.number_input("Tốc độ bus (MHz)", min_value=100.0, max_value=20000.0, value=5600.0, step=100.0)
        sticks = st.number_input("Số thanh RAM", min_value=1, max_value=64, value=2, step=1)
        cl = st.number_input("CAS Latency (CL)", min_value=0.0, max_value=200.0, value=36.0, step=1.0)
        condition = st.selectbox("Tình trạng", ["New", "Used", "Refurbished", "Open box"])
        unit_type = st.selectbox("Đơn vị bán", ["Kit", "Single", "Bulk", "Other"])
        ram_type = st.selectbox("Loại RAM", ["Desktop", "Laptop", "Server", "Other"])
        is_bulk = st.selectbox("Tin bán hàng loạt/server bulk?", [0, 1], format_func=lambda x: "Không" if x == 0 else "Có")
        is_sodimm = st.selectbox("RAM laptop SO-DIMM?", [0, 1], format_func=lambda x: "Không" if x == 0 else "Có")
        has_rgb = st.selectbox("Có RGB?", [0, 1], format_func=lambda x: "Không" if x == 0 else "Có")
        is_ecc = st.selectbox("ECC/RDIMM/server?", [0, 1], format_func=lambda x: "Không" if x == 0 else "Có")
        source = st.selectbox("Nguồn dữ liệu tham chiếu (RF/GB/SVR)", ["A_retail", "B_ebay", "C_costinflation"])
        st.markdown("**Thuộc tính listing bổ sung (XGBoost)**")
        st.caption("Nhập 0 = không biết → dùng giá trị trung vị lúc huấn luyện.")
        was_price = st.number_input("Giá niêm yết cũ (USD, 0 = không có)", min_value=0.0, value=0.0, step=10.0)
        available = st.number_input("Số lượng còn hàng (0 = không biết)", min_value=0.0, value=0.0, step=1.0)
        sold = st.number_input("Số lượng đã bán (0 = không biết)", min_value=0.0, value=0.0, step=1.0)
        submitted = st.form_submit_button("Dự đoán giá RAM", type="primary", use_container_width=True)

if not models:
    st.warning("Chưa có mô hình đã huấn luyện. Hãy đặt các CSV dữ liệu vào thư mục data/ và chạy train_models.py theo hướng dẫn README.")
    st.code("python train_models.py", language="bash")
    st.stop()

if len(models) < 5:
    st.warning(f"Mới tìm thấy {len(models)}/5 mô hình. Hãy chạy lại train_models.py để tạo đủ artifacts.")

if submitted:
    total_capacity = float(capacity)
    stick_count = int(sticks)
    base = {
        # capacity_gb trong cả 3 dataset là dung lượng CẢ BỘ (đã kiểm chứng ở D2) -> truyền tổng dung lượng
        "capacity_gb": total_capacity,
        "total_capacity_gb": total_capacity,
        "bus_speed_mhz": float(speed),
        "speed_mhz": float(speed),
        "sticks_count": stick_count,
        "num_sticks": stick_count,
        "cas_latency": float(cl) if cl > 0 else np.nan,
        "brand": brand,
        "ram_generation": generation,
        "condition": condition,
        "unit_type": unit_type,
        "type": ram_type,
        "is_bulk_server": int(is_bulk),
        "is_sodimm": int(is_sodimm),
        "is_laptop": int(is_sodimm or ram_type == "Laptop"),
        "has_rgb": int(has_rgb),
        "is_rgb": int(has_rgb),
        "is_ecc_server": int(is_ecc),
        "is_ecc_rdimm": int(is_ecc),
        "source": source,
        "bandwidth_index": total_capacity * float(speed) / 1000.0,
        "log_capacity": np.log2(max(total_capacity, 1)),
        "log_bus_speed": np.log2(max(float(speed), 100)),
        "cap_speed_ratio": total_capacity / max(float(speed), 100),
        # 0 = không biết -> NaN, pipeline sẽ điền trung vị lúc huấn luyện
        "wasPrice": float(was_price) if was_price > 0 else np.nan,
        "available": float(available) if available > 0 else np.nan,
        "sold": float(sold) if sold > 0 else np.nan,
    }
    rows = []
    for name, model in models.items():
        info = metadata[name]
        schema = info.get("features", [])
        defaults = info.get("defaults", {})
        row = {col: base.get(col, defaults.get(col, np.nan)) for col in schema}

        frame = pd.DataFrame([row], columns=schema)
        raw = float(np.asarray(model.predict(frame)).reshape(-1)[0])
        if info.get("inverse_log1p", False):
            raw = float(np.expm1(raw))

        # q = ngưỡng conformal (đã lưu lúc huấn luyện), cùng đơn vị với target của mô hình
        q = info.get("conformal_q")
        if info.get("target") == "usd_per_gb":
            usd_per_gb = max(raw, 0.0)
            price_usd = usd_per_gb * total_capacity
            unit_note = "USD/GB → quy đổi theo tổng dung lượng"
            if q is not None:
                lower_price = max(0.0, (raw - q) * total_capacity)
                upper_price = max(0.0, (raw + q) * total_capacity)
            else:
                lower_price = upper_price = np.nan
        else:
            price_usd = max(raw, 0.0)
            usd_per_gb = price_usd / total_capacity
            unit_note = "USD/bộ RAM"
            if q is not None:
                lower_price = max(0.0, raw - q)
                upper_price = max(0.0, raw + q)
            else:
                lower_price = upper_price = np.nan

        coverage = info.get("test_coverage")
        m = info.get("metrics", {})
        mape = m.get("MAPE")
        acc_general = max(0.0, 100.0 - mape) if mape is not None else np.nan
        # Sai số kỳ vọng của LẦN DỰ ĐOÁN NÀY: nửa độ rộng khoảng 90% / giá dự đoán
        rel_err = (q / raw * 100.0) if (q is not None and raw > 0) else np.nan

        # Độ chính xác theo nhóm (thế hệ x dung lượng), đo trên tập test
        cap_col = info.get("segment_cap_col")
        cap_val = total_capacity if cap_col == "__total__" else row.get(cap_col, np.nan)
        seg = lookup_segment(info, generation, cap_val)
        if seg:
            accuracy, w10, w20 = seg["accuracy"], seg["within_10"], seg["within_20"]
            prod_txt = f", {seg['n_products']} sản phẩm" if seg.get("n_products") is not None else ""
            seg_note = f"{seg['generation']} · {seg['cap_group']} (n={seg['n']}{prod_txt})"
        else:
            accuracy, w10, w20 = acc_general, m.get("within_10", np.nan), m.get("within_20", np.nan)
            seg_note = "Chung (nhóm <20 mẫu test hoặc ngoài phạm vi)"

        rows.append({
            "Thuật toán": name,
            "Hãng": brand,
            "Thế hệ": generation,
            "Dataset huấn luyện": info.get("dataset", "?"),
            "Giá dự đoán (USD)": price_usd,
            "Giá quy đổi (USD/GB)": usd_per_gb,
            "Khoảng thấp (USD)": lower_price,
            "Khoảng cao (USD)": upper_price,
            "Coverage test (%)": coverage * 100 if coverage is not None else np.nan,
            "Độ chính xác (%)": accuracy,
            "Độ chính xác chung (%)": acc_general,
            "Nhóm đánh giá": seg_note,
            "Sai lệch ≤10% (%)": w10,
            "Sai lệch ≤20% (%)": w20,
            "Sai số khoảng ± (% giá)": rel_err,
            "MAE": m.get("MAE", np.nan), "RMSE": m.get("RMSE", np.nan), "R²": m.get("R2", np.nan),
            "Đơn vị gốc": unit_note,
        })
    result = pd.DataFrame(rows)

    # Lưu kết quả để đổi thuật toán / kéo slider (Streamlit chạy lại script) không làm mất kết quả
    st.session_state["result"] = result
    st.session_state["ctx"] = {
        "brand": brand, "generation": generation, "total_capacity": total_capacity,
        "stick_count": stick_count, "speed": float(speed), "cl": float(cl),
        "condition": condition, "is_bulk": int(is_bulk), "was_price": float(was_price),
        "base": base,
    }

# Toàn bộ phần hiển thị đọc từ session_state, không phụ thuộc nút bấm
if "result" in st.session_state:
    result = st.session_state["result"]
    ctx = st.session_state["ctx"]
    brand, generation, total_capacity = ctx["brand"], ctx["generation"], ctx["total_capacity"]
    stick_count, speed, cl, condition = ctx["stick_count"], ctx["speed"], ctx["cl"], ctx["condition"]
    is_bulk, was_price, base = ctx["is_bulk"], ctx["was_price"], ctx["base"]

    st.subheader("Kết quả dự đoán")
    st.markdown(
        f"**Cấu hình:** {brand} · {generation} · {total_capacity:g} GB ({stick_count} thanh) · "
        f"{speed:g} MHz · CL{cl:g} · {condition}")
    if generation != "DDR5" or int(is_bulk) == 1:
        st.warning(
            "Random Forest, Gradient Boosting và SVR chỉ được huấn luyện trên RAM DDR5 không bulk (D1+D2+D3). "
            "Với cấu hình hiện tại, kết quả của 3 mô hình này nằm ngoài phạm vi huấn luyện, chỉ nên tham khảo.")
    st.metric("Dung lượng bộ RAM", f"{total_capacity:g} GB")
    st.dataframe(
        result[["Thuật toán", "Hãng", "Thế hệ", "Dataset huấn luyện", "Giá dự đoán (USD)", "Giá quy đổi (USD/GB)"]].style.format(
            {"Giá dự đoán (USD)": "${:,.2f}", "Giá quy đổi (USD/GB)": "${:,.3f}"}),
        use_container_width=True, hide_index=True)
    st.bar_chart(result.set_index("Thuật toán")[["Giá dự đoán (USD)"]])

    st.subheader("Độ chính xác dự đoán")
    st.dataframe(
        result[["Thuật toán", "Nhóm đánh giá", "Độ chính xác (%)", "Độ chính xác chung (%)",
                "Sai lệch ≤10% (%)", "Sai lệch ≤20% (%)", "Sai số khoảng ± (% giá)"]].style.format(
            {"Độ chính xác (%)": "{:.1f}%", "Độ chính xác chung (%)": "{:.1f}%",
             "Sai lệch ≤10% (%)": "{:.1f}%", "Sai lệch ≤20% (%)": "{:.1f}%",
             "Sai số khoảng ± (% giá)": "±{:.1f}%"}, na_rep="—"),
        use_container_width=True, hide_index=True)
    st.caption(
        "• **Độ chính xác (%)** = 100% − MAPE, tính trên các mẫu TEST thuộc cùng nhóm (thế hệ × nhóm dung lượng) với cấu hình bạn nhập; "
        "*n* là số dòng test của nhóm (một sản phẩm bán lẻ có thể lặp lại nhiều dòng), số sản phẩm khác nhau ghi kèm trong cột *Nhóm đánh giá*. "
        "Nếu nhóm có dưới 20 mẫu thì dùng độ chính xác chung của mô hình. "
        "• **Sai lệch ≤10% / ≤20%** = tỷ lệ mẫu test của nhóm có giá dự đoán lệch giá thật không quá 10% / 20%. "
        "Đây là số đo thực nghiệm trên dữ liệu kiểm tra, không phải xác suất đúng của riêng lần dự đoán này. "
        "• **Sai số khoảng ±%** là nửa độ rộng khoảng 90% so với giá dự đoán, nên thay đổi theo đầu vào.")

    st.subheader("Khoảng dự đoán 90% và mức bao phủ")
    cov_view = result[["Thuật toán", "Giá dự đoán (USD)", "Khoảng thấp (USD)", "Khoảng cao (USD)", "Coverage test (%)"]].rename(
        columns={"Coverage test (%)": "Coverage test của mô hình (%)"})
    st.dataframe(
        cov_view.style.format(
            {"Giá dự đoán (USD)": "${:,.2f}", "Khoảng thấp (USD)": "${:,.2f}",
             "Khoảng cao (USD)": "${:,.2f}", "Coverage test của mô hình (%)": "{:.2f}%"}, na_rep="—"),
        use_container_width=True, hide_index=True)
    st.caption(
        "Khoảng dự đoán dùng Split Conformal Prediction, mức bao phủ mục tiêu 90% (hiệu chỉnh trên tập calibration riêng). "
        "Coverage test của mô hình là tỷ lệ mẫu thực tế nằm trong khoảng trên tập test độc lập, có thể cao hoặc thấp hơn 90%; "
        "đây là trung bình trên tập test, không phải độ chính xác của từng lần dự đoán. "
        "Mỗi mô hình có ngưỡng và coverage riêng, tính trên dataset của chính nó.")

    st.subheader("Chỉ số đánh giá trên tập test của từng mô hình")
    st.caption("MAE/RMSE/R² giữ nguyên đơn vị và tập kiểm tra của từng mô hình. Vì RF/GB/SVR dùng USD/GB còn Linear và XGBoost dùng USD/bộ, không so sánh trực tiếp các MAE/RMSE giữa hai nhóm. Tập test của các mô hình cũng không trùng nhau.")
    st.dataframe(
        result[["Thuật toán", "Dataset huấn luyện", "Đơn vị gốc", "MAE", "RMSE", "R²"]].style.format(
            {"MAE": "{:,.4f}", "RMSE": "{:,.4f}", "R²": "{:.4f}"}, na_rep="—"),
        use_container_width=True, hide_index=True)

    # ---- Độ nhạy của XGBoost với các trường listing ----
    if XGB_NAME in models:
        xinfo = metadata[XGB_NAME]
        xschema = xinfo.get("features", [])
        if "wasPrice" in xschema:
            st.subheader("Độ nhạy của XGBoost theo Giá niêm yết cũ")
            cases = [("Không có (dùng trung vị huấn luyện)", np.nan)]
            if was_price > 0:
                cases += [("0.5 × giá đã nhập", was_price * 0.5),
                          ("1.0 × giá đã nhập", was_price),
                          ("1.5 × giá đã nhập", was_price * 1.5)]
            sens = []
            for label, wp in cases:
                b = dict(base)
                b["wasPrice"] = wp
                r = {col: b.get(col, xinfo.get("defaults", {}).get(col, np.nan)) for col in xschema}
                val = float(np.asarray(models[XGB_NAME].predict(pd.DataFrame([r], columns=xschema))).reshape(-1)[0])
                if xinfo.get("inverse_log1p", False):
                    val = float(np.expm1(val))
                sens.append({"Giá niêm yết cũ": "—" if pd.isna(wp) else f"${wp:,.2f}",
                             "Trường hợp": label, "XGBoost dự đoán (USD)": max(val, 0.0)})
            st.dataframe(pd.DataFrame(sens).style.format({"XGBoost dự đoán (USD)": "${:,.2f}"}),
                         use_container_width=True, hide_index=True)
            st.caption(
                "Bảng chạy lại XGBoost với các giá trị Giá niêm yết cũ khác nhau, giữ nguyên các thông số còn lại. "
                "Nếu giá thay đổi nhiều, mô hình đang phụ thuộc mạnh vào thuộc tính listing này, không chỉ vào cấu hình phần cứng.")

    st.info("XGBoost có thể cần các thuộc tính listing bổ sung (đặc biệt Giá niêm yết cũ). Nếu một trường không được nhập (0), demo để mô hình dùng giá trị trung vị lúc huấn luyện, nên kết quả XGBoost phụ thuộc mạnh vào các trường này.")
    st.download_button("Tải kết quả CSV", result.to_csv(index=False).encode("utf-8-sig"), "ram_prediction_results.csv", "text/csv")
if "result" not in st.session_state:
    st.info("Nhập thông số RAM ở thanh bên trái rồi nhấn **Dự đoán giá RAM**.")

# ==========================================================
# PHẦN 2: KỊCH BẢN GIÁ 24 THÁNG
# Nằm ngoài khối `if submitted` để kéo slider không làm mất kết quả phía trên.
# ==========================================================
if "result" in st.session_state:
    res = st.session_state["result"]

    st.divider()
    st.subheader("Kịch bản giá trong 24 tháng tới")
    st.markdown(
        f"**Cấu hình:** {brand} · {generation} · {total_capacity:g} GB ({stick_count} thanh) · "
        f"{speed:g} MHz · CL{cl:g} · {condition}")

    trend_file = MODEL_DIR / "market_trend.json"
    est = 0.0
    ci_half = 0.015
    if trend_file.exists():
        trs = json.loads(trend_file.read_text(encoding="utf-8"))
        if "monthly_change" in trs:  # tương thích file market_trend.json bản cũ (1 nguồn)
            trs = {"D3": trs}
        labels = {"D1": "D1 (bán lẻ, ~2 tháng)", "D3": "D3 (costinflation, ~4 tuần)"}
        src = st.radio("Nguồn đo xu hướng", list(trs.keys()), horizontal=True,
                       format_func=lambda k: labels.get(k, k))
        tr = trs[src]
        est, lo, hi = tr["monthly_change"], tr["monthly_ci_low"], tr["monthly_ci_high"]
        st.caption(
            f"Xu hướng đo từ {src} ({tr['start']} → {tr['end']}, {tr['n_products']} sản phẩm lặp lại, {tr['n_days']} ngày): "
            f"{est*100:+.2f}%/tháng, khoảng tin cậy 95% [{lo*100:+.2f}%, {hi*100:+.2f}%]. "
            "Dữ liệu chỉ vài tuần đến vài tháng nên đây là kịch bản giả định, không phải dự báo đã kiểm chứng.")
        ci_half = (hi - lo) / 2  # biên độ mặc định lấy từ khoảng tin cậy 95% của chính xu hướng đo được
        

    rate = st.slider("Tốc độ thay đổi giá thị trường mỗi tháng (%)", -10.0, 10.0,
                     float(np.clip(est * 100, -10, 10)), 0.1, key=f"rate_{src}") / 100
    spread = st.slider("Biên độ kịch bản thấp/cao (± %/tháng quanh mức trên; mặc định = nửa độ rộng khoảng tin cậy 95%)",
                       0.0, 5.0, float(np.clip(ci_half * 100, 0, 5)), 0.1, key=f"spread_{src}") / 100

    months = np.arange(0, 25)
    scen_rows = []
    for _, r in res.iterrows():
        p0 = r["Giá dự đoán (USD)"]
        # Cận thấp/cao kết hợp 2 nguồn bất định: khoảng dự đoán 90% của mô hình (tháng 0) × xu hướng thấp/cao
        p_lo = r["Khoảng thấp (USD)"] if pd.notna(r["Khoảng thấp (USD)"]) else p0
        p_hi = r["Khoảng cao (USD)"] if pd.notna(r["Khoảng cao (USD)"]) else p0
        for m_ in months:
            scen_rows.append({
                "Thuật toán": r["Thuật toán"], "Tháng": int(m_),
                "Thấp": p_lo * max(1 + rate - spread, 0) ** m_,
                "Cơ sở": p0 * (1 + rate) ** m_,
                "Cao": p_hi * (1 + rate + spread) ** m_,
            })
    scen = pd.DataFrame(scen_rows)

    pick = st.selectbox("Xem kịch bản theo thuật toán", res["Thuật toán"].tolist(), key="scen_algo")
    s = scen[scen["Thuật toán"] == pick].set_index("Tháng")[["Thấp", "Cơ sở", "Cao"]]
    st.line_chart(s)
    st.dataframe(
        s.loc[[0, 6, 12, 18, 24]].reset_index().rename(columns={"Tháng": "Sau (tháng)"})
         .style.format({"Thấp": "${:,.2f}", "Cơ sở": "${:,.2f}", "Cao": "${:,.2f}"}),
        use_container_width=True, hide_index=True)
    st.markdown(f"**Độ chính xác của thuật toán đang xem: {pick}**")
    pr = res[res["Thuật toán"] == pick].iloc[0]
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Độ chính xác (nhóm)", f"{pr['Độ chính xác (%)']:.1f}%")
    a2.metric("Sai lệch ≤10%", f"{pr['Sai lệch ≤10% (%)']:.1f}%")
    a3.metric("Sai lệch ≤20%", f"{pr['Sai lệch ≤20% (%)']:.1f}%")
    a4.metric("Sai số khoảng 90%", f"±{pr['Sai số khoảng ± (% giá)']:.1f}%")
    st.caption(f"Nhóm đánh giá: {pr['Nhóm đánh giá']}.")
    st.caption(
        "• **Cơ sở** = giá dự đoán theo cấu hình × (1 + tốc độ thay đổi)^số tháng. "
        "**Thấp/Cao** kết hợp khoảng dự đoán 90% của mô hình ở tháng 0 với xu hướng thấp/cao, nên là biên rộng. "
        "• Các chỉ số độ chính xác ở trên đo **giá hiện tại** trên tập test. Độ chính xác của giá **sau 24 tháng** "
        "không thể đo được vì chưa có dữ liệu tương lai để đối chiếu. "
        "• Tốc độ đo trong 1-3 tháng khi chiếu ra 24 tháng bị khuếch đại rất mạnh.")

st.divider()
st.caption("Lưu ý: Giá dự đoán là kết quả mô hình học từ dữ liệu nghiên cứu, không phải giá thị trường thời gian thực.")