# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Churn Predictor", layout="wide")
st.title("Prediksi Churn Customer (Random Forest)")

# =========================
# PATH
# =========================
MODEL_PATH = "models/rf_clf.joblib"
FEATURE_COLS_PATH = "models/feature_cols.joblib"

# =========================
# LOAD
# =========================
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    if not os.path.exists(FEATURE_COLS_PATH):
        raise FileNotFoundError(
            f"feature_cols tidak ditemukan: {FEATURE_COLS_PATH}\n"
            f"Simpan dulu list X.columns saat training ke file ini."
        )

    feature_cols = joblib.load(FEATURE_COLS_PATH)

    # Validasi sederhana
    if not isinstance(feature_cols, (list, tuple)) or len(feature_cols) == 0:
        raise ValueError("feature_cols.joblib harus berisi list kolom fitur (X.columns).")

    return model, list(feature_cols)

rf_clf, FEATURE_COLS = load_artifacts()

st.caption(f"Jumlah fitur model: {len(FEATURE_COLS)}")

# =========================
# HELPER
# =========================
def build_input_row(feature_cols: list[str]) -> pd.DataFrame:
    """
    Buat 1 baris input dari user.
    - Untuk fitur tanggal: pakai first_purchase_date dan last_purchase_date jika memang ada di training,
      tapi preprocessing kamu akhirnya drop dua kolom tanggal dan pakai customer_age_days.
    - Jadi di sini kita akan selalu minta input tanggal, lalu bikin customer_age_days.
    - Kolom lain dibuat dari input numerik.
    """

    st.subheader("Input Data Customer")

    c1, c2 = st.columns(2)
    with c1:
        first_date = st.date_input("First Purchase Date")
    with c2:
        last_date = st.date_input("Last Purchase Date")

    # Siapkan dict untuk 1 baris data
    row = {}

    # Isi semua fitur selain kolom tanggal, dengan input numerik default.
    # Karena kita tidak tahu tipe setiap kolom dari dataset kamu, kita buat input float.
    st.markdown("### Input Fitur (Numerik)")
    cols_ui = st.columns(3)
    col_idx = 0

    # Kolom tanggal biasanya sudah kamu drop saat training, jadi jangan ikut di sini
    skip_cols = {"first_purchase_date", "last_purchase_date"}

    for col in feature_cols:
        if col in skip_cols:
            continue

        with cols_ui[col_idx % 3]:
            row[col] = st.number_input(
                label=col,
                value=0.0,
                step=1.0,
                format="%.4f"
            )
        col_idx += 1

    # Tambahkan tanggal untuk proses customer_age_days
    row["first_purchase_date"] = pd.to_datetime(first_date)
    row["last_purchase_date"] = pd.to_datetime(last_date)

    X_input = pd.DataFrame([row])

    # Preprocessing tanggal sesuai kode kamu:
    X_input["first_purchase_date"] = pd.to_datetime(X_input["first_purchase_date"], errors="coerce")
    X_input["last_purchase_date"] = pd.to_datetime(X_input["last_purchase_date"], errors="coerce")
    X_input["customer_age_days"] = (X_input["last_purchase_date"] - X_input["first_purchase_date"]).dt.days
    X_input = X_input.drop(columns=["first_purchase_date", "last_purchase_date"])

    # Handle missing
    X_input = X_input.fillna(0)

    # Samakan kolom dengan training
    # Jika training kamu punya customer_age_days, harus ada di FEATURE_COLS
    X_final = X_input.reindex(columns=feature_cols, fill_value=0)

    return X_final


# =========================
# BUILD INPUT
# =========================
X_final = build_input_row(FEATURE_COLS)

with st.expander("Lihat data input (setelah preprocessing)"):
    st.dataframe(X_final, use_container_width=True)

# =========================
# PREDICT
# =========================
st.markdown("### Prediksi")
btn = st.button("Prediksi Churn")

if btn:
    pred_class = int(rf_clf.predict(X_final)[0])
    pred_proba = float(rf_clf.predict_proba(X_final)[:, 1][0])

    cA, cB, cC = st.columns(3)
    with cA:
        st.metric("Prediksi Kelas", pred_class)
        st.caption("0 = tidak churn, 1 = churn")
    with cB:
        st.metric("Probabilitas Churn", f"{pred_proba:.3f}")
    with cC:
        st.metric("Threshold", "0.500")

    st.divider()

    if pred_proba >= 0.5:
        st.warning("Risiko churn tinggi (>= 0.5).")
    else:
        st.success("Risiko churn rendah (< 0.5).")

    st.markdown("#### Interpretasi singkat")
    st.write(
        "Model menghasilkan label churn (0/1) dan probabilitas churn (0 sampai 1). "
        "Probabilitas lebih cocok untuk analisis risiko dan penentuan tindakan (retention)."
    )
