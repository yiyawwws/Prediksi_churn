import io
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Churn Risk Analysis", layout="wide")
st.title("Churn Risk Analysis")
st.caption(
    "Pipeline: Klasifikasi (Random Forest) → Regresi (Ensemble VotingRegressor). "
    "Tanpa KNN dan tanpa Decision Tree tunggal."
)

# =========================
# LOAD MODEL & ARTIFACT
# =========================
try:
    clf = joblib.load("rf_classifier.pkl")
    reg = joblib.load("reg_ensemble.pkl")
    base_cols = joblib.load("feature_columns_base.pkl")
except Exception as e:
    st.error("Model tidak ditemukan. Pastikan file berikut ada satu folder dengan app.py:")
    st.code(
        "rf_classifier.pkl\n"
        "reg_ensemble.pkl\n"
        "feature_columns_base.pkl\n"
        "requirements.txt"
    )
    st.exception(e)
    st.stop()

# =========================
# PREPROCESS FUNCTION
# =========================
def preprocess_base(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])

    if "churn_risk" in df.columns:
        df = df.drop(columns=["churn_risk"])

    # tanggal ke numerik
    if "first_purchase_date" in df.columns:
        df["first_purchase_date"] = pd.to_datetime(df["first_purchase_date"], errors="coerce")
    if "last_purchase_date" in df.columns:
        df["last_purchase_date"] = pd.to_datetime(df["last_purchase_date"], errors="coerce")

    if "first_purchase_date" in df.columns and "last_purchase_date" in df.columns:
        df["customer_age_days"] = (
            df["last_purchase_date"] - df["first_purchase_date"]
        ).dt.days

    for c in ["first_purchase_date", "last_purchase_date"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    df = df.fillna(0)

    # samakan kolom dengan saat training
    for c in base_cols:
        if c not in df.columns:
            df[c] = 0

    return df[base_cols]

# =========================
# UPLOAD DATA
# =========================
uploaded = st.file_uploader("Upload file CSV", type=["csv"])
if not uploaded:
    st.info("Silakan upload file CSV terlebih dahulu.")
    st.stop()

df_raw = pd.read_csv(uploaded)

st.subheader("Preview Data")
st.dataframe(df_raw.head(10), use_container_width=True)

X_base = preprocess_base(df_raw)

# =========================
# STEP 1: KLASIFIKASI
# =========================
st.header("Step 1: Klasifikasi (Random Forest)")

pred_class = clf.predict(X_base)
proba_churn1 = clf.predict_proba(X_base)[:, 1]

c1, c2 = st.columns(2)

with c1:
    fig = plt.figure()
    pd.Series(pred_class).value_counts().sort_index().plot(kind="bar")
    plt.title("Distribusi Prediksi Kelas Churn")
    plt.xlabel("Kelas")
    plt.ylabel("Jumlah")
    st.pyplot(fig)

with c2:
    if "churn_risk" in df_raw.columns:
        y_true_bin = (df_raw["churn_risk"] >= 0.5).astype(int)
        cm = confusion_matrix(y_true_bin, pred_class, labels=[0, 1])
        fig = plt.figure()
        disp = ConfusionMatrixDisplay(cm, display_labels=[0, 1])
        disp.plot(values_format="d")
        plt.title("Confusion Matrix")
        st.pyplot(fig)
    else:
        st.info("Confusion Matrix tampil jika kolom churn_risk tersedia.")

# ROC Curve
if "churn_risk" in df_raw.columns:
    st.subheader("ROC Curve (Klasifikasi)")
    y_true_bin = (df_raw["churn_risk"] >= 0.5).astype(int)

    fpr, tpr, _ = roc_curve(y_true_bin, proba_churn1)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    st.pyplot(fig)

# =========================
# STEP 2: REGRESI + ENSEMBLE
# =========================
st.header("Step 2: Regresi (Ensemble VotingRegressor)")

X_reg = X_base.copy()
X_reg["pred_proba_churn1"] = proba_churn1

pred_reg = reg.predict(X_reg)

fig = plt.figure()
plt.hist(pred_reg, bins=30)
plt.title("Distribusi Prediksi churn_risk")
plt.xlabel("Nilai churn_risk")
plt.ylabel("Frekuensi")
st.pyplot(fig)

if "churn_risk" in df_raw.columns:
    st.subheader("Actual vs Predicted (Regresi)")
    y_true_reg = df_raw["churn_risk"].astype(float)

    fig = plt.figure()
    plt.scatter(y_true_reg, pred_reg)
    plt.xlabel("Actual churn_risk")
    plt.ylabel("Predicted churn_risk")
    plt.title("Actual vs Predicted")
    st.pyplot(fig)

    st.subheader("Evaluasi Regresi")
    st.write("MAE:", mean_absolute_error(y_true_reg, pred_reg))
    st.write("MSE:", mean_squared_error(y_true_reg, pred_reg))
    st.write("R2 :", r2_score(y_true_reg, pred_reg))

# =========================
# OUTPUT EXCEL
# =========================
hasil = df_raw.copy()
hasil["prediksi_kelas_churn"] = pred_class
hasil["prob_kelas1"] = proba_churn1
hasil["prediksi_nilai_churn_risk"] = pred_reg

st.subheader("Hasil Akhir (Preview)")
st.dataframe(hasil.head(10), use_container_width=True)

buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
    hasil.to_excel(writer, index=False, sheet_name="hasil_pipeline")

st.download_button(
    "Download Hasil (Excel)",
    buffer.getvalue(),
    file_name="hasil_pipeline.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
