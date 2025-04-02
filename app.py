
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="Lightcurve Classifier", layout="wide")
st.title("🔭 Lightcurve Classification App")
st.markdown("Upload lightcurve data to classify each object as **Stable** or **Transient** using statistical features.")

# Load model
model = joblib.load("classical_rf_model.pkl")

# Feature extraction
def extract_features(df):
    features = []
    ids = []
    grouped = df.groupby("ID")
    for obj_id, group in grouped:
        if group["Mag"].nunique() <= 1:
            continue

        mag = group["Mag"].values
        mjd = group["MJD"].values.reshape(-1, 1)

        mean_mag = np.mean(mag)
        std_mag = np.std(mag)
        range_mag = np.max(mag) - np.min(mag)
        skew_mag = skew(mag)
        kurt_mag = kurtosis(mag)

        try:
            slope = LinearRegression().fit(mjd, mag).coef_[0]
        except:
            slope = 0.0

        features.append([mean_mag, std_mag, range_mag, skew_mag, kurt_mag, slope])
        ids.append(obj_id)

    return np.array(features), ids

# Upload
uploaded_file = st.file_uploader("Upload lightcurve CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = df[["ID", "MJD", "Mag"]].dropna()

    st.success("✅ File loaded successfully.")
    st.write("Sample data:")
    st.dataframe(df.head())

    X, ids = extract_features(df)
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    preds = model.predict(X)
    result_df = pd.DataFrame({
        "ID": ids,
        "Predicted Label": preds,
        "Label Description": ["Transient Object" if p == 1 else "Stable Object" for p in preds]
    })

    st.subheader("🔍 Classification Results")
    st.dataframe(result_df)

    if (result_df["Predicted Label"] == 1).any():
        st.error("🚨 Transient detected in the uploaded dataset!")

        # Display transient lightcurve
        st.checkbox("Show all transient plots side-by-side", key="show_all", value=False)
        if st.session_state["show_all"]:
            st.subheader("📊 All Detected Transients")
            for tid in transient_ids:
                obj = df[df["ID"] == tid]
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.scatter(obj["MJD"], obj["Mag"], color='blue')
                ax.set_title(f"ID: {tid}")
                ax.set_xlabel("MJD")
                ax.set_ylabel("Mag")
                ax.invert_yaxis()
                st.pyplot(fig)

        transient_ids = result_df[result_df["Predicted Label"] == 1]["ID"].values
        first_id = transient_ids[0]
        obj_curve = df[df["ID"] == first_id]

        st.subheader("📈 Lightcurve of Detected Transient")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.scatter(obj_curve["MJD"], obj_curve["Mag"], color='blue')
        ax.set_xlabel("MJD")
        ax.set_ylabel("Magnitude")
        ax.set_title(f"Lightcurve for ID: {first_id}")
        ax.invert_yaxis()
        st.pyplot(fig)
        st.caption(f"Object ID: `{first_id}`")
