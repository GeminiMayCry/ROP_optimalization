# ============================================================
# ROP OPTIMIZATION APP
# ============================================================

import os
import joblib

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ROP Optimization",
    page_icon="⛏️",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "drilling_data" not in st.session_state:
    st.session_state.drilling_data = None


# ============================================================
# CONSTANT
# ============================================================

WOB_COL = "Weight_on_Bit"
RPM_COL = "Rotary_RPM"
DEPTH_COL = "Hole_Depth"
TARGET_COL = "Rate_Of_Penetration"


# ============================================================
# FUNCTION: STANDARDIZE COLUMN
# ============================================================

def standardize_columns(df):

    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.replace(" ", "_", regex=False)
    )

    return df


# ============================================================
# FUNCTION: LOAD MODEL
# ============================================================

def load_model(model_path):

    model_path = os.path.abspath(model_path)

    if not os.path.isfile(model_path):

        raise FileNotFoundError(
            f"File model tidak ditemukan:\n{model_path}"
        )

    try:

        with open(model_path, "rb") as f:
            model_package = joblib.load(f)

    except Exception as e:

        raise RuntimeError(
            f"File PKL ditemukan tetapi gagal dibaca.\n"
            f"Path: {model_path}\n"
            f"Error: {e}"
        )

    required_keys = [
        "model",
        "features",
        "medians"
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in model_package
    ]

    if missing_keys:

        raise ValueError(
            f"Isi model PKL tidak lengkap.\n"
            f"Key yang hilang: {missing_keys}"
        )

    return model_package


# ============================================================
# FUNCTION: PREDICT ROP
# ============================================================

def predict_with_model(model_package, input_df):

    model = model_package["model"]
    features = model_package["features"]
    medians = model_package["medians"]

    # Pastikan semua feature tersedia
    missing_features = [
        col
        for col in features
        if col not in input_df.columns
    ]

    if missing_features:

        raise ValueError(
            "Feature berikut tidak tersedia pada data:\n"
            + "\n".join(missing_features)
        )

    X = input_df[features].copy()

    # Konversi numeric
    for col in features:

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

    # Inf menjadi NaN
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Isi missing menggunakan median training
    for col in features:

        if X[col].isna().any():

            median_value = medians.get(
                col,
                np.nan
            )

            X[col] = X[col].fillna(
                median_value
            )

    return model.predict(X)


# ============================================================
# FUNCTION: OPTIMIZATION
# ============================================================

def optimize_rop_real(
    current_row,
    df_history,
    model_package
):

    # --------------------------------------------------------
    # Validasi kolom
    # --------------------------------------------------------

    required_columns = [
        WOB_COL,
        RPM_COL
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df_history.columns
    ]

    if missing_columns:

        raise ValueError(
            "Kolom berikut tidak ditemukan:\n"
            + "\n".join(missing_columns)
        )

    # --------------------------------------------------------
    # Historical WOB
    # --------------------------------------------------------

    wob_history = pd.to_numeric(
        df_history[WOB_COL],
        errors="coerce"
    ).dropna()

    # --------------------------------------------------------
    # Historical RPM
    # --------------------------------------------------------

    rpm_history = pd.to_numeric(
        df_history[RPM_COL],
        errors="coerce"
    ).dropna()

    if wob_history.empty:

        raise ValueError(
            "Tidak terdapat data WOB yang valid."
        )

    if rpm_history.empty:

        raise ValueError(
            "Tidak terdapat data RPM yang valid."
        )

    # --------------------------------------------------------
    # Search range
    #
    # Menggunakan percentile 5% - 95%
    # agar kandidat tidak terlalu ekstrem.
    # --------------------------------------------------------

    wob_min = float(
        wob_history.quantile(0.05)
    )

    wob_max = float(
        wob_history.quantile(0.95)
    )

    rpm_min = float(
        rpm_history.quantile(0.05)
    )

    rpm_max = float(
        rpm_history.quantile(0.95)
    )

    # --------------------------------------------------------
    # Generate kandidat
    # --------------------------------------------------------

    wob_values = np.linspace(
        wob_min,
        wob_max,
        20
    )

    rpm_values = np.linspace(
        rpm_min,
        rpm_max,
        20
    )

    candidates = []

    for wob in wob_values:

        for rpm in rpm_values:

            candidate = current_row.copy()

            candidate[WOB_COL] = wob
            candidate[RPM_COL] = rpm

            candidates.append(candidate)

    candidate_df = pd.DataFrame(
        candidates
    )

    # --------------------------------------------------------
    # Prediksi menggunakan model RF PKL
    # --------------------------------------------------------

    candidate_df[
        "Predicted_ROP"
    ] = predict_with_model(
        model_package,
        candidate_df
    )

    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------

    candidate_df = candidate_df.sort_values(
        by="Predicted_ROP",
        ascending=False
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Best candidate
    # --------------------------------------------------------

    best = candidate_df.iloc[0]

    return {
        "best": best,
        "results": candidate_df,
        "wob_range": (
            wob_min,
            wob_max
        ),
        "rpm_range": (
            rpm_min,
            rpm_max
        )
    }


# ============================================================
# TITLE
# ============================================================

st.title(
    "⛏️ Rate of Penetration Optimization"
)

st.caption(
    "Optimasi WOB dan RPM menggunakan Random Forest"
)


# ============================================================
# SIDEBAR - UPLOAD DATA
# ============================================================

st.sidebar.header(
    "📂 Data Drilling"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV Data Drilling",
    type=["csv"]
)


# ============================================================
# READ UPLOADED DATA
# ============================================================

if uploaded_file is not None:

    try:

        df_upload = pd.read_csv(
            uploaded_file
        )

        # Standardisasi kolom
        df_upload = standardize_columns(
            df_upload
        )

        # Konversi kolom ke numeric jika memungkinkan
        for col in df_upload.columns:

            converted = pd.to_numeric(
                df_upload[col],
                errors="coerce"
            )

            # Jika sebagian besar data berhasil menjadi numeric,
            # gunakan hasil konversinya.
            if converted.notna().mean() > 0.5:

                df_upload[col] = converted

        # ----------------------------------------------------
        # Validasi Hole Depth
        # ----------------------------------------------------

        if DEPTH_COL not in df_upload.columns:

            st.error(
                f"Kolom '{DEPTH_COL}' tidak ditemukan."
            )

            st.stop()

        # ----------------------------------------------------
        # Sort Hole Depth
        # ----------------------------------------------------

        df_upload[DEPTH_COL] = pd.to_numeric(
            df_upload[DEPTH_COL],
            errors="coerce"
        )

        df_upload = df_upload.dropna(
            subset=[DEPTH_COL]
        )

        df_upload = df_upload.sort_values(
            DEPTH_COL
        ).reset_index(drop=True)

        # ----------------------------------------------------
        # Simpan ke session
        # ----------------------------------------------------

        st.session_state.drilling_data = (
            df_upload
        )

        st.sidebar.success(
            f"Data berhasil dimuat: "
            f"{len(df_upload):,} baris"
        )

    except Exception as e:

        st.sidebar.error(
            f"Gagal membaca CSV:\n{e}"
        )


# ============================================================
# CHECK DATA
# ============================================================

if (
    "drilling_data" not in st.session_state
    or st.session_state.drilling_data is None
):

    st.warning(
        "Silakan upload data drilling terlebih dahulu "
        "melalui menu di sebelah kiri."
    )

    st.stop()


# ============================================================
# GET DATA
# ============================================================

df = st.session_state.drilling_data.copy()


if df.empty:

    st.warning(
        "Data drilling masih kosong."
    )

    st.stop()


# ============================================================
# SORT DATA
# ============================================================

df[DEPTH_COL] = pd.to_numeric(
    df[DEPTH_COL],
    errors="coerce"
)

df = df.dropna(
    subset=[DEPTH_COL]
)

df = df.sort_values(
    DEPTH_COL
).reset_index(drop=True)

st.session_state.drilling_data = df


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    DEPTH_COL,
    WOB_COL,
    RPM_COL
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(
        "Kolom yang dibutuhkan tidak ditemukan:\n\n"
        + "\n".join(
            f"- {col}"
            for col in missing_columns
        )
    )

    st.stop()


# ============================================================
# LATEST DRILLING CONDITION
# ============================================================

latest = df.iloc[-1]


st.subheader(
    "Current Drilling Condition"
)


col1, col2, col3, col4 = st.columns(4)


# ------------------------------------------------------------
# Hole Depth
# ------------------------------------------------------------

with col1:

    st.metric(
        "Current Hole Depth",
        f"{latest[DEPTH_COL]:.2f}"
    )


# ------------------------------------------------------------
# WOB
# ------------------------------------------------------------

with col2:

    current_wob = pd.to_numeric(
        latest[WOB_COL],
        errors="coerce"
    )

    st.metric(
        "Current WOB",
        f"{current_wob:.2f}"
        if pd.notna(current_wob)
        else "-"
    )


# ------------------------------------------------------------
# RPM
# ------------------------------------------------------------

with col3:

    current_rpm = pd.to_numeric(
        latest[RPM_COL],
        errors="coerce"
    )

    st.metric(
        "Current RPM",
        f"{current_rpm:.2f}"
        if pd.notna(current_rpm)
        else "-"
    )


# ------------------------------------------------------------
# Current ROP
# ------------------------------------------------------------

with col4:

    if TARGET_COL in df.columns:

        current_rop = pd.to_numeric(
            latest[TARGET_COL],
            errors="coerce"
        )

    else:

        current_rop = np.nan

    st.metric(
        "Current ROP",
        f"{current_rop:.2f}"
        if pd.notna(current_rop)
        else "-"
    )


st.divider()


# ============================================================
# LOAD MODEL
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "rop_final_rf.joblib"
)


try:

    model_package = load_model(
        MODEL_PATH
    )

except Exception as e:

    st.error(
        "Model Random Forest tidak dapat dimuat."
    )

    st.code(
        str(e)
    )

    st.stop()


# ============================================================
# MODEL INFORMATION
# ============================================================

st.info(
    "Model yang digunakan: "
    "Random Forest hasil training dan tuning Optuna."
)


# ============================================================
# OPTIMIZATION BUTTON
# ============================================================

if st.button(
    "🔎 Optimize ROP",
    type="primary",
    use_container_width=True
):

    try:

        with st.spinner(
            "Mencari kombinasi WOB dan RPM terbaik..."
        ):

            optimization = optimize_rop_real(
                current_row=latest,
                df_history=df,
                model_package=model_package
            )

        best = optimization["best"]

        results = optimization["results"]

        best_wob = float(
            best[WOB_COL]
        )

        best_rpm = float(
            best[RPM_COL]
        )

        best_rop = float(
            best["Predicted_ROP"]
        )

        # ====================================================
        # RESULT
        # ====================================================

        st.success(
            "Optimasi berhasil menggunakan "
            "model Random Forest."
        )

        st.subheader(
            "Recommended Drilling Condition"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Recommended WOB",
                f"{best_wob:.2f}"
            )

        with c2:

            st.metric(
                "Recommended RPM",
                f"{best_rpm:.2f}"
            )

        with c3:

            st.metric(
                "Predicted ROP",
                f"{best_rop:.2f}"
            )


        # ====================================================
        # CURRENT VS RECOMMENDED
        # ====================================================

        st.subheader(
            "Current vs Recommended"
        )

        comparison = pd.DataFrame({

            "Parameter": [
                "WOB",
                "RPM",
                "ROP"
            ],

            "Current": [
                current_wob,
                current_rpm,
                current_rop
            ],

            "Recommended": [
                best_wob,
                best_rpm,
                best_rop
            ]

        })


        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # SEARCH RANGE
        # ====================================================

        st.subheader(
            "Optimization Search Range"
        )

        r1, r2 = st.columns(2)


        with r1:

            st.metric(
                "WOB Minimum",
                f"{optimization['wob_range'][0]:.2f}"
            )

            st.metric(
                "WOB Maximum",
                f"{optimization['wob_range'][1]:.2f}"
            )


        with r2:

            st.metric(
                "RPM Minimum",
                f"{optimization['rpm_range'][0]:.2f}"
            )

            st.metric(
                "RPM Maximum",
                f"{optimization['rpm_range'][1]:.2f}"
            )


        # ====================================================
        # TOP 10
        # ====================================================

        st.subheader(
            "Top 10 WOB × RPM Candidates"
        )

        top10 = results[
            [
                WOB_COL,
                RPM_COL,
                "Predicted_ROP"
            ]
        ].head(10).copy()


        top10.columns = [
            "WOB",
            "RPM",
            "Predicted ROP"
        ]


        st.dataframe(
            top10,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # HEATMAP
        # ====================================================

        st.subheader(
            "Predicted ROP Heatmap"
        )

        heatmap_data = results.pivot_table(
            index=WOB_COL,
            columns=RPM_COL,
            values="Predicted_ROP"
        )


        fig = px.imshow(

            heatmap_data,

            labels={
                "x": "Rotary RPM",
                "y": "Weight on Bit",
                "color": "Predicted ROP"
            },

            title=(
                "Predicted ROP berdasarkan "
                "WOB dan RPM"
            ),

            aspect="auto"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # ====================================================
        # LATEST DATA
        # ====================================================

        st.subheader(
            "Latest Drilling Data"
        )

        latest_display = (
            latest
            .to_frame(name="Value")
        )


        st.dataframe(
            latest_display,
            use_container_width=True
        )


    except Exception as e:

        st.error(
            "Optimasi gagal."
        )

        st.exception(e)