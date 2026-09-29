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

if "original_data" not in st.session_state:
    st.session_state.original_data = None

if "data_initialized" not in st.session_state:
    st.session_state.data_initialized = False


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

# ============================================================
# FUNCTION: LOAD MODEL
# ============================================================

@st.cache_resource
def load_model(model_path):

    model_path = os.path.abspath(model_path)

    # Cek file
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"File model tidak ditemukan:\n{model_path}"
        )

    if not os.path.isfile(model_path):
        raise RuntimeError(
            f"Path model bukan file:\n{model_path}"
        )

    # Cek ukuran
    file_size = os.path.getsize(model_path)

    if file_size == 0:
        raise RuntimeError(
            "File model berukuran 0 byte."
        )

    try:

        # Load langsung dari path
        model_package = joblib.load(model_path)

    except PermissionError as e:

        raise RuntimeError(
            "Permission denied saat membaca file model.\n"
            f"Path: {model_path}\n"
            f"Ukuran: {file_size / (1024 * 1024):.2f} MB\n"
            f"Detail: {e}"
        )

    except Exception as e:

        raise RuntimeError(
            f"File model ditemukan tetapi gagal dibaca.\n"
            f"Path: {model_path}\n"
            f"Ukuran: {file_size / (1024 * 1024):.2f} MB\n"
            f"Jenis error: {type(e).__name__}\n"
            f"Error: {e}"
        )

    # Cek package model
    if not isinstance(model_package, dict):

        raise ValueError(
            "Isi file model bukan dictionary/package."
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
            "Isi model tidak lengkap.\n"
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

    for col in features:

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

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

    wob_history = pd.to_numeric(
        df_history[WOB_COL],
        errors="coerce"
    ).dropna()

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

    # ========================================================
    # SEARCH RANGE
    # ========================================================

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

    # ========================================================
    # GENERATE CANDIDATE
    # ========================================================

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

    # ========================================================
    # PREDICT
    # ========================================================

    candidate_df[
        "Predicted_ROP"
    ] = predict_with_model(
        model_package,
        candidate_df
    )

    # ========================================================
    # RANKING
    # ========================================================

    candidate_df = candidate_df.sort_values(
        by="Predicted_ROP",
        ascending=False
    ).reset_index(drop=True)

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
# FUNCTION: CONVERT INPUT TO CORRECT TYPE
# ============================================================

def convert_value(value, original_dtype):

    if pd.isna(value):
        return np.nan

    try:

        if pd.api.types.is_integer_dtype(original_dtype):
            return int(value)

        elif pd.api.types.is_float_dtype(original_dtype):
            return float(value)

        elif pd.api.types.is_numeric_dtype(original_dtype):
            return float(value)

        else:
            return str(value)

    except Exception:

        return value


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

    # Gunakan nama file sebagai identitas dataset
    file_id = (
        uploaded_file.name,
        uploaded_file.size
    )

    if (
        "current_file_id" not in st.session_state
        or st.session_state.current_file_id != file_id
    ):

        try:

            df_upload = pd.read_csv(
                uploaded_file
            )

            # Standardisasi kolom
            df_upload = standardize_columns(
                df_upload
            )

            # Konversi numeric jika memungkinkan
            for col in df_upload.columns:

                converted = pd.to_numeric(
                    df_upload[col],
                    errors="coerce"
                )

                if converted.notna().mean() > 0.5:

                    df_upload[col] = converted

            # Validasi Hole Depth
            if DEPTH_COL not in df_upload.columns:

                st.error(
                    f"Kolom '{DEPTH_COL}' tidak ditemukan."
                )

                st.stop()

            # Sort Hole Depth
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

            # Simpan original
            st.session_state.original_data = (
                df_upload.copy()
            )

            # Simpan working data
            st.session_state.drilling_data = (
                df_upload.copy()
            )

            st.session_state.current_file_id = file_id

            st.session_state.data_initialized = True

        except Exception as e:

            st.sidebar.error(
                f"Gagal membaca CSV:\n{e}"
            )


# ============================================================
# CHECK DATA
# ============================================================

if (
    st.session_state.drilling_data is None
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
# DATA MANAGEMENT
# ============================================================

st.divider()

st.header(
    "🗃️ Data Management"
)

st.caption(
    "Gunakan menu ini untuk menambah, mengedit, atau "
    "menghapus data drilling sebagai simulasi kondisi real-time."
)


# ============================================================
# TABS
# ============================================================

tab_add, tab_edit, tab_delete, tab_reset = st.tabs(
    [
        "➕ Tambah Data",
        "✏️ Edit Data",
        "🗑️ Hapus Data",
        "🔄 Reset Data"
    ]
)


# ============================================================
# TAB 1 - ADD DATA
# ============================================================

with tab_add:

    st.subheader(
        "Tambah Baris Drilling"
    )

    st.write(
        f"Jumlah variabel: **{len(df.columns)}**"
    )

    st.write(
        "Masukkan nilai untuk setiap variabel."
    )

    with st.form(
        "add_row_form",
        clear_on_submit=True
    ):

        new_values = {}

        columns = df.columns.tolist()

        # Buat input berdasarkan tipe data
        for col in columns:

            dtype = df[col].dtype

            if pd.api.types.is_numeric_dtype(dtype):

                current_mean = pd.to_numeric(
                    df[col],
                    errors="coerce"
                ).median()

                if pd.isna(current_mean):
                    current_mean = 0.0

                new_values[col] = st.number_input(
                    col,
                    value=float(current_mean),
                    format="%.4f",
                    key=f"add_{col}"
                )

            else:

                new_values[col] = st.text_input(
                    col,
                    key=f"add_{col}"
                )

        add_submit = st.form_submit_button(
            "➕ Tambahkan Baris",
            use_container_width=True
        )

        if add_submit:

            new_row = {}

            for col in columns:

                new_row[col] = convert_value(
                    new_values[col],
                    df[col].dtype
                )

            new_df = pd.DataFrame(
                [new_row]
            )

            df = pd.concat(
                [
                    df,
                    new_df
                ],
                ignore_index=True
            )

            # Sort berdasarkan Hole Depth
            if DEPTH_COL in df.columns:

                df[DEPTH_COL] = pd.to_numeric(
                    df[DEPTH_COL],
                    errors="coerce"
                )

                df = df.sort_values(
                    DEPTH_COL
                ).reset_index(drop=True)

            st.session_state.drilling_data = df

            st.success(
                "✅ Data berhasil ditambahkan."
            )

            st.rerun()


# ============================================================
# TAB 2 - EDIT DATA
# ============================================================

with tab_edit:

    st.subheader(
        "Edit Data Drilling"
    )

    if len(df) == 0:

        st.info(
            "Tidak ada data yang dapat diedit."
        )

    else:

        selected_index = st.selectbox(
            "Pilih index/baris yang ingin diedit",
            options=df.index.tolist(),
            format_func=lambda x: (
                f"Index {x} | "
                f"Hole Depth = {df.loc[x, DEPTH_COL]}"
            )
        )

        selected_row = df.loc[
            selected_index
        ]

        with st.form(
            "edit_row_form"
        ):

            edited_values = {}

            for col in df.columns:

                dtype = df[col].dtype

                value = selected_row[col]

                if pd.api.types.is_numeric_dtype(dtype):

                    if pd.isna(value):
                        value = 0.0

                    edited_values[col] = st.number_input(
                        col,
                        value=float(value),
                        format="%.4f",
                        key=f"edit_{selected_index}_{col}"
                    )

                else:

                    edited_values[col] = st.text_input(
                        col,
                        value="" if pd.isna(value)
                        else str(value),
                        key=f"edit_{selected_index}_{col}"
                    )

            edit_submit = st.form_submit_button(
                "💾 Simpan Perubahan",
                use_container_width=True
            )

            if edit_submit:

                for col in df.columns:

                    df.loc[
                        selected_index,
                        col
                    ] = convert_value(
                        edited_values[col],
                        df[col].dtype
                    )

                # Sort kembali berdasarkan depth
                df[DEPTH_COL] = pd.to_numeric(
                    df[DEPTH_COL],
                    errors="coerce"
                )

                df = df.sort_values(
                    DEPTH_COL
                ).reset_index(drop=True)

                st.session_state.drilling_data = df

                st.success(
                    "✅ Data berhasil diperbarui."
                )

                st.rerun()


# ============================================================
# TAB 3 - DELETE DATA
# ============================================================

with tab_delete:

    st.subheader(
        "Hapus Data Drilling"
    )

    if len(df) == 0:

        st.info(
            "Tidak ada data untuk dihapus."
        )

    else:

        selected_delete = st.multiselect(
            "Pilih index/baris yang ingin dihapus",
            options=df.index.tolist(),
            format_func=lambda x: (
                f"Index {x} | "
                f"Hole Depth = {df.loc[x, DEPTH_COL]}"
            )
        )

        if selected_delete:

            st.warning(
                f"{len(selected_delete)} baris akan dihapus."
            )

            if st.button(
                "🗑️ Hapus Baris Terpilih",
                type="primary",
                use_container_width=True
            ):

                df = df.drop(
                    index=selected_delete
                ).reset_index(drop=True)

                st.session_state.drilling_data = df

                st.success(
                    f"✅ {len(selected_delete)} baris berhasil dihapus."
                )

                st.rerun()


# ============================================================
# TAB 4 - RESET
# ============================================================

with tab_reset:

    st.subheader(
        "Reset Data"
    )

    st.write(
        "Reset akan mengembalikan data ke kondisi "
        "saat pertama kali CSV di-upload."
    )

    if st.button(
        "🔄 Reset ke Data Awal",
        use_container_width=True
    ):

        st.session_state.drilling_data = (
            st.session_state.original_data.copy()
        )

        st.success(
            "✅ Data berhasil dikembalikan ke kondisi awal."
        )

        st.rerun()


# ============================================================
# CURRENT DATA TABLE
# ============================================================

st.subheader(
    "📋 Current Drilling Dataset"
)

st.write(
    f"Total baris saat ini: **{len(df):,}**"
)

st.dataframe(
    df,
    use_container_width=True,
    height=350
)


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


st.divider()

st.subheader(
    "Current Drilling Condition"
)


col1, col2, col3, col4 = st.columns(4)


# ============================================================
# HOLE DEPTH
# ============================================================

with col1:

    st.metric(
        "Current Hole Depth",
        f"{latest[DEPTH_COL]:.2f}"
    )


# ============================================================
# WOB
# ============================================================

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


# ============================================================
# RPM
# ============================================================

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


# ============================================================
# CURRENT ROP
# ============================================================

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

# MODEL ADA DI ROOT REPOSITORY
MODEL_PATH = os.path.join(
    BASE_DIR,
    "rop_final_rf.joblib"
)


try:

    model_package = load_model(
        MODEL_PATH
    )

except Exception as e:

    st.error(
        "❌ Model Random Forest tidak dapat dimuat."
    )

    st.code(
        str(e)
    )

    st.write("### 🔍 Debug Model")

    st.write(
        "Path:",
        MODEL_PATH
    )

    st.write(
        "File exists:",
        os.path.exists(MODEL_PATH)
    )

    st.write(
        "Is file:",
        os.path.isfile(MODEL_PATH)
    )

    if os.path.exists(MODEL_PATH):

        st.write(
            "Ukuran file:",
            f"{os.path.getsize(MODEL_PATH) / (1024 * 1024):.2f} MB"
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
