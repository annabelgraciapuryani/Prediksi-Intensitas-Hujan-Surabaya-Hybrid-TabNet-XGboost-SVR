"""
Halaman Data Input — upload dataset, validasi, dan pemilihan kolom.
"""

import pandas as pd
import streamlit as st

from core.preprocessing import compute_missing_summary, count_duplicate_rows
from core.state import KEY_COLUMN_MAPPING, KEY_DF_CLEAN, KEY_DF_RAW


def _detect_default_tanggal(df: pd.DataFrame) -> str:
    datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if datetime_cols:
        return datetime_cols[0]

    name_match = [c for c in df.columns if "tanggal" in c.lower() or "date" in c.lower()]
    if name_match:
        return name_match[0]

    return df.columns[0]


def _is_numeric_like(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True
    if series.dtype != object:
        return False
    converted = pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    return converted.notna().mean() >= 0.9


st.title("Data Input")
st.write("Upload dataset radiosonde untuk memulai pipeline prediksi.")

uploaded_file = st.file_uploader(
    "Pilih file dataset",
    type=["xlsx", "csv"],
    help="Format yang didukung: XLSX dan CSV",
)

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Gagal membaca file. Pastikan format file valid. Detail: {e}")
        st.stop()

    if KEY_DF_RAW in st.session_state:
        existing = st.session_state[KEY_DF_RAW]
        if not existing.equals(df_raw):
            for key in list(st.session_state.keys()):
                if key not in [KEY_DF_RAW]:
                    del st.session_state[key]

    st.session_state[KEY_DF_RAW] = df_raw

    st.subheader("Informasi Dataset Mentah")

    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("Jumlah Baris", df_raw.shape[0])
    with col_info2:
        st.metric("Jumlah Kolom", df_raw.shape[1])
    with col_info3:
        st.metric("Total Missing Value", int(df_raw.isnull().sum().sum()))

    st.subheader("Preview Dataset (20 baris pertama)")
    preview = df_raw.head(20).copy()
    for col in preview.columns:
        if preview[col].dtype == object:
            preview[col] = preview[col].astype(str)
    st.dataframe(preview, width="stretch")

    st.subheader("Tipe Data per Kolom")
    dtype_df = pd.DataFrame({
        "Kolom": df_raw.dtypes.index,
        "Tipe Data": df_raw.dtypes.values.astype(str),
    })
    st.dataframe(dtype_df, width="stretch")

    st.subheader("Missing Value per Kolom")
    missing_summary = compute_missing_summary(df_raw)
    if missing_summary.empty:
        st.success("Tidak ada missing value di dataset ini.")
    else:
        st.dataframe(missing_summary, width="stretch")

    st.subheader("Pemilihan Kolom")

    all_columns = df_raw.columns.tolist()

    default_tanggal = _detect_default_tanggal(df_raw)
    col_tanggal = st.selectbox(
        "Kolom Tanggal",
        options=all_columns,
        index=all_columns.index(default_tanggal),
        key="sel_col_tanggal",
    )

    dup_count = count_duplicate_rows(df_raw, col_tanggal)
    if dup_count > 0:
        st.warning(
            f"Ditemukan {dup_count} baris duplikat berdasarkan kolom tanggal yang dipilih. "
            "Pengaturan penghapusan duplikat tersedia di halaman Configuration."
        )

    remaining_cols = [c for c in all_columns if c != col_tanggal]
    default_target = "CH" if "CH" in remaining_cols else remaining_cols[0]
    col_target = st.selectbox(
        "Variabel Target",
        options=remaining_cols,
        index=remaining_cols.index(default_target),
        key="sel_col_target",
    )

    numeric_like_cols = [c for c in all_columns if _is_numeric_like(df_raw[c])]
    default_predictors = [c for c in numeric_like_cols if c not in [col_tanggal, col_target]]

    predictor_options = [c for c in all_columns if c not in [col_tanggal, col_target]]
    col_predictors = st.multiselect(
        "Variabel Prediktor",
        options=predictor_options,
        default=[c for c in default_predictors if c in predictor_options],
        key="sel_col_predictors",
    )

    if col_predictors:
        st.session_state[KEY_COLUMN_MAPPING] = {
            "col_tanggal": col_tanggal,
            "col_target": col_target,
            "col_predictors": col_predictors,
        }
        if KEY_DF_CLEAN in st.session_state:
            del st.session_state[KEY_DF_CLEAN]
        st.success(
            f"Kolom berhasil dikonfigurasi. "
            f"Target: {col_target} | Prediktor: {len(col_predictors)} kolom"
        )
    else:
        st.warning("Pilih minimal satu variabel prediktor untuk melanjutkan.")

else:
    st.info("Upload file dataset untuk memulai.")
    if KEY_DF_RAW not in st.session_state:
        st.stop()
