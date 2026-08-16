"""
Konten halaman Home.
"""

import streamlit as st

st.title("Prediksi Intensitas Curah Hujan Surabaya")
st.markdown(
    "Sistem prediksi berbasis Machine Learning menggunakan "
    "Hybrid TabNet-XGBoost, Hybrid TabNet-XGBoost Hujan Lebat, dan Hybrid TabNet-SVR."
)

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Data Input")
    st.write(
        "Upload dataset radiosonde dalam format XLSX atau CSV. "
        "Sistem menampilkan informasi dataset, tipe data, missing value, "
        "dan memungkinkan pemilihan kolom tanggal, target, serta prediktor."
    )

with col2:
    st.subheader("Configuration")
    st.write(
        "Preprocessing data, feature engineering berbasis Pearson Correlation, "
        "TabNet Feature Importance, dan CCF. Konfigurasi model pipeline dan "
        "pilihan antara model tersimpan atau pelatihan ulang."
    )

with col3:
    st.subheader("Analysis & Results")
    st.write(
        "Evaluasi performa model dengan metrik RMSE, MAE, dan MAPE. "
        "Prediksi intensitas curah hujan untuk satu tanggal yang dipilih "
        "berdasarkan data historis."
    )

st.divider()
st.caption("Gunakan navigasi sidebar untuk berpindah halaman.")
