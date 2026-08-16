"""
Entry point aplikasi — mendefinisikan navigasi dan konfigurasi halaman global.
"""

import streamlit as st

st.set_page_config(
    page_title="Prediksi Curah Hujan Surabaya",
    layout="wide",
    initial_sidebar_state="expanded",
)

pg = st.navigation([
    st.Page("home.py", title="Home", default=True),
    st.Page("pages/1_data_input.py", title="Data Input"),
    st.Page("pages/2_configuration.py", title="Configuration"),
    st.Page("pages/3_analysis_results.py", title="Analysis & Results"),
])

pg.run()
