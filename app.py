import joblib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

# ==========================================
# Konfigurasi Halaman
# ==========================================
st.set_page_config(
    page_title="Aplikasi Prediksi Performa VSWR Antenna",
    page_icon="📡",
    layout="wide",
)


# ==========================================
# Load Model dan Scaler
# ==========================================
@st.cache_resource
def load_ml_components():
    # Memuat model dan scaler yang sudah ditraining sebelumnya
    model = joblib.load("model.pkl")
    try:
        scaler = joblib.load("scaler.pkl")
    except FileNotFoundError:
        scaler = None
    return model, scaler


model, scaler = load_ml_components()

# ==========================================
# Judul dan Deskripsi
# ==========================================
st.title("📡 Aplikasi Prediksi Performa VSWR Antenna")
st.markdown("""
**Aplikasi ini memprediksi kualitas antenna berdasarkan dimensi dan material.**
- **Good Antenna**: VSWR $\le$ 2 (Performa baik)
- **Poor Antenna**: VSWR > 2 (Performa buruk)
""")

st.divider()

# ==========================================
# Input User
# ==========================================
st.header("📝 Input Parameter Antenna")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Dimensi Antenna")
    length = st.number_input(
        "Length (mm)", min_value=0.0, max_value=100.0, value=45.0, step=0.01
    )
    width = st.number_input(
        "Width (mm)", min_value=0.0, max_value=100.0, value=35.0, step=0.01
    )
    height = st.number_input(
        "Height (mm)", min_value=0.0, max_value=10.0, value=1.2, step=0.01
    )

with col2:
    st.subheader("Material Properties")
    permittivity = st.number_input(
        "Permittivity (εr)", min_value=0.0, max_value=20.0, value=2.0, step=0.01
    )
    conductivity = st.number_input(
        "Conductivity (S/m)",
        min_value=0.0,
        max_value=20000.0,
        value=10000.0,
        step=100.0,
    )

# ==========================================
# Tombol Prediksi
# ==========================================
st.divider()

st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: #1e3799;
        color: white;
        border-color: #1e3799;
    }
    div.stButton > button:first-child:hover {
        background-color: #0c2461;
        color: white;
        border-color: #0c2461;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
if st.button(
    "🔮 Prediksi Performa Antenna", use_container_width=True
):

    # Siapkan data input sesuai dengan urutan fitur saat training
    features = ["Length", "Width", "Height", "Permittivity", "Conductivity"]
    input_data = pd.DataFrame(
        [[length, width, height, permittivity, conductivity]], columns=features
    )

    # Tampilkan input user
    st.subheader("📊 Data Input")
    st.dataframe(input_data.T, column_config={0: "Parameter"}, hide_index=False)

    try:
        # Proses Normalisasi Fitur menggunakan Scaler yang dimuat
        if scaler is not None:
            input_scaled = scaler.transform(input_data)
        else:
            st.warning(
                "⚠️ File 'scaler.pkl' tidak ditemukan. Prediksi dilakukan tanpa normalisasi (Hasil mungkin tidak akurat)."
            )
            input_scaled = input_data

        # Lakukan prediksi
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        # ==========================================
        # Hasil Prediksi
        # ==========================================
        st.divider()
        st.header("🎯 Hasil Prediksi")

        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.subheader("Kualitas Antenna")
            if prediction[0] == 1:
                st.success("✅ **GOOD ANTENNA**")
                st.info("VSWR $\le$ 2 - Antenna memiliki performa yang baik!")
            else:
                st.error("❌ **POOR ANTENNA**")
                st.warning(
                    "VSWR > 2 - Antenna memiliki performa yang kurang baik."
                )

        with res_col2:
            st.subheader("Probabilitas")
            prob_good = prediction_proba[0][1] * 100
            prob_poor = prediction_proba[0][0] * 100

            st.metric("Probabilitas Good", f"{prob_good:.2f}%")
            st.metric("Probabilitas Poor", f"{prob_poor:.2f}%")

            # Visualisasi probabilitas ke arah "Good"
            st.progress(prob_good / 100)
            st.caption(f"Keyakinan model: {max(prob_good, prob_poor):.2f}%")
        
      # ==========================================
        # Visualisasi Faktor Paling Berpengaruh (REMAKE DESIGN)
        # ==========================================
        st.divider()
        st.subheader("📊 Faktor Paling Berpengaruh (Feature Importances)")

        try:
            import matplotlib.pyplot as plt

            # Ambil nilai importance score dan urutkan
            importances = model.feature_importances_
            indices = np.argsort(importances)
            sorted_features = [features[i] for i in indices]
            sorted_importances = importances[indices]

            # 1. Gunakan style minimalis modern
            fig, ax = plt.subplots(figsize=(10, 4.5))

            # Buat background gambar & grafik transparan agar menyatu dengan dark/light mode Streamlit
            fig.patch.set_alpha(0.0)
            ax.set_alpha(0.0)
            ax.set_facecolor((0, 0, 0, 0))

            # 2. Plotting dengan warna biru yang lebih soft & modern (#2575fc)
            bars = ax.barh(
                sorted_features, sorted_importances, color="#2575fc", height=0.6
            )

            # 3. Merapikan garis pembatas (Spines)
            for spine in ["top", "right", "bottom"]:
                ax.spines[spine].set_visible(False)
            ax.spines["left"].set_color("#4a5568")  # Garis abu-abu gelap lembut
            ax.spines["left"].set_linewidth(1.5)

            # 4. Menambahkan label nilai di ujung bar dengan posisi yang lebih renggang & rapi
            max_val = max(sorted_importances)
            for bar in bars:
                width = bar.get_width()
                ax.text(
                    width
                    + (max_val * 0.02),  # Padding dinamis agar tidak menempel
                    bar.get_y() + bar.get_height() / 2,
                    f"{width:.2f}",
                    va="center",
                    ha="left",
                    fontsize=10,
                    fontweight="medium",
                    color="#e2e8f0",  # Warna teks putih soft (bagus untuk dark mode)
                )

            # 5. Kustomisasi teks sumbu X dan Y
            ax.set_xlabel(
                "Importance Score",
                fontsize=10,
                color="#a0aec0",
                labelpad=10,
            )
            ax.tick_params(
                axis="x", colors="#a0aec0", labelsize=9
            )  # Angka sumbu X lebih kecil
            ax.tick_params(
                axis="y", colors="#e2e8f0", labelsize=11, pad=8
            )  # Nama fitur lebih jelas

            # Berikan ruang ekstra di sebelah kanan agar angka tidak terpotong
            ax.set_xlim(0, max_val * 1.15)

            # Tampilkan ke Streamlit dengan parameter transparan diaktifkan
            st.pyplot(fig, clear_figure=True, use_container_width=True)

        except AttributeError:
            st.info(
                "💡 Informasi: Model aktif saat ini tidak menggunakan basis pohon keputusan (tree-based), "
                "sehingga representasi grafik 'Feature Importance' tidak dapat ditampilkan."
            )

      
        # ==========================================
        # Rekomendasi
        # ==========================================
        st.divider()
        st.header("💡 Rekomendasi")

        if prediction[0] == 0:
            st.markdown("""
            **Antenna diprediksi memiliki performa yang kurang baik. Pertimbangkan untuk:**
            - Menyesuaikan dimensi antenna (*Length, Width, Height*).
            - Mengubah material dengan *permittivity* yang lebih sesuai.
            - Meningkatkan tingkat *conductivity* pada material.
            - Melakukan simulasi ulang secara terukur dengan parameter komparatif baru.
            """)
        else:
            st.markdown("""
            **Antenna diprediksi memiliki performa yang baik!**
            - Parameter kombinasi yang dimasukkan dinilai sudah optimal oleh model.
            - Desain dapat dipertimbangkan untuk lanjut ke tahap fabrikasi prototipe.
            - Tetap direkomendasikan melakukan pengujian fisik (*laboratory testing*) untuk validasi final.
            """)

        # ==========================================
        # Informasi Teknis
        # ==========================================
        with st.expander("📚 Informasi Teknis VSWR"):
            st.markdown("""
            **VSWR (Voltage Standing Wave Ratio)** adalah indeks ukuran efisiensi transmisi daya dari saluran transmisi ke antenna.
            
            - **VSWR = 1**: *Perfect match* (Seluruh daya terpancar sempurna tanpa refleksi).
            - **VSWR $\le$ 2**: *Good match* (Efisiensi transmisi daya > 90%, standar aman industri).
            - **VSWR > 2**: *Poor match* (Daya mengalami banyak pantulan kembali ke sumber, risiko *overheating* komponen).
            
            **Sistem Labelisasi Target:**
            - Kelas 1 (*Good Antenna*): VSWR $\le$ 2
            - Kelas 0 (*Poor Antenna*): VSWR > 2
            """)

    except Exception as e:
        st.error(f"⚠️ Terjadi kesalahan sistem: {str(e)}")
        st.info(
            "Pastikan arsitektur file 'model.pkl' dan 'scaler.pkl' cocok dengan dimensi input data saat ini."
        )

# ==========================================
# Sidebar Informasi
# ==========================================
with st.sidebar:
    st.header("ℹ️ Tentang Aplikasi")
    st.markdown("""
    Sistem berbasis Machine Learning ini memprediksi kecenderungan performa VSWR struktur antena mikrostrip/padat.
    
    **Fitur Input:**
    - **Dimensi**: *Length, Width, Height*
    - **Material**: *Permittivity, Conductivity*
    
    **Alur Kerja Sistem:**
    1. Konfigurasi nilai parameter pada panel utama.
    2. Klik tombol **Prediksi Performa Antenna**.
    3. Sistem melakukan standarisasi z-score lalu mengevaluasi probabilitas via model terpilih.
    """)

    st.divider()

    st.header("📈 Nilai Referensi Umum")
    st.markdown("""
    **Dimensi Standar (mm):**
    - Length: 30.0 - 50.0
    - Width: 30.0 - 40.0
    - Height: 0.8 - 1.6
    
    **Karakteristik Material:**
    - Permittivity ($\epsilon_r$): 1.5 - 2.5
    - Conductivity: 3000 - 15000 S/m
    """)

# ==========================================
# Footer
# ==========================================
st.divider()
st.caption(
    "Aplikasi Prediksi Performa VSWR Antenna | Dikembangkan dengan Streamlit | © 2026"
)