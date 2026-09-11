
import os
import io
import glob
import json
import time
import random
import numpy as np
import pandas as pd
from PIL import Image
from typing import Optional, Tuple

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import tensorflow as tf
from tensorflow import keras

# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------
KAGGLE_URL   = ("https://www.kaggle.com/datasets/kipshidze/"
                "apple-vs-orange-binary-classification"
                "?resource=download&select=fruit-dataset")
DATASET_ROOT = "c:/week2"                          
APPLE_DIR    = os.path.join(DATASET_ROOT, "apple")
ORANGE_DIR   = os.path.join(DATASET_ROOT, "orange")
MODEL_PATHS  = {
    "MobileNetV2 (Transfer Learning)": "models/apple_orange_mobilenetv2.keras",
    "Scratch CNN (4-Layer Custom)":     "models/apple_orange_scratch_cnn.keras",
}
BEST_MODEL_KEY = "MobileNetV2 (Transfer Learning)"
IMG_SIZE     = (150, 150)
CLASSES      = ["Apple", "Orange"]

N_APPLE  = len(os.listdir(APPLE_DIR))  if os.path.exists(APPLE_DIR)  else 396
N_ORANGE = len(os.listdir(ORANGE_DIR)) if os.path.exists(ORANGE_DIR) else 400
N_TOTAL  = N_APPLE + N_ORANGE

st.set_page_config(
    page_title="Fruit Classifier Deployment App",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --bg:          #F8FAFC;
    --surface:     #FFFFFF;
    --border:      #E2E8F0;
    --text:        #0F172A;
    --text-muted:  #64748B;
    --apple:       #DC2626;
    --apple-soft:  #FEF2F2;
    --orange:      #EA580C;
    --orange-soft: #FFEDD5;
    --teal:        #0D9488;
    --teal-soft:   #CCFBF1;
    --indigo:      #4F46E5;
    --indigo-soft: #EEF2FF;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: var(--text);
}

.main { background: var(--bg); }
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

.grad-title {
    background: linear-gradient(90deg, var(--apple), var(--orange));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-weight: 800; font-size: 2.5rem; line-height: 1.2; margin-bottom: 4px;
}
.grad-sub {
    color: var(--text-muted);
    font-weight: 600; font-size: 1.15rem; margin-bottom: 20px;
}

.champ-card {
    background: linear-gradient(135deg, #FEF2F2 0%, #FFEDD5 100%);
    border: 2px solid #FDBA74;
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 10px 25px rgba(234, 88, 12, 0.08);
}

.pred-class { font-size: 2.2rem; font-weight: 800; color: var(--text); }
.pred-conf  { font-size: 1rem; color: var(--text-muted); font-weight: 600; margin-top: 4px; }

.badge {
    display: inline-block; padding: 6px 16px; border-radius: 999px;
    font-size: .85rem; font-weight: 700; letter-spacing: .3px;
}
.b-apple  { background: var(--apple-soft);  color: var(--apple);  border: 1px solid var(--apple); }
.b-orange { background: var(--orange-soft); color: var(--orange); border: 1px solid var(--orange); }
.b-teal   { background: var(--teal-soft);   color: var(--teal);   border: 1px solid var(--teal); }
.b-indigo { background: var(--indigo-soft); color: var(--indigo); border: 1px solid var(--indigo); }

div[data-testid="metric-container"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px; padding: 16px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.02);
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Memuat model terbaik...")
def load_models() -> dict:
    out = {}
    for name, path in MODEL_PATHS.items():
        if os.path.exists(path):
            try:
                out[name] = keras.models.load_model(path)
            except Exception as e:
                st.error(f"Gagal memuat {name}: {e}")
    return out

@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    path = "artifacts/vision_metrics.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_data(show_spinner=False)
def get_sample_images(cls: str, n: int = 9, seed: int = 42):
    folder = "samples"
    paths  = sorted(glob.glob(os.path.join(folder, f"{cls}_*.jpg")))
    if not paths:
        src = APPLE_DIR if cls == "apple" else ORANGE_DIR
        paths = sorted(glob.glob(os.path.join(src, "*.jpeg")) +
                       glob.glob(os.path.join(src, "*.jpg")))
    random.seed(seed)
    chosen = random.sample(paths, min(n, len(paths)))
    return [Image.open(p).convert("RGB") for p in chosen], chosen

def predict(model, pil_img: Image.Image) -> Tuple[str, float, float, float, float]:
    arr = np.expand_dims(np.array(pil_img.resize(IMG_SIZE)), axis=0).astype(np.float32)
    t0  = time.perf_counter()
    raw = float(model.predict(arr, verbose=0)[0][0])
    lat = (time.perf_counter() - t0) * 1000

    orange_p = raw
    apple_p  = 1.0 - raw
    label    = "Orange" if orange_p >= 0.5 else "Apple"
    conf     = orange_p if orange_p >= 0.5 else apple_p
    return label, conf, apple_p, orange_p, lat


models  = load_models()
metrics = load_metrics()

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 12px 0;'>
        <div style='font-weight:800;font-size:1.3rem;color:var(--text);'>Fruit Classifier</div>
        <div style='font-size:.8rem;color:var(--text-muted);margin-top:2px;'>Rafi Ikbar Fahrezy</div>
    </div>
    <hr style='border-color:var(--border);margin:10px 0;'>
    """, unsafe_allow_html=True)

    page = st.radio("Navigasi Aplikasi:", [
        "Dashboard Model Terbaik",
        "Prediksi Gambar Tunggal",
        "Prediksi Massal (CSV)",
        "Jelajahi Dataset",
        "Komparasi Model & Riwayat",
    ])

    st.markdown("<hr style='border-color:var(--border);margin:16px 0;'>", unsafe_allow_html=True)

    st.markdown("### Model Champion")
    st.markdown("""
    <div style='background:var(--indigo-soft);border:1px solid var(--indigo);border-radius:12px;padding:12px;text-align:center;'>
        <div style='font-size:.85rem;font-weight:800;color:var(--indigo);'>MobileNetV2 (Transfer Learning)</div>
        <div style='font-size:1.4rem;font-weight:800;color:var(--text);margin-top:4px;'>Akurasi: 91.82%</div>
        <div style='font-size:.75rem;color:var(--text-muted);'>Terbaik & Paling Presisi</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    conf_warn = st.slider("Ambang Batas Keyakinan Low:", .50, .95, .72, .01)
    show_lat  = st.toggle("Tampilkan Latensi Inferensi", value=True)

# HALAMAN 1  DASHBOARD MODEL TERBAIK
if page == "Dashboard Model Terbaik":
    st.markdown('<div class="grad-title">Apple vs Orange Classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="grad-sub">Dashboard Performa Model Terbaik & Analisis Dataset</div>', unsafe_allow_html=True)

    mob_acc  = metrics.get("mobilenetv2", {}).get("accuracy", 0.9182) * 100
    cnn_acc  = metrics.get("scratch_cnn", {}).get("accuracy", 0.8616) * 100
    mob_time = metrics.get("mobilenetv2", {}).get("train_time_sec", 48.1)
    cnn_time = metrics.get("scratch_cnn", {}).get("train_time_sec", 31.6)

    st.markdown(f"""
    <div class="champ-card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <span class="badge b-orange">MODEL TERBAIK & UTAMA</span>
                <h2 style="margin:8px 0 4px 0; color:#9A3412; font-weight:800;">MobileNetV2 Transfer Learning</h2>
            </div>
            <div style="text-align:right; min-width:180px;">
                <div style="font-size:2.8rem; font-weight:800; color:#9A3412;">{mob_acc:.2f}%</div>
                <div style="font-size:.85rem; font-weight:700; color:#C2410C;">AKURASI VALIDASI</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Akurasi Best Model", f"{mob_acc:.2f}%", "+5.66% vs Baseline")
    k2.metric("Total Dataset Gambar", f"{N_TOTAL:,}", f"{N_APPLE} Apple | {N_ORANGE} Orange")
    k3.metric("Kecepatan Inferensi", "~15.2 ms", "Real-Time Response")
    k4.metric("Resolution Input", "150 x 150 px", "RGB 3-Channel")

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.2, 1], gap="large")

    with col_left:
        st.markdown("### Perbandingan Performa Model (Baseline vs Champion)")
        df_acc = pd.DataFrame({
            "Model": ["MobileNetV2 (Champion)", "Scratch CNN (Baseline)"],
            "Akurasi (%)": [mob_acc, cnn_acc],
            "Tipe": ["Transfer Learning", "Custom 4-Layer"]
        })
        fig = px.bar(
            df_acc, x="Model", y="Akurasi (%)", color="Model",
            color_discrete_sequence=["#EA580C", "#94A3B8"],
            text_auto=".2f", range_y=[0, 100],
            title="Akurasi Validasi Model (%)"
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#0F172A", showlegend=False,
            yaxis_title="Akurasi Validasi (%)", xaxis_title="",
            margin=dict(t=40, b=20, l=20, r=20)
        )
        fig.update_traces(textfont_size=14, textfont_color="#0F172A", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("### Distribusi Data Kelas Biner")
        fig2 = px.pie(
            values=[N_APPLE, N_ORANGE],
            names=["Apple (Apel)", "Orange (Jeruk)"],
            color_discrete_sequence=["#DC2626", "#EA580C"],
            hole=0.5,
            title="Keseimbangan Data Latih"
        )
        fig2.update_traces(textinfo="percent+label", textfont_color="#0F172A")
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font_color="#0F172A",
            annotations=[dict(text=f"<b>{N_TOTAL}</b><br>Gambar", x=.5, y=.5, showarrow=False, font=dict(size=18, color="#0F172A"))],
            margin=dict(t=40, b=20, l=20, r=20)
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    st.markdown("### Spesifikasi Teknis Model Terbaik (MobileNetV2)")
    spec_df = pd.DataFrame([
        {"Komponen": "Model Backbone", "Konfigurasi": "MobileNetV2 (Pretrained on ImageNet)"},
        {"Komponen": "Metode Pelatihan", "Konfigurasi": "Two-Stage Transfer Learning (Feature Extraction + Fine-Tuning)"},
        {"Komponen": "Layer Tambahan", "Konfigurasi": "GlobalAveragePooling2D -> Dropout (0.3) -> Dense (1, Sigmoid)"},
        {"Komponen": "Optimizer", "Konfigurasi": "Adam (Learning Rate: 1e-3 untuk awal, 1e-5 untuk Fine-Tuning)"},
        {"Komponen": "Fungsi Loss", "Konfigurasi": "Binary Crossentropy"},
        {"Komponen": "Augmentasi Data", "Konfigurasi": "Random Rotation (20 deg), Zoom (0.15), Shift, Horizontal Flip"}
    ])
    st.dataframe(spec_df, use_container_width=True, hide_index=True)

# HALAMAN 2 - PREDIKSI GAMBAR TUNGGAL
elif page == "Prediksi Gambar Tunggal":
    st.markdown('<div class="grad-title">Prediksi Gambar Tunggal</div>', unsafe_allow_html=True)
    st.markdown(f"<p style='color:var(--text-muted);'>Uji klasifikasi gambar buah secara langsung dengan <b>MobileNetV2 Champion Model (Akurasi 91.82%)</b></p>", unsafe_allow_html=True)

    c_left, c_right = st.columns([1, 1], gap="large")

    with c_left:
        st.markdown("### Pilih Gambar Input")
        method = st.radio("Metode input:", ["Upload Gambar", "Preset Dataset", "Kamera"], horizontal=True)
        img_input: Optional[Image.Image] = None

        if method == "Upload Gambar":
            upl = st.file_uploader("Upload file (JPG / PNG):", type=["jpg","jpeg","png"])
            if upl:
                img_input = Image.open(upl).convert("RGB")

        elif method == "Preset Dataset":
            cls_pick = st.radio("Pilih kelas:", ["Apple", "Orange"], horizontal=True)
            cls_key  = "apple" if "Apple" in cls_pick else "orange"
            imgs_p, paths_p = get_sample_images(cls_key, n=12)

            cols_prev = st.columns(4)
            for i, im in enumerate(imgs_p[:8]):
                cols_prev[i % 4].image(im, use_container_width=True)

            idx = st.select_slider("Pilih sampel:", options=list(range(1, len(imgs_p)+1)), format_func=lambda x: f"#{x}")
            img_input = imgs_p[idx - 1]
            st.caption(f"Sumber: Dataset Kaggle - kelas `{cls_key}` - `{os.path.basename(paths_p[idx-1])}`")

        elif method == "Kamera":
            cam = st.camera_input("Ambil foto:")
            if cam:
                img_input = Image.open(cam).convert("RGB")

        if img_input is not None:
            st.image(img_input, caption="Gambar input yang dipilih", use_container_width=True)

    with c_right:
        st.markdown("### Hasil Prediksi Model Terbaik")

        if img_input is None:
            st.info("Upload gambar, pilih preset, atau gunakan kamera untuk mulai inferensi.")
        else:
            best_model = models.get(BEST_MODEL_KEY)
            if not best_model:
                st.error("Model MobileNetV2 tidak ditemukan di folder `models/`.")
            else:
                label, conf, apple_p, orange_p, lat = predict(best_model, img_input)

                is_apple  = "Apple" in label
                grad_css  = ("rgba(220,38,38,.12),rgba(220,38,38,.02)" if is_apple else "rgba(234,88,12,.12),rgba(234,88,12,.02)")
                border_c  = "#DC2626" if is_apple else "#EA580C"
                conf_cls  = " conf-low" if conf < conf_warn else ""

                st.markdown(f"""
                <div style="background:linear-gradient(135deg,{grad_css}); border:2px solid {border_c}; border-radius:18px; padding:24px; text-align:center; margin-bottom:16px;">
                    <span class="badge {'b-apple' if is_apple else 'b-orange'}">HASIL PREDIKSI</span>
                    <div class="pred-class" style="margin-top:8px;">{label}</div>
                    <div class="pred-conf{conf_cls}">Tingkat Keyakinan (Confidence): {conf*100:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

                if conf < conf_warn:
                    st.warning(f"Tingkat keyakinan rendah (< {conf_warn*100:.0f}%). Coba gambar yang lebih jelas/bersih.")

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=["Apple (Apel)", "Orange (Jeruk)"],
                    y=[apple_p * 100, orange_p * 100],
                    marker_color=["#DC2626", "#EA580C"],
                    text=[f"{apple_p*100:.2f}%", f"{orange_p*100:.2f}%"],
                    textposition="outside",
                    textfont_color="#0F172A",
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#0F172A", showlegend=False,
                    yaxis=dict(range=[0, 115], title="Probabilitas (%)"),
                    xaxis_title="Kelas Buah",
                    title="Rincian Probabilitas Model Champion",
                    margin=dict(t=40, b=20, l=20, r=20),
                )
                st.plotly_chart(fig, use_container_width=True)

                if show_lat:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Waktu Inferensi", f"{lat:.1f} ms")
                    m2.metric("Resolusi Input", "150 x 150 px")
                    m3.metric("Model Champion", "MobileNetV2")

# HALAMAN 3 - PREDIKSI MASSAL (CSV)
elif page == "Prediksi Massal (CSV)":
    st.markdown('<div class="grad-title">Prediksi Massal</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:var(--text-muted);'>Upload banyak gambar sekaligus untuk diprediksi secara otomatis menggunakan <b>MobileNetV2 Best Model</b> dan unduh hasilnya sebagai file CSV.</p>", unsafe_allow_html=True)

    best_model = models.get(BEST_MODEL_KEY)

    uploaded_batch = st.file_uploader(
        "Upload gambar (bisa banyak sekaligus):",
        type=["jpg","jpeg","png"],
        accept_multiple_files=True,
    )

    if uploaded_batch:
        st.markdown(f"**{len(uploaded_batch)} gambar dipilih.** Klik tombol di bawah untuk memulai prediksi.")

        if st.button("Jalankan Prediksi Massal", type="primary", use_container_width=True):
            if not best_model:
                st.error("Model MobileNetV2 tidak ditemukan.")
            else:
                results = []
                prog = st.progress(0.0, text="Memproses...")

                for i, f in enumerate(uploaded_batch):
                    pil = Image.open(f).convert("RGB")
                    label, conf, ap, op, lat = predict(best_model, pil)
                    results.append({
                        "No":                       i + 1,
                        "Nama File":                f.name,
                        "Prediksi":                 label,
                        "Tingkat Keyakinan (%)":    round(conf * 100, 2),
                        "Probabilitas Apple (%)":   round(ap  * 100, 2),
                        "Probabilitas Orange (%)":  round(op  * 100, 2),
                        "Latensi (ms)":             round(lat, 2),
                        "Keyakinan Rendah":         "Ya" if conf < conf_warn else "Tidak",
                    })
                    prog.progress((i + 1) / len(uploaded_batch), text=f"Memproses {i+1}/{len(uploaded_batch)} - {f.name}")

                prog.empty()
                df_res = pd.DataFrame(results)

                n_apple  = (df_res["Prediksi"].str.contains("Apple")).sum()
                n_orange = len(df_res) - n_apple
                avg_conf = df_res["Tingkat Keyakinan (%)"].mean()
                n_low    = (df_res["Keyakinan Rendah"] == "Ya").sum()

                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Diprediksi Apple",  n_apple)
                s2.metric("Diprediksi Orange", n_orange)
                s3.metric("Rata-rata Keyakinan", f"{avg_conf:.2f}%")
                s4.metric("Keyakinan Rendah",  n_low)

                st.markdown("#### Preview (maks 8 gambar)")
                gcols = st.columns(min(4, len(uploaded_batch)))
                for i, f in enumerate(uploaded_batch[:8]):
                    f.seek(0)
                    gcols[i % 4].image(
                        Image.open(f),
                        caption=f"{results[i]['Prediksi']} ({results[i]['Tingkat Keyakinan (%)']:.1f}%)",
                        use_container_width=True,
                    )

                st.markdown("#### Tabel Hasil Prediksi")
                st.dataframe(df_res, use_container_width=True, hide_index=True)

                fig_dist = px.pie(df_res, names="Prediksi", color_discrete_sequence=["#DC2626","#EA580C"], title="Distribusi Hasil Prediksi Massal")
                fig_dist.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#0F172A")
                st.plotly_chart(fig_dist, use_container_width=True)

                buf = io.BytesIO()
                df_res.to_csv(buf, index=False, encoding="utf-8-sig")
                st.download_button(
                    "Download Hasil Prediksi (CSV)",
                    data=buf.getvalue(),
                    file_name="hasil_prediksi_massal_mobilenetv2.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# HALAMAN 4 - JELAJAHI DATASET
elif page == "Jelajahi Dataset":
    st.markdown('<div class="grad-title">Jelajahi Dataset</div>', unsafe_allow_html=True)
    st.markdown(f"<p style='color:var(--text-muted);'>Eksplorasi dataset buah dari Kaggle (Apple vs Orange Binary Classification).</p>", unsafe_allow_html=True)

    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Apple",      f"{N_APPLE:,}")
    d2.metric("Orange",     f"{N_ORANGE:,}")
    d3.metric("Total Gambar", f"{N_TOTAL:,}")
    d4.metric("Resolusi",   "150 x 150 px")
    d5.metric("Format",     "JPEG / RGB")

    c_pie, c_bar = st.columns(2)
    with c_pie:
        fig_p = px.pie(values=[N_APPLE, N_ORANGE], names=["Apple","Orange"], color_discrete_sequence=["#DC2626","#EA580C"], hole=.4, title="Distribusi Kelas Dataset")
        fig_p.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#0F172A")
        st.plotly_chart(fig_p, use_container_width=True)
    with c_bar:
        split_df = pd.DataFrame({"Split": ["Train (80%)", "Validasi (20%)"], "Jumlah": [int(N_TOTAL*.8), int(N_TOTAL*.2)]})
        fig_b = px.bar(split_df, x="Split", y="Jumlah", color="Split", color_discrete_sequence=["#DC2626","#0D9488"], text_auto=True, title="Pembagian Data Train / Validasi")
        fig_b.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#0F172A", showlegend=False)
        st.plotly_chart(fig_b, use_container_width=True)

    st.markdown("### Galeri Sampel Dataset")
    n_show = st.slider("Jumlah sampel per kelas:", 3, 12, 6, 3)

    tab_a, tab_o, tab_mix = st.tabs(["Apple", "Orange", "Campuran"])
    def show_grid(images, paths, ncols=3):
        cols = st.columns(ncols)
        for i, (img, path) in enumerate(zip(images, paths)):
            cols[i % ncols].image(img, caption=os.path.basename(path), use_container_width=True)

    with tab_a:
        imgs_a, paths_a = get_sample_images("apple", n=n_show, seed=11)
        show_grid(imgs_a, paths_a)

    with tab_o:
        imgs_o, paths_o = get_sample_images("orange", n=n_show, seed=22)
        show_grid(imgs_o, paths_o)

    with tab_mix:
        ia, pa = get_sample_images("apple",  n=n_show//2, seed=33)
        io_, po = get_sample_images("orange", n=n_show//2, seed=44)
        combined = list(zip(ia + io_, pa + po))
        random.shuffle(combined)
        show_grid([c[0] for c in combined], [c[1] for c in combined])

# HALAMAN 5 - KOMPARASI MODEL & RIWAYAT
elif page == "Komparasi Model & Riwayat":
    st.markdown('<div class="grad-title">Komparasi Model & Riwayat Versi</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:var(--text-muted);'>Analisis komparatif arsitektur model dan riwayat iterasi pengembangan.</p>", unsafe_allow_html=True)

    mob_acc  = metrics.get("mobilenetv2", {}).get("accuracy", 0.9182) * 100
    cnn_acc  = metrics.get("scratch_cnn", {}).get("accuracy", 0.8616) * 100
    mob_time = metrics.get("mobilenetv2", {}).get("train_time_sec", 48.1)
    cnn_time = metrics.get("scratch_cnn", {}).get("train_time_sec", 31.6)

    h1, h2 = st.columns(2)
    with h1:
        st.markdown(f"""
        <div style="background:var(--surface); border:2px solid #EA580C; border-radius:18px; padding:24px; text-align:center;">
            <span class="badge b-orange">MODEL CHAMPION TERBAIK</span>
            <h3 style="color:#EA580C; margin:8px 0;">MobileNetV2 (Transfer Learning)</h3>
            <p style="color:var(--text-muted); font-size:.85rem;">Pretrained ImageNet + Fine-Tuning</p>
            <div style="font-size:2.8rem; font-weight:800; color:var(--text);">{mob_acc:.2f}%</div>
            <div style="color:var(--text-muted); font-size:.9rem; font-weight:700;">AKURASI VALIDASI</div>
            <br>
            <span class="badge b-teal">Training Time: {mob_time:.1f}s</span>
        </div>
        """, unsafe_allow_html=True)

    with h2:
        st.markdown(f"""
        <div style="background:var(--surface); border:1px solid var(--border); border-radius:18px; padding:24px; text-align:center;">
            <span class="badge b-indigo">BASELINE MODEL</span>
            <h3 style="color:var(--indigo); margin:8px 0;">Scratch CNN (4-Layer Custom)</h3>
            <p style="color:var(--text-muted); font-size:.85rem;">Dilatih dari nol (Custom Conv2D)</p>
            <div style="font-size:2.8rem; font-weight:800; color:var(--text);">{cnn_acc:.2f}%</div>
            <div style="color:var(--text-muted); font-size:.9rem; font-weight:700;">AKURASI VALIDASI</div>
            <br>
            <span class="badge b-teal">Training Time: {cnn_time:.1f}s</span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### Riwayat Changelog & Iterasi")
    st.markdown(f"""
    - **v3.0.0 (Best Model Deployment)**: Implementasi MobileNetV2 Transfer Learning sebagai model utama (Akurasi **{mob_acc:.2f}%**), integrasi dashboard analytics modern, prediksi masal CSV, dan galeri dataset.
    - **v2.0.0 (Custom CNN)**: Pelatihan 4-Layer Scratch CNN dari nol (Akurasi **{cnn_acc:.2f}%**).
    - **v1.0.0 (Initial Setup)**: Setup dataset biner Kaggle Apple vs Orange ({N_TOTAL} gambar).
    """)
