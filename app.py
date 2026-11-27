import altair as alt
import pandas as pd
import streamlit as st

import logic

st.set_page_config(page_title="Cek Kewajaran Harga Mobil Bekas", page_icon="🚗", layout="wide")


# ---------- Muat model ----------
@st.cache_resource
def muat_model():
    return logic.muat_semua()


@st.cache_data
def muat_hasil():
    return pd.read_csv(logic.FOLDER / "penilaian_kewajaran.csv")


art, aturan, ref = muat_model()


# ---------- Fungsi bantu tampilan ----------
def rupiah(v):
    """Rp ... juta dengan titik sebagai pemisah ribuan (contoh: Rp1.808 juta)."""
    return "Rp" + f"{v / 1e6:,.0f}".replace(",", ".") + " juta"


def nama_model(m):
    if "(lainnya)" in m:
        return "Model lain (data sedikit)"
    return m.title()


def nama_tipe(t):
    return t.upper() if len(t) <= 3 else t.title()


def grafik_rentang(prediksi, bawah, atas, harga=None):
    """Batang abu = kisaran wajar, garis biru = prediksi, garis merah putus-putus = harga listing."""
    skala = alt.Scale(zero=False, nice=True)
    sumbu = alt.Axis(title="Harga (juta Rp)")
    rentang = pd.DataFrame({"a": [bawah / 1e6], "b": [atas / 1e6], "y": ["harga"]})
    dasar = alt.Chart(rentang).mark_bar(height=26, color="#4B5565").encode(
        x=alt.X("a:Q", scale=skala, axis=sumbu), x2="b:Q", y=alt.Y("y:N", axis=None))
    garis = alt.Chart(pd.DataFrame({"x": [prediksi / 1e6]})).mark_rule(color="#4FA3E0", size=3).encode(
        x=alt.X("x:Q", scale=skala))
    lapisan = dasar + garis
    if harga:
        merah = alt.Chart(pd.DataFrame({"x": [harga / 1e6]})).mark_rule(
            color="#FF6B5E", size=3, strokeDash=[5, 4]).encode(x=alt.X("x:Q", scale=skala))
        lapisan = lapisan + merah
    return lapisan.properties(height=90, width="container")


# ---------- Header ----------
st.title("Cek Kewajaran Harga Mobil Bekas")
st.caption("Tugas Akhir Sains Data · Kelompok 5 · Model Random Forest dilatih dari listing Carmudi "
            "(diambil 2 September 2026)")

tab_cek, tab_jelajah, tab_model = st.tabs(["Cek harga", "Jelajahi hasil", "Tentang model"])

# =====================================================================
# TAB 1 - CEK HARGA
# =====================================================================
with tab_cek:
    kiri, kanan = st.columns([1, 1.2], gap="large")

    with kiri:
        st.subheader("Data mobil")
        merek = st.selectbox("Merek", sorted(ref["merek"].unique()), format_func=lambda m: m.title())
        ref_m = ref[ref["merek"] == merek]
        model_grup = st.selectbox("Model", ref_m["model_grup"].tolist(), format_func=nama_model)
        baris = ref_m[ref_m["model_grup"] == model_grup].iloc[0]

        c1, c2 = st.columns(2)
        tahun = c1.number_input("Tahun produksi", min_value=int(ref["tahun_min"].min()),
                                max_value=logic.TAHUN_DATA, value=min(2020, logic.TAHUN_DATA), step=1)
        km = c2.number_input("Kilometer", min_value=0, max_value=400_000, value=50_000, step=5_000)

        opsi_tipe = list(art["ohe"].categories_[1])  # urutan kolom_ohe: merek, tipe_bodi, lokasi
        idx = opsi_tipe.index(baris["tipe_bodi"]) if baris["tipe_bodi"] in opsi_tipe else 0
        tipe = st.selectbox("Tipe bodi", opsi_tipe, index=idx, format_func=nama_tipe,
                            key=f"tipe_{model_grup}")  # key per model: default ikut model yang dipilih

        st.divider()
        ada_harga = st.toggle("Nilai harga listing", value=True)
        harga_juta = st.number_input("Harga listing (juta Rp)", min_value=10.0, value=250.0, step=5.0,
                                        disabled=not ada_harga)

        if not (baris["tahun_min"] <= tahun <= baris["tahun_maks"]):
            st.warning(f"Data latih untuk model ini hanya mencakup tahun {int(baris['tahun_min'])}"
                        f"-{int(baris['tahun_maks'])}. Prediksi di luar rentang itu kurang andal.")

    # ----- hitung -----
    prediksi = logic.prediksi_harga(art, merek, model_grup, int(tahun), km, tipe)
    harga = harga_juta * 1e6 if ada_harga else None
    hasil = logic.nilai_kewajaran(aturan, prediksi, harga)

    with kanan:
        st.subheader("Hasil")
        st.metric("Perkiraan harga menurut model", rupiah(prediksi))
        st.write(f"Kisaran wajar: **{rupiah(hasil['harga_wajar_bawah'])} - {rupiah(hasil['harga_wajar_atas'])}**")
        st.caption(f"Segmen {hasil['segmen']} · rasio wajar {hasil['batas_bawah']:.2f}-{hasil['batas_atas']:.2f} "
                    f"· tingkat kepastian {hasil['kepastian']}")
        st.altair_chart(grafik_rentang(prediksi, hasil["harga_wajar_bawah"], hasil["harga_wajar_atas"], harga))
        st.caption("Garis biru = harga prediksi · batang abu = kisaran wajar"
                    + (" · garis merah putus-putus = harga listing" if harga else ""))

        if harga:
            warna = {"di bawah wajar": "red", "wajar": "gray", "di atas wajar": "blue"}[hasil["label"]]
            penjelasan = {
                "di bawah wajar": "Harga listing lebih murah dari kisaran wajar menurut model. Periksa kondisi fisik, "
                                    "kelengkapan dokumen, dan verifikasi penjual sebelum memutuskan.",
                "wajar": "Harga listing berada dalam kisaran wajar menurut model.",
                "di atas wajar": "Harga listing lebih mahal dari kisaran wajar menurut model. Ada ruang untuk menawar, "
                                    "atau cek apakah ada kelebihan yang tidak tertangkap data.",
            }[hasil["label"]]
            st.markdown(f"### :{warna}[{hasil['label'].capitalize()}]")
            st.metric("Rasio harga (listing ÷ prediksi)", f"{hasil['rasio']:.2f}")
            st.write(penjelasan)
        if hasil["kepastian"] == "rendah":
            st.info("Segmen ini berkepastian rendah (akurasi model lebih rendah di harga tinggi). "
                    "Anggap hasilnya sebagai indikasi awal, bukan keputusan.")

# =====================================================================
# TAB 2 - JELAJAHI HASIL
# =====================================================================
with tab_jelajah:
    df = muat_hasil()
    st.write("Hasil penilaian untuk seluruh listing di notebook. Data latih memakai prediksi *out-of-fold*, "
                "jadi tiap listing dibandingkan dengan model yang tidak pernah melihatnya.")
    f1, f2, f3 = st.columns(3)
    pilih_split = f1.multiselect("Data", sorted(df["split"].unique()), default=sorted(df["split"].unique()))
    pilih_merek = f2.multiselect("Merek", sorted(df["merek"].unique()), default=sorted(df["merek"].unique()),
                                    format_func=lambda m: m.title())
    urutan_label = ["di bawah wajar", "wajar", "di atas wajar"]
    pilih_label = f3.multiselect("Label", urutan_label, default=urutan_label)

    f = df[df["split"].isin(pilih_split) & df["merek"].isin(pilih_merek) & df["label"].isin(pilih_label)]
    m0, m1, m2, m3 = st.columns(4)
    m0.metric("Jumlah listing", f"{len(f):,}".replace(",", "."))
    for kolom, nama in zip((m1, m2, m3), urutan_label):
        n = int((f["label"] == nama).sum())
        persen = n / len(f) * 100 if len(f) else 0
        kolom.metric(nama.capitalize(), f"{n:,}".replace(",", "."), f"{persen:.1f}%", delta_color="off")

    tampil = pd.DataFrame({
        "Listing": f["judul"], "Merek": f["merek"].str.title(), "Model": f["model_grup"],
        "Tahun": f["tahun_produksi"], "KM": f["km_rata2"].where(~f["km_kosong"]).round(0),
        "Tipe bodi": f["tipe_bodi"], "Data": f["split"],
        "Harga (jt)": (f["harga"] / 1e6).round(0), "Prediksi (jt)": (f["prediksi"] / 1e6).round(0),
        "Rasio": f["rasio"].round(2), "Label": f["label"], "Kepastian": f["kepastian"], "Segmen": f["segmen"],
    })
    st.dataframe(tampil, hide_index=True)

# =====================================================================
# TAB 3 - TENTANG MODEL
# =====================================================================
with tab_model:
    st.subheader("Seberapa akurat model ini?")
    st.caption("Hasil pada 882 listing data uji yang belum pernah dilihat model.")
    a, b, c, d = st.columns(4)
    a.metric("R² log", "0,880")
    b.metric("MAPE", "10,3%")
    c.metric("Median error", "4,7%")
    d.metric("Dalam ±15%", "81,2%")
    st.write("Target proyek R² ≥ 0,75 **terpenuhi**; interval kepercayaan 95% (bootstrap 1.000×) untuk R² log "
                "adalah 0,822-0,921.")

    st.subheader("Cara kerja")
    st.markdown(
        "1. Mobil yang Anda isi diubah menjadi 16 fitur (umur, kilometer, merek, model, tipe bodi).\n"
        "2. Model Random Forest memprediksi harga (dihitung di skala log, lalu dikembalikan ke rupiah).\n"
        "3. Rasio = harga listing ÷ harga prediksi.\n"
        "4. Ambang wajar dihitung **per segmen harga prediksi**: 80% tengah wajar, 10% termurah di bawah wajar, "
        "10% termahal di atas wajar."
    )

    st.subheader("Keterbatasan")
    st.markdown(
        "- Data listing Carmudi pada satu waktu (2 September 2026), bukan harga transaksi.\n"
        "- Lokasi terpusat di Jabodetabek dan sekitarnya; Automatic 94,6% data.\n"
        "- Model kurang stabil di segmen ≥ Rp600 juta (kepastian rendah).\n"
        "- Kondisi fisik mobil tidak tertangkap data.\n"
        "- Label adalah indikasi berbasis model dan perlu validasi manual."
    )