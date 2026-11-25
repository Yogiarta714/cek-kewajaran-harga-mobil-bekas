"""Logika prediksi dan penilaian kewajaran (tanpa Streamlit, supaya mudah dites)."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

FOLDER = Path(__file__).parent / "artifacts"
TAHUN_DATA = 2026  # tahun acuan umur kendaraan di notebook (umur = 2026 - tahun produksi)


def muat_semua():
    """Baca model, aturan ambang, dan tabel referensi model dari folder artifacts."""
    art = joblib.load(FOLDER / "model_final.joblib")
    with open(FOLDER / "ambang_kewajaran.json") as f:
        aturan = json.load(f)
    ref = pd.read_csv(FOLDER / "referensi_model.csv")
    return art, aturan, ref


def bangun_fitur(art, merek, model_grup, tahun, km, tipe_bodi):
    """Susun 16 fitur persis seperti di notebook untuk SATU mobil."""
    umur = TAHUN_DATA - tahun
    # target encoding: model yang tidak dikenal dapat rata-rata global (sama seperti data uji di notebook)
    model_encoded = art["peta_model_encoded"].get(model_grup, art["rata2_global_harga_log"])

    num = pd.DataFrame([{"km_rata2": km, "umur_kendaraan": umur, "model_encoded": model_encoded}])
    # lokasi tidak dipakai model final (dibuang saat seleksi fitur)
    kategori = pd.DataFrame([{"merek": merek, "tipe_bodi": tipe_bodi, "lokasi": "Lainnya"}])
    oh = pd.DataFrame(
        art["ohe"].transform(kategori[art["kolom_ohe"]]).astype(int),
        columns=art["ohe"].get_feature_names_out(art["kolom_ohe"]),
    )
    X = pd.concat([num, oh], axis=1)
    return X[art["fitur"]]  # urutan kolom harus sama dengan saat model dilatih


def prediksi_harga(art, merek, model_grup, tahun, km, tipe_bodi):
    """Prediksi harga dalam rupiah."""
    X = bangun_fitur(art, merek, model_grup, tahun, km, tipe_bodi)
    return float(np.expm1(art["model"].predict(X)[0]))  # model memprediksi log harga


def nilai_kewajaran(aturan, prediksi, harga=None):
    """Tentukan segmen, kisaran wajar, dan (jika harga diisi) label kewajaran."""
    batas = aturan["batas_segmen_rp"]
    i = int(np.clip(np.searchsorted(batas, prediksi, side="left") - 1, 0, len(aturan["segmen"]) - 1))
    nama_segmen = aturan["segmen"][i]
    a = next(x for x in aturan["ambang"] if x["segmen"] == nama_segmen)

    hasil = {
        "segmen": nama_segmen,
        "batas_bawah": a["batas_bawah"],
        "batas_atas": a["batas_atas"],
        "kepastian": a["kepastian"],
        "harga_wajar_bawah": prediksi * a["batas_bawah"],
        "harga_wajar_atas": prediksi * a["batas_atas"],
        "rasio": None,
        "label": None,
    }
    if harga:
        rasio = harga / prediksi
        hasil["rasio"] = rasio
        if rasio < a["batas_bawah"]:
            hasil["label"] = "di bawah wajar"
        elif rasio > a["batas_atas"]:
            hasil["label"] = "di atas wajar"
        else:
            hasil["label"] = "wajar"
    return hasil