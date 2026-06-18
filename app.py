from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

import joblib
import pandas as pd
import os

app = Flask(__name__)

# Hanya izinkan frontend milikmu
CORS(
    app,
    resources={
        r"/prediksi": {
            "origins": [
                "https://sipredia.netlify.app"
            ]
        }
    }
)

# Rate limiting
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["100 per hour"]
)

# Memuat model Random Forest
model_rf = joblib.load('model_prediksi_diabetes5.pkl')

# Urutan fitur harus sama dengan saat training
KOLOM_DATASET = [
    'riwayat_darah_tinggi',
    'riwayat_kolesterol',
    'riwayat_jantung',
    'kesehatan_berdasar_keluhan',
    'kesehatan_buruk_sebulan',
    'bmi',
    'jenis_kelamin',
    'kategori_umur'
]

# Endpoint pengecekan server
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "running",
        "service": "sipredia-ml"
    })

# Endpoint prediksi
@app.route('/prediksi', methods=['POST'])
@limiter.limit("20 per minute")
def prediksi():
    try:
        data_pasien = request.get_json()

        if not data_pasien:
            return jsonify({
                "error": "Data JSON tidak ditemukan"
            }), 400

        # Validasi field wajib
        field_hilang = [
            kolom for kolom in KOLOM_DATASET
            if kolom not in data_pasien
        ]

        if field_hilang:
            return jsonify({
                "error": f"Field tidak lengkap: {field_hilang}"
            }), 400

        df_input = pd.DataFrame(
            [data_pasien],
            columns=KOLOM_DATASET
        )

        hasil = model_rf.predict(df_input)

        return jsonify({
            "hasil_prediksi": int(hasil[0])
        }), 200

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# Error jika rate limit tercapai
@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        "error": "Terlalu banyak permintaan. Silakan coba lagi beberapa saat lagi."
    }), 429


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))

    app.run(
        host='0.0.0.0',
        port=port
    )