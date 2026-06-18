from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# Memuat model Random Forest
model_rf = joblib.load('model_prediksi_diabetes5.pkl')

# Mencegah model bingung jika urutan data dari frontend berubah
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

# Endpoint untuk mengecek apakah server berjalan
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "running",
        "service": "sipredia-ml"
    })

# Endpoint prediksi
@app.route('/prediksi', methods=['POST'])
def prediksi():
    try:
        data_pasien = request.get_json()

        df_input = pd.DataFrame(
            [data_pasien],
            columns=KOLOM_DATASET
        )

        hasil = model_rf.predict(df_input)

        return jsonify({
            'hasil_prediksi': int(hasil[0])
        }), 200

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(
        host='0.0.0.0',
        port=port
    )