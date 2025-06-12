from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

app = Flask(__name__)

MODEL_PATH = "my_model.h5"
SCALER_PATH = "scaler.pkl"
MAJORS_DATA_PATH = "data/majors (1).csv"
UNIVERSITIES_DATA_PATH = "data/universities (1).csv"

model = None
scaler = None
major_univ = None

def load_resources():
    global model, scaler, major_univ
    
    # Memuat Model Keras
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model '{MODEL_PATH}' tidak ditemukan. Pastikan ada di root folder.")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Model berhasil dimuat dari {MODEL_PATH}")
    except Exception as e:
        raise Exception(f"Gagal memuat model dari {MODEL_PATH}: {e}")

    # Memuat Scaler
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler '{SCALER_PATH}' tidak ditemukan. Pastikan ada di root folder.")
    try:
        scaler = joblib.load(SCALER_PATH)
        print(f"Scaler berhasil dimuat dari {SCALER_PATH}")
    except Exception as e:
        raise Exception(f"Gagal memuat scaler dari {SCALER_PATH}: {e}")

    # Memuat Data Major dan Univ
    if not os.path.exists(MAJORS_DATA_PATH) or not os.path.exists(UNIVERSITIES_DATA_PATH):
        raise FileNotFoundError(f"File data '{MAJORS_DATA_PATH}' atau '{UNIVERSITIES_DATA_PATH}' tidak ditemukan. Pastikan ada di folder 'data/'.")
    try:
        major_df = pd.read_csv(MAJORS_DATA_PATH, index_col=0)
        univ_df = pd.read_csv(UNIVERSITIES_DATA_PATH, index_col=0)

        # Normalisasi dan pemetaan tipe (sesuai streamlit)
        major_df['type'] = major_df['type'].astype(str).str.lower().str.strip()
        major_df['type'] = major_df['type'].replace({'saintek': 'science', 'soshum': 'humanities'})
        
        major_df['utbk_capacity'] = (0.4 * major_df['capacity']).apply(int)
        major_df['passed_count'] = 0 
        major_univ = pd.merge(major_df, univ_df, on='id_university', how='left')
        major_univ.set_index('id_major', inplace=True)
        print(f"Data jurusan dan universitas berhasil dimuat.")
    except Exception as e:
        raise Exception(f"Gagal memuat atau memproses data jurusan/universitas: {e}")

# Panggil fungsi load_resources
try:
    load_resources()
except Exception as e:
    print(f"ERROR: Gagal memuat sumber daya awal: {e}")

# --- Fungsi Utility untuk Cek Kesesuaian Jurusan ---
def major_univ_check(cols):
    major_id = str(cols[0])
    univ_id = str(cols[1])
    major_type = cols[2]
    test_type = cols[3]

    # cek id major apakah sesuai atau tidak dengan tipe tes
    if (major_type == test_type) and (major_id.startswith(univ_id)):
        return True
    else:
        return False

# --- Endpoint Utama  ---
@app.route('/')
def home():
    return "API Rekomendasi Jurusan UTBK sedang berjalan!"

# --- Endpoint Prediksi/Rekomendasi ---
@app.route('/recommend', methods=['POST'])
def recommend():

    if model is None or scaler is None or major_univ is None:
        return jsonify({"error": "Model, scaler, atau data jurusan belum dimuat sepenuhnya. Periksa log server."}), 500

    try:
        # Menerima data JSON dari permintaan POST
        data = request.get_json(force=True)

        # Validasi input 
        required_keys = ['scores', 'test_type']
        for key in required_keys:
            if key not in data:
                return jsonify({"error": f"Kunci '{key}' hilang dari input JSON."}), 400
        
        input_scores = data['scores']
        student_test_type_str = data['test_type'].lower().strip() # 'science' atau 'humanities'

        # Validasi skor 
        tps_keys = ['score_kpu', 'score_kua', 'score_ppu', 'score_kmb']
        for key in tps_keys:
            if key not in input_scores:
                return jsonify({"error": f"Kunci '{key}' hilang dari skor TPS."}), 400
        
        # Hitung skor rata-rata berdasarkan input
        general_score_rec = np.mean([input_scores['score_kpu'], input_scores['score_kua'], input_scores['score_ppu'], input_scores['score_kmb']])
        
        specialize_score_rec = 0.0 
        tka_scores = []
        if student_test_type_str == 'science':
            tka_keys = ['score_mat_tka', 'score_fis', 'score_kim', 'score_bio']
            for key in tka_keys:
                if key not in input_scores: return jsonify({"error": f"Kunci '{key}' hilang dari skor TKA Sains."}), 400
                tka_scores.append(input_scores[key])
            specialize_score_rec = np.mean(tka_scores)
            
        elif student_test_type_str == 'humanities':
            tka_keys = ['score_mat_tka', 'score_geo', 'score_sej', 'score_sos', 'score_eko']
            for key in tka_keys:
                if key not in input_scores: return jsonify({"error": f"Kunci '{key}' hilang dari skor TKA Soshum."}), 400
                tka_scores.append(input_scores[key])
            specialize_score_rec = np.mean(tka_scores)
        else:
            return jsonify({"error": "Tipe tes tidak valid. Harus 'science' atau 'humanities'."}), 400

        all_scores_flat = list(input_scores.values())
        score_mean_rec = np.mean(all_scores_flat)

        test_type_encoded = 0 if student_test_type_str == 'science' else 1

        recommendations = []
        # Filter jurusan berdasarkan tipe tes siswa
        filtered_majors = major_univ[major_univ['type'] == student_test_type_str].copy()

        if filtered_majors.empty:
            return jsonify({"message": f"Tidak ada jurusan '{student_test_type_str}' yang ditemukan di dataset untuk direkomendasikan."}), 200
        
        # Iterasi dan prediksi untuk setiap jurusan yang cocok
        for index, row in filtered_majors.iterrows():
            id_major_candidate = index
            id_university_candidate = row['id_university']

            simulated_input = np.array([[
                id_major_candidate,
                id_university_candidate,
                id_major_candidate, 
                id_university_candidate, 
                general_score_rec,
                specialize_score_rec,
                score_mean_rec,
                test_type_encoded
            ]], dtype=np.float32)

            scaled_input = scaler.transform(simulated_input)
            prob_pass = model.predict(scaled_input)[0][0]

            recommendations.append({
                "id_major": id_major_candidate,
                "major_name": row['major_name'],
                "university_name": row['university_name'],
                "prob_pass": float(prob_pass), 
                "capacity": int(row['capacity']),
                "utbk_capacity": int(row['utbk_capacity'])
            })

        recom_df = pd.DataFrame(recommendations)
        threshold_rekomendasi = 0.5 

        final_recommendations = recom_df[recom_df['prob_pass'] >= threshold_rekomendasi].sort_values(by='prob_pass', ascending=False)
        
        if not final_recommendations.empty:
            # Mengubah DataFrame ke format list of dicts untuk JSON
            result = final_recommendations[[
                'major_name', 
                'university_name', 
                'prob_pass', 
                'utbk_capacity'
            ]].head(15).to_dict(orient='records')
            
            return jsonify({
                "status": "success",
                "message": "Rekomendasi jurusan berhasil ditemukan.",
                "recommendations": result
            }), 200
        else:
            return jsonify({
                "status": "no_recommendations",
                "message": f"Maaf, tidak ada jurusan yang direkomendasikan dengan probabilitas kelulusan di atas {threshold_rekomendasi:.0%} untuk jenis tes yang Anda pilih.",
                "recommendations": []
            }), 200

    except Exception as e:
        
        return jsonify({"error": f"Terjadi kesalahan internal: {str(e)}"}), 500

if __name__ == '__main__':

    app.run(debug=True, port=5000)