from flask import Flask, render_template, request
import joblib
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Muat model yang sudah dilatih
model = joblib.load('model_iris_knn.joblib')

# Definisikan nama kelas/spesies
nama_spesies = {
    0: 'Iris Setosa',
    1: 'Iris Versicolor',
    2: 'Iris Virginica'
}

@app.route('/')
def home():
    # Tampilkan halaman utama dengan form input
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil data dari form
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])
    
    # Buat array numpy untuk prediksi
    fitur = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Lakukan prediksi
    prediksi_angka = model.predict(fitur)[0]
    
    # Dapatkan nama spesies dari hasil prediksi
    hasil_prediksi = nama_spesies[prediksi_angka]
    
    # Tampilkan halaman hasil
    return render_template('hasil.html', hasil=hasil_prediksi)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
