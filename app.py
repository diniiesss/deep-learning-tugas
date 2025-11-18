from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename 
import pandas as pd
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense, Bidirectional, Input
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.preprocessing import MinMaxScaler
from PIL import Image 
import io 
from ultralytics import YOLO 

app = Flask(__name__)

LOOK_BACK = 60 
FILE_SAHAM = "Data Historis BBCA_Test2.csv"
SEQ_LENGTH_T2 = 5 
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 


def hitung_logika(a_str: str, b_str: str, operasi: str) -> str:
    """Menghitung hasil operasi logika."""
    a = (a_str == '1'); b = (b_str == '1'); a_int = int(a); b_int = int(b)
    if operasi == "AND": hasil = a and b
    elif operasi == "OR": hasil = a or b
    elif operasi == "XOR": hasil = bool(a_int ^ b_int)
    elif operasi == "NAND": hasil = not (a and b)
    elif operasi == "NOR": hasil = not (a or b)
    elif operasi == "XNOR": hasil = not bool(a_int ^ b_int)
    else: return "Operasi Tidak Valid"
    return str(int(hasil))

def buat_tabel_kebenaran(operasi: str) -> list:
    """Membuat data tabel kebenaran lengkap."""
    data = []; input_values = [('0', '0'), ('0', '1'), ('1', '0'), ('1', '1')]
    for p_str, q_str in input_values:
        hasil_str = hitung_logika(p_str, q_str, operasi)
        data.append({'P': p_str, 'Q': q_str, 'Hasil': hasil_str})
    return data

def prepare_data(text, seq_length=SEQ_LENGTH_T2):
    """Memproses teks input untuk RNN."""
    text = text.strip().lower(); chars = sorted(list(set(text))); vocab_size = len(chars)
    char_to_index = {char: i for i, char in enumerate(chars)}; index_to_char = {i: char for i, char in enumerate(chars)}
    sequences, labels = [], []
    for i in range(len(text) - seq_length):
        seq = text[i:i + seq_length]; label = text[i + seq_length]
        sequences.append([char_to_index[char] for char in seq]); labels.append(char_to_index[label])
    X = np.array(sequences); y = np.array(labels)
    X_one_hot = tf.one_hot(X, vocab_size).numpy(); y_one_hot = tf.one_hot(y, vocab_size).numpy()
    return X_one_hot, y_one_hot, vocab_size, char_to_index, index_to_char

def build_model(model_type, seq_length, vocab_size):
    """Membangun model Sequential Keras RNN."""
    model = Sequential(); model.add(Input(shape=(seq_length, vocab_size)))
    if model_type == "Vanilla RNN": model.add(SimpleRNN(50, activation='relu'))
    elif model_type == "Bidirectional RNN": model.add(Bidirectional(SimpleRNN(50, activation='relu')))
    elif model_type == "LSTM": model.add(LSTM(50))
    elif model_type == "GRU": model.add(GRU(50))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def generate_text_and_plot(input_text, model_choice, seed_text, gen_length, epochs):
    """Melatih RNN, menghasilkan teks, dan membuat plot."""
    if len(input_text) < SEQ_LENGTH_T2 + 5: return None, None, "Teks pelatihan terlalu pendek."
    if len(seed_text) < SEQ_LENGTH_T2: return None, None, f"'Seed Text' harus minimal {SEQ_LENGTH_T2} karakter."
    
    X_one_hot, y_one_hot, vocab_size, char_to_index, index_to_char = prepare_data(input_text, seq_length=SEQ_LENGTH_T2)
    model = build_model(model_choice, SEQ_LENGTH_T2, vocab_size)
    history = model.fit(X_one_hot, y_one_hot, epochs=int(epochs), batch_size=32, verbose=0)
    
    generated = seed_text; current_seed = seed_text[-SEQ_LENGTH_T2:].lower()
    for _ in range(int(gen_length)):
        seq = current_seed[-SEQ_LENGTH_T2:]; x_indices = [char_to_index.get(c, 0) for c in seq]
        x = np.array([x_indices]); x_one_hot_tensor = tf.one_hot(x, vocab_size)
        prediction = model.predict(x_one_hot_tensor.numpy(), verbose=0); next_char_index = np.argmax(prediction)
        next_char = index_to_char.get(next_char_index, ' ')
        generated += next_char; current_seed += next_char
        
    fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(history.history['accuracy'], label='Akurasi Training', color='purple')
    ax.set_title(f'Akurasi Training Model: {model_choice}', fontsize=12); ax.set_xlabel('Epoch'); ax.set_ylabel('Akurasi'); ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)
    buf = BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight'); plt.close(fig); buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plot_html = f"<img src='data:image/png;base64,{image_base64}' style='width:100%; max-width: 500px; height:auto; display: block; margin: 10px auto;'/>"
    return generated, plot_html, None 

def create_sequences_t3(data, look_back=LOOK_BACK):
    """Membuat sekuens X dan y untuk training RNN."""
    X, y = [], []
    for i in range(len(data) - look_back): X.append(data[i:(i + look_back), 0]); y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

def build_lstm_model_t3(model_type, look_back):
    """Membangun model Time Series."""
    model = Sequential(); model.add(Input(shape=(look_back, 1)))
    if model_type == "LSTM": model.add(LSTM(units=50, return_sequences=False))
    elif model_type == "GRU": model.add(GRU(units=50, return_sequences=False))
    elif model_type == "Bidirectional LSTM": model.add(Bidirectional(LSTM(units=50, return_sequences=False)))
    elif model_type == "Vanilla RNN": model.add(SimpleRNN(units=50, activation='relu', return_sequences=False))
    model.add(Dense(units=1)); model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_stock_price(pred_days, model_choice_t3):
    """Melatih dan memprediksi harga saham."""
    if not os.path.exists(FILE_SAHAM): return None, f"File dataset '{FILE_SAHAM}' tidak ditemukan di server."
    df = pd.read_csv(FILE_SAHAM); df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y', errors='coerce'); df.dropna(subset=['Date'], inplace=True)
    data = df.sort_values('Date')['Close'].values.reshape(-1, 1); scaler = MinMaxScaler(feature_range=(0, 1)); scaled_data = scaler.fit_transform(data)
    X, y = create_sequences_t3(scaled_data, look_back=LOOK_BACK); X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    model = build_lstm_model_t3(model_choice_t3, LOOK_BACK); model.fit(X, y, epochs=10, batch_size=32, verbose=0)
    
    last_60_days = scaled_data[-LOOK_BACK:]; temp_input = list(last_60_days.flatten()); predicted_prices = []
    for _ in range(pred_days):
        x_input = np.array(temp_input[-LOOK_BACK:]); x_input = x_input.reshape((1, LOOK_BACK, 1))
        pred_scaled = model.predict(x_input, verbose=0)[0]; temp_input.append(pred_scaled[0]); predicted_prices.append(pred_scaled[0])

    predicted_prices_np = np.array(predicted_prices).reshape(-1, 1); predicted_prices_actual = scaler.inverse_transform(predicted_prices_np)
    plot_data = data[-200:].flatten(); hist_indices = np.arange(len(plot_data)); pred_indices = np.arange(len(plot_data), len(plot_data) + pred_days)
    
    fig, ax = plt.subplots(figsize=(10, 5)); ax.plot(hist_indices, plot_data, label='Harga Historis (200 Hari Terakhir)', color='purple')
    ax.plot(pred_indices, predicted_prices_actual, label=f'Prediksi {pred_days} Hari Kedepan', color='pink', linestyle='dashed')
    ax.set_title(f'Prediksi Harga Saham BBCA Menggunakan {model_choice_t3}', fontsize=14); ax.set_xlabel('Hari (Indeks Data)', fontsize=12); ax.set_ylabel('Harga Penutupan (IDR)', fontsize=12); ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)
    buf = BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight'); plt.close(fig); buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plot_html = f"<img src='data:image/png;base64,{image_base64}' style='width:100%; height:auto; display: block; margin: 10px auto;'/>"
    return predicted_prices_actual[0][0], plot_html


YOLO_MODEL = None

def detect_objects_yolo(image_path):
    """Memuat model YOLOv8n dan mendeteksi objek, lalu menyimpan hasilnya sebagai base64."""
    global YOLO_MODEL
    
    if YOLO_MODEL is None:
        try:
            YOLO_MODEL = YOLO('yolov8n.pt') 
        except Exception as e:
            return None, f"Gagal memuat model YOLOv8n. Pastikan 'ultralytics' terinstal: {e}"

    results = YOLO_MODEL(image_path, save=False, conf=0.50) 
    
    result_image_array = results[0].plot() 
    
    result_image_rgb = Image.fromarray(result_image_array[..., ::-1]) 
    
    buf = BytesIO()
    result_image_rgb.save(buf, format='PNG')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    plot_html = f"<img src='data:image/png;base64,{image_base64}' style='width:100%; height:auto; display: block; margin: 10px auto;'/>"

    names = YOLO_MODEL.names
    detected_objects = []
    
    for box in results[0].boxes.data:
        class_id = int(box[5])
        confidence = float(box[4])
        detected_objects.append(f"{names[class_id]} ({confidence:.2f})")

    return detected_objects, plot_html

@app.route('/')
def index():
    """Menampilkan halaman utama dengan semua tugas."""
    return render_template('index.html', hasil_t1=None, hasil_t2=None, hasil_t3=None, hasil_t4=None)

@app.route('/tugas1', methods=['POST'])
def tugas1_predict():
    if request.method == 'POST':
        try:
            input_a = request.form.get('input_a', '1')
            input_b = request.form.get('input_b', '0')
            operator = request.form.get('operator', 'AND')
            hasil_kalkulator = hitung_logika(input_a, input_b, operator)
            tabel_kebenaran = buat_tabel_kebenaran(operator)
            hasil = {'input_a': input_a, 'input_b': input_b, 'operator': operator, 'output': hasil_kalkulator, 'tabel': tabel_kebenaran}
            return render_template('index.html', hasil_t1=hasil)
        except Exception as e:
            return render_template('index.html', hasil_t1={'error': f"Terjadi kesalahan pada Tugas 1: {e}"})

@app.route('/tugas2', methods=['POST'])
def tugas2_predict():
    if request.method == 'POST':
        try:
            DEFAULT_TEXT = "Ini adalah contoh teks yang diaktifkan oleh Recurrent Neural Networks. Deep learning mempelajari data sekuensial dan model RNN digunakan untuk prediksi karakter berikutnya, seperti LSTM dan GRU. Coba bandingkan hasil teks dari berbagai model!"
            input_text = request.form.get('input_text', DEFAULT_TEXT); model_choice = request.form.get('model_choice', "LSTM") 
            seed_text = request.form.get('seed_text', "Model"); gen_length = int(request.form.get('gen_length', 150))
            epochs = int(request.form.get('epochs', 30))
            predicted_text, plot_html, error_msg = generate_text_and_plot(input_text=input_text, model_choice=model_choice, seed_text=seed_text, gen_length=gen_length, epochs=epochs)
            hasil = {'input_text': input_text, 'model_choice': model_choice, 'seed_text': seed_text, 'predicted_text': predicted_text, 'plot_html': plot_html, 'error': error_msg}
            return render_template('index.html', hasil_t2=hasil)
        except Exception as e:
            return render_template('index.html', hasil_t2={'error': f"❌ Terjadi kesalahan saat menjalankan model: {e}"})

@app.route('/tugas3', methods=['POST'])
def tugas3_predict():
    if request.method == 'POST':
        try:
            stock_symbol = request.form.get('stock_symbol', 'BBCA'); pred_days = int(request.form.get('pred_days', 7))
            model_choice_t3 = request.form.get('model_choice_t3', 'LSTM')
            if pred_days < 1 or pred_days > 30: raise ValueError("Jumlah hari prediksi harus antara 1 dan 30.")
            first_day_prediction, plot_html = predict_stock_price(pred_days, model_choice_t3)
            if first_day_prediction is None: return render_template('index.html', hasil_t3={'error': plot_html})
            hasil = {'symbol': stock_symbol, 'days': pred_days, 'model_choice_t3': model_choice_t3, 'prediction': "{:,.2f}".format(first_day_prediction), 'plot_html': plot_html}
            return render_template('index.html', hasil_t3=hasil)
        except Exception as e:
            error_msg = f"❌ Terjadi kesalahan pada Tugas 3: {e}"
            return render_template('index.html', hasil_t3={'error': error_msg})


@app.route('/tugas4', methods=['POST'])
def tugas4_predict():
    """Memproses upload gambar dan menjalankan deteksi objek YOLO."""
    if request.method == 'POST':
        if 'image_file' not in request.files:
            return render_template('index.html', hasil_t4={'error': 'Tidak ada file gambar yang diupload.'})
        
        file = request.files['image_file']
        
        if file.filename == '':
            return render_template('index.html', hasil_t4={'error': 'Nama file tidak valid.'})
            
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            try:
                detected_objects, plot_html = detect_objects_yolo(file_path)
                
                with open(file_path, 'rb') as f:
                    uploaded_img_base64 = base64.b64encode(f.read()).decode('utf-8')

                os.remove(file_path)

                hasil = {
                    'detected_objects': detected_objects,
                    'plot_html': plot_html,
                    'uploaded_img': f"data:image/png;base64,{uploaded_img_base64}"
                }
                return render_template('index.html', hasil_t4=hasil)

            except Exception as e:
                if os.path.exists(file_path):
                    os.remove(file_path)
                return render_template('index.html', hasil_t4={'error': f"❌ Error saat deteksi YOLO: {e}"})

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
        
    port = int(os.environ.get("PORT", 5000))
    
    app.run(host='0.0.0.0', port=port, debug=False)