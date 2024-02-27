import tensorflow as tf
import numpy as np
import requests
from PIL import Image
from io import BytesIO

# Langkah 1: Impor TensorFlow

# Langkah 2: Muat model dari file .h5
model = tf.keras.models.load_model('model.h5')

# Langkah 3: Tentukan nama kelas yang sesuai dengan model Anda
class_names = ['class2', 'class1']  # Ganti dengan nama kelas yang sesuai dengan model Anda

# Langkah 4: Fungsi untuk memuat gambar dari URL
def load_image_from_url(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return image
    except Exception as e:
        print("Error loading image from URL:", e)
        return None

# Langkah 5: Fungsi untuk memproses gambar dan membuat prediksi
def predict_class_from_image_url(image_url):
    # Memuat gambar dari URL
    image = load_image_from_url(image_url)
    if image is not None:
        # Preprocessing gambar (sesuaikan dengan preprocessing yang dilakukan saat melatih model)
        # Contoh: Resize gambar menjadi ukuran yang diharapkan oleh model dan normalisasi nilai pixel
        image = image.resize((200, 200))  # Ganti ukuran gambar sesuai kebutuhan model
        print(image)
        image_array = np.array(image) / 255.0  # Normalisasi nilai pixel
        #print(image_array)

        # Lakukan prediksi pada gambar
        predictions = model.predict(np.expand_dims(image_array, axis=0))
        
        # Ambil indeks kelas dengan probabilitas tertinggi
        predicted_class_index = np.argmax(predictions)
        print(predicted_class_index)
        # Dapatkan nama kelas dari indeks
        predicted_class_name = class_names[predicted_class_index]

        return predicted_class_name
    else:
        return None

# Langkah 6: Contoh penggunaan
# URL gambar yang ingin Anda prediksi kelasnya
image_url = "http://ppdb-premium.mkom/1.png"  # Ganti dengan URL gambar yang sesuai
# image_url = "https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEjij9ChamshdLlX4RsAAIbTwL556oJKVCnvWpa94V6dSyAAwlm_viRuh_fuJfuMJY9qegQM8-mp1EgFJfQCySzP4LaVN2S5qX97Su2myGCIV4Y5xglzxXFQ9vJ6CDzAfCce2YUrf_rrbQfjGZ8G8aI38OMgHKq_cvYkzFpRk5pb_f9n--MMuJMm5Dgb/w633-h259/Screenshot%20from%202023-05-12%2010-19-36.png"  # Ganti dengan URL gambar yang sesuai

# Lakukan prediksi pada gambar dari URL
predicted_class = predict_class_from_image_url(image_url)

if predicted_class is not None:
    # Tampilkan hasil prediksi
    print("Predicted class:", predicted_class)
else:
    print("Failed to make prediction.")