import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import base64

def add_local_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/jpeg;base64,{encoded_string}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def customize_sidebar_and_header(sidebar_bg_color, header_bg_color):
    st.markdown(
        f"""
        <style>
        /* Mengubah latar belakang sidebar */
        [data-testid="stSidebar"] {{
            background-color: {sidebar_bg_color};
            color: white;
        }}
        /* Mengubah warna header */
        header[data-testid="stHeader"] {{
            background-color: {header_bg_color};
        }}
        /* Mengubah warna teks di header */
        header[data-testid="stHeader"] .css-1kyxreq {{
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Warna latar belakang dan teks sidebar
customize_sidebar_and_header("#2d2345", "#191e5e")  # Background navy, teks putih

# Gambar lokal
image_path = "./Background.jpg"  # Ganti dengan path gambar Anda
add_local_background(image_path)

# Definisikan fungsi untuk setiap halaman
def halaman_utama():
    # Load the model
    MODEL_PATH = "Firearm-Detector-1.h5"  # Pastikan file model ada di direktori yang sama
    model = load_model(MODEL_PATH)

    # Define the class names (sesuaikan dengan label pada dataset Anda)
    class_names = ["Class 1", "Class 2"]  # Ganti dengan nama kelas sebenarnya

# Function to preprocess the image
    def preprocess_image(image):
    # Pastikan input adalah RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
    
    # Resize gambar sesuai input model
        image = image.resize((224, 224))  # Ganti (224, 224) jika ukuran input model berbeda
        image_array = img_to_array(image) / 255.0  # Normalisasi nilai piksel
        image_array = np.expand_dims(image_array, axis=0)  # Tambahkan dimensi batch
        return image_array

# Streamlit UI
    st.title("Prediksi Gambar")
    st.write("Masukkan gambar untuk diprediksi")

# Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Processing...")

        # Preprocess the image and predict
        image_array = preprocess_image(image)
        prediction = model.predict(image_array)

        # Display the prediction results
        st.write("Prediction Results:")
        for i, score in enumerate(prediction[0]):
            st.write(f"{class_names[i]}: {score*100:.2f}%")

        # Get the predicted class
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        st.write(f"Predicted Class: **{predicted_class}**")
        st.write(f"Confidence: **{confidence:.2f}%**")

def halaman_profil():
    st.title("Halaman Data")
    st.write("Di sini Anda dapat mengelola data.")

def halaman_github():
    st.title("Link Github")
    st.write("Ini adalah halaman tentang aplikasi Anda.")

# Navigasi halaman melalui sidebar
def main():
    st.sidebar.title("Navigasi")
    pilihan = st.sidebar.radio("Pilih Halaman", ("Prediksi Gambar", "Profil", "Github"))

    if pilihan == "Prediksi Gambar":
        halaman_utama()
    elif pilihan == "Profil":
        halaman_profil()
    elif pilihan == "Github":
        halaman_github()

if __name__ == "__main__":
    main()
