
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import os

downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")

# Load the pre-trained Haar Cascade classifier for number plates
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

def detect_number_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 50))

    plate_img = image.copy()
    plate = None
    for (x, y, w, h) in plates:
        plate = image[y:y+h, x:x+w]
        cv2.rectangle(plate_img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    return plate_img, plate

st.title("🚗 Number Plate Detection (CV + Streamlit)")
st.write("Upload a vehicle image to detect the number plate.")

uploaded_file = st.file_uploader("Upload Vehicle Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    st.image(image, caption="🖼️ Original Image", use_column_width=True)

    detected_img, plate = detect_number_plate(image_cv)

    st.subheader("📦 Detected Number Plate Area")
    detected_img_rgb = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)
    st.image(detected_img_rgb, use_column_width=True)

    if plate is not None:
        plate_path = os.path.join(downloads_folder, "detected_plate.png")
        cv2.imwrite(plate_path, plate)
        st.success("✅ Number plate extracted and saved in Downloads!")
        st.image(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB), caption="📍 Cropped Plate")
    else:
        st.warning("⚠️ No number plate found!")
