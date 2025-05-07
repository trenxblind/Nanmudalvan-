
import cv2
import numpy as np
import streamlit as st
import os
from PIL import Image

downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return blur

def segment_tumor(image):
    preprocessed = preprocess_image(image)
    _, thresh = cv2.threshold(preprocessed, 45, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = image.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    return result, morphed

st.title("🧠 Brain Tumor Segmentation (CV + Streamlit)")
st.write("Upload an MRI scan to detect and segment possible tumor areas.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    st.image(image, caption="🧾 Original Image", use_column_width=True)

    segmented, mask = segment_tumor(image_cv)

    segmented_rgb = cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)

    st.subheader("🧠 Segmented Output")
    st.image(segmented_rgb, use_column_width=True)

    st.subheader("🖤 Tumor Mask")
    st.image(mask, use_column_width=True)

    segmented_path = os.path.join(downloads_folder, "tumor_segmented.png")
    mask_path = os.path.join(downloads_folder, "tumor_mask.png")
    cv2.imwrite(segmented_path, segmented)
    cv2.imwrite(mask_path, mask)

    st.success("✅ Files saved to your Downloads folder!")
    st.write(f"🟢 Saved: `tumor_segmented.png`, `tumor_mask.png`")
