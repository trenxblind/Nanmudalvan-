
import cv2
import streamlit as st
import tempfile
import os
import numpy as np
from datetime import datetime

# Helper function to apply domain-specific filtering
def domain_filter(keypoints, domain):
    if domain == "Security":
        return [kp for kp in keypoints if kp.size > 5]
    elif domain == "Wildlife":
        return [kp for kp in keypoints if 3 < kp.size < 20]
    elif domain == "Sports":
        return [kp for kp in keypoints if kp.size > 5]
    return keypoints

# Motion tracking logic
def process_video(input_path, domain):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.avi')
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (frame_width, frame_height))

    orb = cv2.ORB_create(nfeatures=1000)
    fast = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)

    prev_kp = None
    prev_des = None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp = fast.detect(gray, None)
        if not kp:
            continue
        kp = domain_filter(kp, domain)
        if not kp:
            continue
        kp, des = orb.compute(gray, kp)
        if des is None or len(des) < 2:
            continue

        if prev_kp is not None and prev_des is not None:
            matches = bf.match(prev_des, des)
            matches = sorted(matches, key=lambda x: x.distance)
            for m in matches[:50]:
                pt1 = tuple(np.round(prev_kp[m.queryIdx].pt).astype(int))
                pt2 = tuple(np.round(kp[m.trainIdx].pt).astype(int))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
                cv2.circle(frame, pt2, 4, (0, 0, 255), -1)

        prev_kp = kp
        prev_des = des

        out.write(frame)

    cap.release()
    out.release()
    return temp_output.name

# Streamlit UI
st.title("Drone Video Motion Tracking (Improved FAST + ORB)")

st.sidebar.header("Options")
domain = st.sidebar.selectbox("Select Domain", ["Security", "Wildlife", "Sports"])

uploaded_video = st.file_uploader("Upload a Drone Video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_input:
        tmp_input.write(uploaded_video.read())
        tmp_input_path = tmp_input.name

    st.text("Processing video... Please wait.")
    output_video_path = process_video(tmp_input_path, domain)
    st.success("Video processing complete!")

    st.video(output_video_path)

    with open(output_video_path, "rb") as f:
        st.download_button(
            label="Download Tracked Video",
            data=f,
            file_name=f"motion_tracked_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi",
            mime="video/avi"
        )

    os.remove(tmp_input_path)
    os.remove(output_video_path)
