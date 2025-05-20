import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import tempfile
import requests

# Load YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.25
model.iou = 0.35
allowed_classes = ['car', 'motorbike', 'bus', 'truck']

# Fungsi untuk deteksi dari frame
def detect_parking_violation(frame, zone_side='left', zone_ratio=0.2):
    height, width, _ = frame.shape

    if zone_side == 'left':
        x1, x2 = 0, int(width * zone_ratio)
    elif zone_side == 'right':
        x1, x2 = int(width * (1 - zone_ratio)), width
    elif zone_side == 'center':
        center = width // 2
        half_zone = int(width * zone_ratio) // 2
        x1, x2 = center - half_zone, center + half_zone
    else:
        x1, x2 = 0, int(width * zone_ratio)

    # Zona larangan parkir
    cv2.rectangle(frame, (x1, 0), (x2, height), (255, 0, 0), 2)
    cv2.putText(frame, f"Zona Larangan Parkir ({zone_side})", (x1 + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    results = model(frame)
    detections = results.pandas().xyxy[0]

    for _, row in detections.iterrows():
        label = row['name']
        conf = row['confidence']
        if label in allowed_classes and conf > 0.4:
            dx1, dy1, dx2, dy2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            if dx1 >= x1 and dx2 <= x2:
                cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({conf:.2f})", (dx1, dy1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame

# Sidebar
st.sidebar.title("🅿️ Sistem Deteksi Parkir Liar")
page = st.sidebar.radio("Navigasi", ["🔴 Live Stream", "🖼️ Upload Gambar"])

# Upload Gambar
if page == "🖼️ Upload Gambar":
    st.title("📤 Deteksi Parkir Liar dari Gambar")

    uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "png", "jpeg"])
    zone_side = st.selectbox("Pilih sisi zona:", ["left", "right", "center"])
    zone_ratio = st.slider("Rasio zona (%)", 0.1, 0.5, 0.2)

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        result = detect_parking_violation(image, zone_side, zone_ratio)
        st.image(result, channels="BGR", caption="Hasil Deteksi")

# Live Stream (via frame refresh, bukan video stream karena Streamlit)
elif page == "🔴 Live Stream":
    st.title("🔴 Deteksi Parkir Liar Realtime")

    stream_url = "http://stream.cctv.malangkota.go.id/WebRTCApp/streams/383467698805698590809269.m3u8"
    stframe = st.empty()

    cap = cv2.VideoCapture(stream_url)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Gagal memuat stream. Memuat ulang...")
            cap = cv2.VideoCapture(stream_url)
            continue

        height, width, _ = frame.shape
        left_zone_x = int(width * 0.18)

        cv2.rectangle(frame, (0, 0), (left_zone_x, height), (255, 0, 0), 2)
        cv2.putText(frame, "Zona Larangan Parkir", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        results = model(frame)
        detections = results.pandas().xyxy[0]

        for _, row in detections.iterrows():
            label = row['name']
            conf = row['confidence']
            if label in allowed_classes and conf > 0.4:
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                if x1 < left_zone_x:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        stframe.image(frame, channels="BGR")

    cap.release()
