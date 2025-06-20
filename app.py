import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from utils import alert_user

model = load_model("mask_detector.h5")
mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

st.title("😷 Face Mask Detection App (Live)")
run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])
cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Camera not detected.")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y = int(bbox.xmin * w), int(bbox.ymin * h)
            w_box, h_box = int(bbox.width * w), int(bbox.height * h)
            face = frame[y:y+h_box, x:x+w_box]
            try:
                face = cv2.resize(face, (224, 224))
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)
                result = model.predict(face)[0]
                label = "Mask" if np.argmax(result) == 0 else "No Mask"
                color = (0,255,0) if label=="Mask" else (0,0,255)
                if label == "No Mask":
                    alert_user("No Mask Detected!")
                cv2.rectangle(frame, (x, y), (x+w_box, y+h_box), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            except:
                pass

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()