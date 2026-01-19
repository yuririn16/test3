import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
from PIL import Image

# MediaPipeの顔検出機能をセットアップ
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

st.title("Android顔認識カメラアプリ")
st.write("写真を撮ると、AIが顔を自動で認識します。")

# カメラ入力
img_file = st.camera_input("Take a photo")

if img_file is not None:
    # 画像を読み込む
    image = Image.open(img_file)
    img_array = np.array(image)

    # 顔検出の実行
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(img_array)

        # 検出された顔に枠を描く
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(img_array, detection)
            st.success(f"{len(results.detections)} 人の顔を検出しました！")
        else:
            st.warning("顔が見つかりませんでした。")

    # 結果を表示
    st.image(img_array, caption="判定結果", use_container_width=True)
