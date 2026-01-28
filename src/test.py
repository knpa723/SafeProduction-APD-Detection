from streamlit_webrtc import webrtc_streamer
import streamlit as st
import av
from handlers.pipeline_handler import run_pipeline
from models.downloader import download_files
import cv2
import traceback
import numpy as np
from dotenv import load_dotenv
import os
import time

load_dotenv()

MODELS_GOOGLE_DRIVE = os.getenv("MODELS_GOOGLE_DRIVE")

@st.cache_resource
def prepare_models():
    # Run models downloader
    download_files(MODELS_GOOGLE_DRIVE, 'src/models')
    
prepare_models()

# def video_frame_callback(frame):
#     try:
        
#         img = frame.to_ndarray(format="bgr24")
        
#         # For detections
#         # result = run_pipeline(img)
#         # print(result)
        
#         # for human in result.get("detections", []):
#         #     x1, y1, x2, y2 = human["bbox"]
#         #     label = human["label"]      
#         #     conf = human["confidence"]   
            
#         #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         #     cv2.putText(img, f"{label} ({conf:.2f})",
#         #                 (x1, y1 - 10), 
#         #                 cv2.FONT_HERSHEY_SIMPLEX, 
#         #                 0.6, 
#         #                 (0, 255, 0), 
#         #                 2)
        
#         # return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        
#         # For Tracking
#         annotated = run_pipeline(img)
#         if annotated is None:
#             raise ValueError("run_pipeline returned None")

#         if not isinstance(annotated, np.ndarray):
#             raise TypeError(f"Expected ndarray, got {type(annotated)}")
        
#         return av.VideoFrame.from_ndarray(annotated, format="bgr24")
    
#     except Exception as e:
#         print("VIDEO CALLBACK ERROR")
#         print(traceback.format_exc())

#         # Return raw frame so stream does NOT freeze
#         return frame

video_path = "src/test_video/video_1.mp4"

def video_frame_callback(img): 
    try:
        annotated = run_pipeline(img)
        
        if annotated is None:
            raise ValueError("run_pipeline returned None")

        if not isinstance(annotated, np.ndarray):
            raise TypeError(f"Expected ndarray, got {type(annotated)}")
        
        return annotated
    
    except Exception as e:
        print("VIDEO CALLBACK ERROR")
        print(traceback.format_exc())

        return img

if os.path.exists(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_placeholder = st.image([])
    stop_button = st.button("Stop Video")

    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        processed_frame = video_frame_callback(frame)

        frame_placeholder.image(processed_frame, channels="BGR")
        
        time.sleep(0.01)

    cap.release()
else:
    st.error(f"Video tidak ditemukan: {video_path}")
    

    
webrtc_streamer(key="example", video_frame_callback=video_frame_callback)