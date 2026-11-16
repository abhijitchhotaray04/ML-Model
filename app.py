import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from ultralytics import YOLO

st.set_page_config(page_title="Classroom Object Detector", layout="centered")
st.title("Classroom Object Detector (YOLO11n)")

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()
CLASSROOM_CLASSES = [0, 13, 24, 26, 39, 56, 62, 63, 67, 73]

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    results = model.predict(
        source=img,
        classes=CLASSROOM_CLASSES,
        conf=0.40,
        device="cpu",
        verbose=False,
    )
    annotated = results[0].plot()
    return av.VideoFrame.from_ndarray(annotated, format="bgr24")

webrtc_streamer(
    key="classroom-yolo",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)