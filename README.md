# 😄 Image-to-Emoji: Real-Time Facial Expression Recognition

From Face to Emoji using Deep Learning

## 📌 Overview

This project is a fun and interactive application that maps real-time facial expressions to corresponding emojis using computer vision and deep learning. It enables dynamic and emotion-aware emoji overlays through a web interface, making human-computer interaction more expressive.

Created as part of an academic project at **Euromed University of Fes**, under the **Image Processing** course taught by Prof. **Oumaima Moutik**.

---

## 🎯 Objectives

- Detect and classify facial emotions from webcam input or static images.
- Overlay relevant emojis onto the face based on detected emotion.
- Deliver real-time performance via a Python-Flask web interface.

---

## 🛠️ System Architecture

### 🔄 Workflow:
1. **Capture Input**: Webcam captures video using `getUserMedia()`.
2. **Emotion Detection**: 
   - Faces are detected using OpenCV (SSD model).
   - Emotions are classified using `DeepFace`.
3. **Emoji Mapping**: Detected emotions are mapped to corresponding PNG emoji icons.
4. **Overlay & Display**: Emojis are blended onto faces and returned to browser in real time.

---

## 🧪 Technologies Used

| Category       | Tools/Libs         |
|----------------|--------------------|
| Backend        | Python, Flask      |
| Frontend       | HTML5, JavaScript  |
| Vision & ML    | OpenCV, DeepFace, Keras/TensorFlow |
| Visualization  | Alpha blending with OpenCV |
| Assets         | Transparent PNG emoji set |

---

## ⚙️ Setup Instructions

> ⚠️ Make sure you have Python 3.7+ installed.

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/image-to-emoji.git
cd image-to-emoji
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Run the App
```bash
python app.py
```

### 5. Visit in Browser
Open [http://localhost:5000](http://localhost:5000) to try it out.

---

## 🚧 Project Structure

```
📁 image-to-emoji/
├── 📁 static/             # Static files (JS, CSS, Emoji PNGs)
├── 📁 templates/          # HTML templates
├── app.py                # Main Flask backend
├── emotion_utils.py      # Utility for emotion detection
├── requirements.txt      # Dependencies
└── README.md             # Project info
```

---

## 🎉 Demo

- Live webcam feed shows real-time emoji overlay.
- Emotions like **happy**, **sad**, and **angry** are recognized best.

![Demo Preview](insert-demo-image-or-gif-here)

---

## 📈 Results

- Training Accuracy: ~70%
- Validation Accuracy: ~60%
- Best performance on expressive emotions (Happy, Sad, Angry)

---

## 🧑‍💻 Contributions

✔️ Trained CNN model  
✔️ Integrated webcam feed  
✔️ Emoji overlay using alpha blending  
✔️ Flask web interface  
✔️ Testing and documentation

---

## 📚 Future Ideas

- Improve emotion classification accuracy.
- Add support for multiple faces.
- Extend to custom avatars or 3D emojis.
- Mobile app version with camera support.

---

## 📄 License

This project is for academic and non-commercial use.

---

## 📂 Source Code

> 👇 You can find the source code here :

```
[Upimport os
import urllib.request
import cv2
import numpy as np
from flask import Flask, render_template, request, Response
from deepface import DeepFace

app = Flask(__name__)

# ----------------------------------------------------------------------
# Download & load OpenCV SSD face detector
# ----------------------------------------------------------------------
MODEL_DIR     = os.path.join(app.root_path, "models")
PROTO_PATH    = os.path.join(MODEL_DIR, "deploy.prototxt")
WEIGHTS_PATH  = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

def _ensure_face_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(PROTO_PATH):
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
            PROTO_PATH,
        )
    if not os.path.exists(WEIGHTS_PATH):
        urllib.request.urlretrieve(
            "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
            WEIGHTS_PATH,
        )

_ensure_face_model()
face_net = cv2.dnn.readNetFromCaffe(PROTO_PATH, WEIGHTS_PATH)

# ----------------------------------------------------------------------
# DeepFace emotion predictor
# ----------------------------------------------------------------------
# ---------- inside predict_emotion() in app.py ----------

def predict_emotion(face_bgr: np.ndarray) -> str:
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    result = DeepFace.analyze(
        rgb,
        actions=["emotion"],
        enforce_detection=False,
        silent=True
    )
    return result[0]["dominant_emotion"]


# ----------------------------------------------------------------------
# Load emoji PNGs
# ----------------------------------------------------------------------
EMOJI_DIR  = os.path.join(app.root_path, "static", "emojis")
EMOJI_MAP  = {
    "angry":    "angry.png",
    "disgust":  "disgust.png",
    "fear":     "fear.png",
    "happy":    "happy.png",
    "sad":      "sad.png",
    "surprise": "surprise.png",
    "neutral":  "neutral.png",
}
emoji_cache = {k: cv2.imread(os.path.join(EMOJI_DIR, v), cv2.IMREAD_UNCHANGED)
               for k, v in EMOJI_MAP.items() if os.path.exists(os.path.join(EMOJI_DIR, v))}

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def detect_faces(frame: np.ndarray, conf: float = 0.5):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0), swapRB=False, crop=False)
    face_net.setInput(blob)
    detections = face_net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        if detections[0, 0, i, 2] >= conf:
            x1, y1, x2, y2 = (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
            boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes

def overlay_emoji(frame: np.ndarray, x: int, y: int, w: int, h: int, emoji: np.ndarray | None):
    if emoji is None:
        return
    emoji = cv2.resize(emoji, (w, h))
    if emoji.shape[2] == 4:
        alpha = emoji[:, :, 3] / 255.0
        rgb   = emoji[:, :, :3]
        roi   = frame[y:y+h, x:x+w]
        for c in range(3):
            roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * rgb[:, :, c]
        frame[y:y+h, x:x+w] = roi
    else:
        frame[y:y+h, x:x+w] = emoji

def process_frame(frame: np.ndarray) -> np.ndarray:
    H, W = frame.shape[:2]
    for x, y, w, h in detect_faces(frame):
        pad      = int(0.20 * h)
        x1, y1   = max(0, x - pad), max(0, y - pad)
        x2, y2   = min(W, x + w + pad), min(H, y + h + pad)
        face_crop = frame[y1:y2, x1:x2]
        emotion   = predict_emotion(face_crop)
        if emotion not in emoji_cache:
            emotion = "neutral"
        size = w
        new_x = x
        new_y = max(0, y - size)

        overlay_emoji(frame, new_x, new_y, size, size, emoji_cache[emotion])

    return frame
# ----------------------------------------------------------------------
# Flask routes
# ----------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.post("/process_image")
def process_image_route():
    if "image" not in request.files:
        return "No image field", 400
    img = cv2.imdecode(np.frombuffer(request.files["image"].read(), np.uint8),
                       cv2.IMREAD_COLOR)
    if img is None:
        return "Bad image", 400
    out = process_frame(img)
    _, buf = cv2.imencode(".jpg", out)
    return Response(buf.tobytes(), mimetype="image/jpeg")

@app.post("/process_frame")
def process_frame_route():
    if "frame" not in request.files:
        return "No frame field", 400
    img = cv2.imdecode(np.frombuffer(request.files["frame"].read(), np.uint8),
                       cv2.IMREAD_COLOR)
    if img is None:
        return "Bad frame", 400
    out = process_frame(img)
    _, buf = cv2.imencode(".jpg", out)
    return Response(buf.tobytes(), mimetype="image/jpeg")



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
loading app.py…]()

```

Simply create this directory and move your Python files, emoji assets, and frontend templates there. Update paths in your code as needed.

---

## 👨‍🏫 Authors

Achraf Gebli  
Aymen El Achhab  
Aymen Jabbar

**Euromed University of Fes – 2025**
