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

> 👇 You can place your source code files in the following folder:

```
📁 /src
```

Simply create this directory and move your Python files, emoji assets, and frontend templates there. Update paths in your code as needed.

---

## 👨‍🏫 Authors

Achraf Gebli  
Aymen El Achhab  
Aymen Jabbar

**Euromed University of Fes – 2025**
