# Hand Gesture Learning

A personal project where I use my webcam, MediaPipe, and OpenCV to detect hand landmarks in real time and later classify gestures into letters or words.  
This project helps me learn machine learning, computer vision, and real‑time processing.

---

## 🚀 Current Features
- Real‑time webcam capture  
- Hand detection using MediaPipe  
- 21‑point hand landmark tracking  
- Smooth drawing of hand skeleton  
- Press `q` to exit the window  

---

## 🧠 Project Goals
- Build a gesture classifier (KNN, SVM, or small neural network)  
- Create a dataset of hand landmarks  
- Recognize a small set of sign‑language gestures  
- Convert recognized gestures into text  
- (Optional) Add text‑to‑speech output  

---

## 🗂️ Branch Structure
- **main** → stable, working code  
- **experiments** → testing new ideas  
- **ML-classifier** → training and testing gesture recognition models  

---

## 📦 Technologies Used
- Python  
- OpenCV  
- MediaPipe  
- NumPy  
- Scikit‑learn (for classifier later)  

---

## 🎥 Demo (coming soon)
A short demo video will be added once the classifier is working.

---

## 📌 How to Run
```bash
pip install opencv-python mediapipe numpy
python main.py
