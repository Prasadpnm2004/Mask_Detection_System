# Mask_Detection_System

# 😷 Face Mask Detection with Live Alert System

A real-time face mask detection system using deep learning and OpenCV, with live alerts to ensure public safety during pandemics or in sensitive zones.

---

## 📌 Table of Contents
- [🔍 Overview](#-overview)
- [🧰 Tools & Technologies](#-tools--technologies)
- [⚙️ Features](#️-features)
- [📁 Project Structure](#-project-structure)
- [🚀 Getting Started](#-getting-started)
- [🧠 How It Works](#-how-it-works)
- [📸 Sample Output](#-sample-output)
- [📄 License](#-license)
- [🙋‍♂️ Author](#-author)
- [💬 Feedback](#-feedback)

---

## 🔍 Overview

This project detects whether a person is wearing a face mask in **real-time** using a webcam. If a person is found without a mask, a **live sound alert** is triggered. It utilizes a **pre-trained CNN model** for classification and **OpenCV** for video capture and face detection.

---

## 🧰 Tools & Technologies

- Python 3.7+
- TensorFlow / Keras
- OpenCV
- NumPy
- Imutils
- Playsound
- Haar Cascade Classifier

---

## ⚙️ Features

- Real-time face detection and mask classification
- Audio alert when no mask is detected
- Lightweight and easy to deploy
- Compatible with any system with a webcam
- Clean and modular Python code

---

## 📁 Project Structure

```
face_mask_detection/
│
├── dataset/                  # Dataset (with_mask / without_mask)
├── model/                    # Trained model (.h5)
├── face_detector/            # Haar cascade XML files
├── app.py                    # Main script for real-time detection
├── train_model.py            # Script to train CNN model
├── alert.wav                 # Sound played on no-mask detection
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 🚀 Getting Started

### 🔧 Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.7 or above
- pip (Python package manager)
- A webcam

### 🛠️ Installation Process

Follow the steps below to set up the project on your system:

#### 1. Clone the Repository
```bash
git clone https://github.com/your-username/face-mask-detection.git
cd face-mask-detection
```

#### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Linux/macOS:
source .venv/bin/activate
```

#### 3. Install Required Packages
```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, use:
```bash
pip install opencv-python tensorflow keras numpy imutils playsound
```

#### 4. Ensure Model and Alert Files Are in Place

Make sure you have:
- `mask_detector.model` or `.h5` file inside the `model/` directory
- `haarcascade_frontalface_default.xml` inside `face_detector/`
- `alert.wav` in the project root

#### 5. Run the Application
```bash
python app.py
```

---

## 🧠 How It Works

1. The webcam captures live video.
2. Faces are detected using Haar Cascade Classifier.
3. Detected faces are passed through a pre-trained CNN model to classify as **Mask** or **No Mask**.
4. If a face without a mask is detected:
   - An alert sound is played.
   - A red box with label is shown on the video feed.

---

## 📸 Sample Output

> *(Include screenshots or demo GIFs here if available)*

Example:
- 🟩 Person with mask → Green box labeled “Mask”
- 🟥 Person without mask → Red box labeled “No Mask” + Sound Alert

---

## 📄 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for more details.

---

## 🙋‍♂️ Author

**PM**  
🔗 [GitHub](https://github.com/Prasadpnm2004)  
✉️ Contributions and feedback are welcome!

---

## 💬 Feedback

Feel free to open an issue or submit a pull request for any feature requests, bug fixes, or suggestions.

---
