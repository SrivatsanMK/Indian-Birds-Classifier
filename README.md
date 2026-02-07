# 🦜 Bird Species Classification Using Deep Learning

The **Indian Birds Classifier** is a deep learning–based image classification project designed to identify bird species native to India from images.  
This project was **developed and trained on Kaggle**, utilizing **Kaggle’s free GPU environment (up to 30 hours per week)** for efficient model training and experimentation.

The system allows users to upload bird images and receive accurate species predictions using a trained convolutional neural network / transfer learning model.

---

## 📌 Table of Contents

- 🚀 Features
- 🛠 Tech Stack
- 💻 Development Environment
- 📁 Project Structure
- 📥 Dataset
- 🧠 Model & Training
- ⚙️ Installation & Setup
- ▶️ How to Run the Project

---

## 🚀 Features

✔️ Classifies bird species found in India using image input  
✔️ Trained using deep learning with GPU acceleration  
✔️ Developed in Kaggle Notebook environment  
✔️ Web-based interface for image upload and prediction  
✔️ Scalable and extendable for adding more species  
✔️ Suitable for academic, research, and portfolio use

---

## 🛠 Tech Stack

| Category | Technology |
|--------|------------|
| Programming Language | Python |
| Deep Learning | TensorFlow / Keras |
| Image Processing | OpenCV, Pillow |
| Model Type | CNN / Transfer Learning |
| Web Framework | Flask / Streamlit |
| Platform | Kaggle (GPU) |
| Version Control | Git & GitHub |

---

## 💻 Development Environment

This project was **entirely developed and trained on Kaggle**, using:

- ✅ **Kaggle Notebooks**
- ✅ **Free NVIDIA GPU**
- ✅ **Up to 30 GPU hours per week**
- ✅ Pre-installed deep learning libraries

Kaggle was chosen to ensure faster training, easy experimentation, and reproducibility without local hardware limitations.

---

## 📁 Project Structure
```
Indian-Birds-Classifier/
│
├── models/ # Saved trained models
├── notebooks/ # Kaggle notebooks (training & evaluation)
├── static/ # Static files (CSS, images)
├── templates/ # HTML templates (if Flask is used)
├── utils/ # Helper and preprocessing scripts
├── app.py # Application entry point
├── requirements.txt # Project dependencies
└── README.md # Project documentation
```


---

## 📥 Dataset

The dataset used for this project was sourced from **Kaggle**.

- 📌 Contains images of **Indian bird species**
- 📌 Organized by class (one folder per species)
- 📌 Used for training, validation, and testing

> 🔹 **Note:**  
> The dataset link :
```
https://www.kaggle.com/datasets/srivatsanmk2004/25-indian-birds-species
```

---

## 🧠 Model & Training

- Model training was performed using **Kaggle GPU**
- Image preprocessing includes resizing, normalization, and augmentation
- Transfer learning / CNN architecture used for better accuracy
- Training and evaluation scripts are available in the Kaggle notebooks

Typical training workflow:

1. Load dataset from Kaggle
2. Preprocess and augment images
3. Train model using GPU acceleration
4. Evaluate performance
5. Save best performing model

---

## ⚙️ Installation & Setup

### 📥 Clone the Repository

```bash
git clone https://github.com/SrivatsanMK/Indian-Birds-Classifier.git
cd Indian-Birds-Classifier
```

### 🐍 Create Virtual Environment
```
python -m venv venv
venv\Scripts\activate
```

### 📦 Install Dependencies
```
pip install -r requirements.txt
```

---

## ▶️ How to Run the Project

### Run Locally
```
python app.py
```

### Then open your browser and visit:
- http://127.0.0.1:5000
- Upload a bird image and view the predicted species.
