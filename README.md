# 😷 Face Mask Detection using Transfer Learning | Deep Learning + Streamlit + Docker + AWS

This project implements a **Face Mask Detection System** that classifies whether a person is wearing a mask or not in real time using **Transfer Learning**.  
The application is built with **TensorFlow/Keras**, deployed using **Streamlit**, containerized using **Docker**, and hosted on **AWS EC2** for scalable access.

---

## 🚀 Project Overview

The model classifies images into two categories:

- ✔️ **With Mask**
- ❌ **Without Mask**

Using a pretrained CNN model (VGG16/ResNet50), the system extracts rich facial features and fine-tunes them for mask-based classification.

---

## 🧰 Tech Stack

| Component | Technology |
|----------|------------|
| Deep Learning | TensorFlow / Keras |
| Transfer Learning | VGG16 / ResNet50 |
| Frontend UI | Streamlit |
| Deployment | Docker & AWS EC2 |
| Language | Python |

---

## 📌 Key Features

🔹 Achieved **98–99% accuracy** after fine-tuning  
🔹 **Real-time prediction** using webcam or uploaded image  
🔹 **Data Augmentation & Regularization** to overcome overfitting  
🔹 **Dockerized for seamless deployment** anywhere  
🔹 **Cloud-hosted** on AWS EC2 for scalability and reliability

---



---

## 🖥️ How to Run Locally

```bash
# Clone the repository
git clone https://github.com/ramashishpanchal/face-mask-detection
cd face-mask-detection

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py


## Run with docker

# Pull image from Docker Hub
docker pull ramashishpanchal/face-mask:v3

# Run container
docker run -p 8501:8501 ramashishpanchal/face-mask:v3


## 📂 Directory Structure

