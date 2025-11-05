live DEMO - [# Tumor-X](https://tumor-x.streamlit.app/)

## 🧠 TumorX: Brain Tumor Detection via Deep Learning

An end-to-end deep learning system for automated brain tumor analysis using MRI scans. TumorX performs both **segmentation** and **classification** independently, leveraging U-Net and CNN architectures. Designed for academic and clinical research, it features a modular Streamlit interface and AI-powered PDF reporting.

---

### 🚀 Features

- **Tumor Segmentation** using U-Net with 90%+ Dice coefficient
- **Tumor Classification** using CNN with 95%+ accuracy
- **Streamlit Interface** for real-time image upload and result visualization
- **AI-Powered PDF Reports** with tumor-specific insights and risk assessment
- **Modular Pipeline** for independent execution of segmentation and classification

---

### 🧬 Tumor Types Classified

- **Glioma** – Irregular malignant tumors from glial cells
- **Meningioma** – Well-defined tumors from meninges
- **Pituitary Tumor** – Central region tumors near pituitary gland
- **No Tumor** – Healthy MRI scans

---

### 📊 Model Performance

| Task           | Accuracy | Dice Coefficient | IoU   |
|----------------|----------|------------------|-------|
| Segmentation   | -        | 0.90             | 0.82  |
| Classification | 95.4%    | —                | —     |

---

### 🧠 Architecture Overview

#### 🔹 U-Net (Segmentation)
- Encoder-decoder with skip connections
- Loss: Binary Cross-Entropy + Dice Loss
- Metrics: Accuracy, Dice, IoU

#### 🔹 CNN (Classification)
- 4 Conv layers + MaxPooling + Dense + Softmax
- Loss: Categorical Cross-Entropy
- Metrics: Accuracy, Confusion Matrix, ROC, PR Curve

---

### 🗂️ Dataset

- **Source**: Kaggle Brain Tumor MRI Dataset
- **Total Images**: 4,500 (segmentation + classification)
- **Segmentation**: 2,000 image–mask pairs
- **Classification**: 4 classes, labeled via folder names
- **Preprocessing**: Resizing, normalization, augmentation (flip, rotate, zoom, noise)

---

### 🛠️ Tech Stack

| Component     | Version     | Purpose                          |
|---------------|-------------|----------------------------------|
| Python        | 3.10        | Core language                    |
| TensorFlow    | 2.20.0      | Deep learning framework          |
| Keras         | 3.11.3      | High-level model API             |
| OpenCV        | 4.x         | Image processing                 |
| Streamlit     | 1.28        | Web interface                    |
| Scikit-learn  | 1.3         | Evaluation metrics               |
| Matplotlib & Seaborn | 3.x / 0.12 | Visualization              |

---

### 🖥️ Deployment

- **Interface**: Streamlit app for MRI upload, mask visualization, and classification
- **Output**: Segmentation mask, predicted tumor type, confidence score
- **Report**: Auto-generated PDF with tumor insights, risk level, and disclaimers

---

### 📄 Sample Report Contents

- MRI scan + segmentation overlay
- Tumor classification with confidence
- Risk priority: LOW / MEDIUM / HIGH
- Tumor-specific medical info
- AI disclaimer and reference guide

---

### 📦 Installation

```bash
git clone https://github.com/JustCallMeSidd/Tumor-X.git
cd Tumor-X
pip install -r requirements.txt
streamlit run app.py
```

---

### 📌 Future Enhancements

- Zoom/pan, opacity control, and tooltip overlays
- High-risk alerts and dynamic legends
- Integration with hospital PACS systems
- Cloud-based analysis and multi-modal support

---
---

Let me know if you'd like this broken into sections for GitHub Pages or if you want a shorter version for the repo header.
