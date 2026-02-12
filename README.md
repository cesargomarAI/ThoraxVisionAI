# 🏥 ThoraxVision AI: Medical Diagnosis Support System
### *Interpretability & Deep Learning in Chest X-Rays*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg.svg)](https://share.streamlit.io/)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)

---

## Descripción

**ThoraxVision** es una herramienta de Inteligencia Artificial avanzada diseñada para asistir a profesionales de la salud en la interpretación de radiografías de tórax. El núcleo del proyecto es un modelo **DenseNet121** entrenado para detectar múltiples patologías simultáneamente.

Lo que hace único a este proyecto no es solo la clasificación, sino su **interpretabilidad**. Mediante una implementación personalizada de **Grad-CAM (Gradient-weighted Class Activation Mapping)**, el sistema genera mapas de calor que señalan exactamente qué región del pulmón está activando el diagnóstico, resolviendo los problemas comunes de visualización en arquitecturas con conexiones densas.

### ✨ Características Clave:
* **🩺 Diagnóstico Multiclase:** Detección de Atelectasia, Cardiomegalia, Efusión, Infiltración, Masa, Nódulo, Neumonía y Neumotórax.
* **🔍 AI FOCUS:** Visualización de la zona de interés mediante mapas de calor de alta precisión.
* **🧠 Lógica Médica Calibrada:** Ajuste de sensibilidad para patologías críticas (como Neumonía) para reducir falsos negativos.
* **📊 Informe Profesional:** Interfaz interactiva para la descarga y revisión de hallazgos.

---

## Description

**ThoraxVision** is an advanced AI tool designed to assist healthcare professionals in interpreting chest X-rays. Built on a **DenseNet121** architecture, it is capable of detecting multiple pathologies simultaneously with high confidence.

The core value of this project lies in its **Interpretability**. By using a custom **Grad-CAM** implementation, the system generates heatmaps that pinpoint the exact lung regions driving the AI's decision. This solves a common technical challenge in DenseNet architectures where traditional Grad-CAM often fails to produce clear anatomical heatmaps.

### ✨ Key Features:
* **🩺 Multi-label Diagnosis:** Real-time detection of Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, and Pneumothorax.
* **🔍 AI FOCUS:** High-precision heatmap visualization of the affected areas.
* **🧠 Calibrated Medical Logic:** Custom sensitivity thresholds for critical conditions (like Pneumonia) to minimize false negatives.
* **📊 Professional Reporting:** Interactive interface for analyzing and downloading clinical findings.

---

## 🛠️ Tech Stack / Tecnologías
* **Backend:** PyTorch, Torchvision.
* **Frontend:** Streamlit.
* **Computer Vision:** OpenCV, Matplotlib, PIL.
* **Model:** Pre-trained DenseNet121 (Fine-tuned for NIH ChestX-ray8).

## 🚀 Installation / Instalación
1. Clone the repo:
   ```bash
   git clone [https://github.com/TU_USUARIO/ThoraxVision.git](https://github.com/TU_USUARIO/ThoraxVision.git)

2. Install dependencies:

    pip install -r requirements.txt

3. Run the app.py

    streamlit run app.py

