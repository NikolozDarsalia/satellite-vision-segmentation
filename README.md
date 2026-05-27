# Satellite Vision: Multi-Modal Semantic Segmentation

This repository showcases the application of Deep Learning to Earth Observation (EO) data. It features two core implementations: **Building Footprint Extraction** from high-resolution optical imagery and **Flood Inundation Mapping** using Synthetic Aperture Radar (SAR).

---

## 🛰️ Project Overview

### 1. Building Footprint Extraction (Optical)
* **Architecture:** U-Net (Vanilla implementation)
* **Dataset:** Massachusetts Buildings Dataset (RGB Aerial Imagery)
* **Objective:** Automated mapping of urban structures to assist in urban planning and population estimation.
* **Key Challenge:** Addressing pixel-level class imbalance (~11% building coverage) using optimized thresholding ($T=0.567$) and focal loss.

### 2. Flood Detection & Disaster Response (SAR)
* **Architecture:** ResNet-backboned U-Net
* **Dataset:** ETCI 2021 (Sentinel-1 SAR)
* **Objective:** Near-Real-Time flood mapping that operates through cloud cover and night-time conditions.
* **Key Engineering:** Multi-polarization (VV/VH) data fusion and heavy class-imbalance mitigation (107:1 ratio) to ensure high recall for emergency responders.

---

## 🛠️ Tech Stack & Tooling

* **Framework:** PyTorch
* **Architectures:** U-Net, ResNet (Backbone)
* **Package Management:** [uv](https://github.com/astral-sh/uv) (Extremely fast Python package installer and resolver)
* **Visualization:** Matplotlib, OpenCV, Geoplotlib
* **Hardware Acceleration:** NVIDIA T4/P100 (Mixed Precision Training)

---

## 📁 Repository Structure

* `building_segmentation.ipynb`: Full pipeline for optical building extraction.
* `flood-segmentation.ipynb`: Specialized SAR pipeline for disaster monitoring.
* `pyproject.toml` & `uv.lock`: Modern environment configuration for reproducible builds.

---

## 📈 Performance Highlights

| Project | Metric | Result |
| :--- | :--- | :--- |
| **Building Extraction** | Global F1-Score | **0.750** |
| **Flood Detection** | Validation F1-Score | **0.837+** |

---

## 🚀 Getting Started

This project uses `uv` for lightning-fast environment setup. To replicate this environment:

1.  **Install uv:**
    ```bash
    curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
    ```

2.  **Sync Dependencies:**
    ```bash
    uv sync
    ```

3.  **Run Notebooks:**
    Use your preferred editor (VS Code/Jupyter) and point the kernel to the `.venv` created by uv.

---

## 🔮 Future Roadmap

* **Polygonization:** Adding post-processing scripts to convert masks into GeoJSON polygons for GIS software.
* **Foundation Models:** Fine-tuning IBM/NASA's Prithvi model for improved SAR-optical fusion.
* **Temporal Analysis:** Implementing change-detection (Siamese Networks) to isolate new flood waters from permanent water bodies.

---

**Authors:** [Nikoloz Darsalia](https://github.com/NikolozDarsalia), [Tina Sikharulidze](https://github.com/tinasikharulidze)
