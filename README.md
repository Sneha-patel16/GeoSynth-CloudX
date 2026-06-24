# 🛰️ GeoSynth CloudX

### AI-Powered Cloud Detection & SAR-Assisted Satellite Image Reconstruction

GeoSynth CloudX is an intelligent Earth Observation system designed to detect cloud-covered regions in optical satellite imagery and reconstruct affected areas using Sentinel-1 Synthetic Aperture Radar (SAR) data and Deep Learning.

The system leverages cloud-penetrating SAR imagery together with Sentinel-2 optical imagery to improve satellite image usability under adverse atmospheric conditions.

---

## Problem Statement

Cloud coverage is one of the biggest challenges in satellite-based Earth Observation. Optical satellites such as Sentinel-2 frequently capture cloud-obstructed images, making accurate monitoring of land surfaces difficult.

GeoSynth CloudX addresses this challenge by:

* Detecting cloud-covered regions
* Generating cloud masks
* Reconstructing affected areas using SAR-assisted Deep Learning
* Comparing results against clear ground-truth imagery

---

## Key Features

✅ Cloud Detection

✅ Cloud Mask Generation

✅ SAR-Assisted Reconstruction

✅ Deep Learning-Based Restoration

✅ Interactive Gradio Dashboard

✅ Real-Time Evaluation Metrics

✅ Ground Truth Comparison

✅ Visual Analytics

---

## Dataset

The project utilizes paired Sentinel imagery:

### Sentinel-1 SAR

* Weather independent
* Cloud penetrating
* Structural information

### Sentinel-2 Optical

* Multi-spectral imagery
* High visual detail
* Susceptible to cloud obstruction

Dataset Split:

| Dataset    | Samples |
| ---------- | ------- |
| Train      | 3004    |
| Validation | 208     |
| Test       | 328     |

---

## Methodology

### Step 1: Data Preparation

Input:

Sentinel-1 SAR + Cloudy Sentinel-2

Target:

Clear Sentinel-2

---

### Step 2: Cloud Detection

The system identifies cloud-covered regions using:

* Brightness Analysis
* Whiteness Analysis
* Morphological Processing

Output:

Cloud Mask

---

### Step 3: Deep Learning Reconstruction

Model:

GeoSynth U-Net

Input:

16 Channels

* 3 SAR Channels
* 13 Sentinel-2 Channels

Output:

13-channel reconstructed Sentinel-2 image

---

### Step 4: Selective Reconstruction

Detected cloud regions are reconstructed while preserving visible regions from the original optical image.

---

## System Architecture

Sentinel-1 SAR + Cloudy Sentinel-2
↓
Cloud Detection
↓
Cloud Mask Generation
↓
GeoSynth U-Net
↓
SAR-Assisted Reconstruction
↓
Selective Blending
↓
Final Clear Image Estimate
↓
Performance Evaluation

---

## Dashboard

The project includes an interactive Gradio dashboard that provides:

* Sample Selection
* Cloud Detection Visualization
* Cloud Mask Display
* Reconstruction Visualization
* Ground Truth Comparison
* Evaluation Metrics

---

## Evaluation Metrics

The system evaluates reconstruction quality using:

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)
* PSNR (Peak Signal-to-Noise Ratio)
* SSIM (Structural Similarity Index)

Example Results:

| Metric | Value    |
| ------ | -------- |
| MAE    | 0.0738   |
| RMSE   | 0.0980   |
| PSNR   | 20.18 dB |
| SSIM   | 0.6086   |

---

## Technology Stack

### AI / Deep Learning

* PyTorch
* U-Net Architecture

### Data Processing

* NumPy
* SciPy

### Visualization

* Matplotlib
* Gradio

### Remote Sensing

* Sentinel-1 SAR
* Sentinel-2 Optical Imagery

---

## Future Enhancements

* Transformer-Based Reconstruction Models
* Multi-Temporal Satellite Fusion
* Real-Time Satellite Stream Processing
* ISRO Earth Observation Integration
* Large Scale Geospatial Deployment

---

## Author

**Sneha Patel**

B.Tech Data Science

AI/ML • Remote Sensing • Earth Observation • Computer Vision

---

## License

This project is intended for research, educational, and hackathon purposes.

