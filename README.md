# 🛰️ GeoSynth CloudX

GeoSynth CloudX is an AI-powered satellite image restoration system for cloud detection and SAR-assisted reconstruction.

## Features

- Sentinel-1 SAR input visualization
- Cloudy Sentinel-2 image analysis
- Cloud mask detection using brightness and whiteness thresholds
- SAR-assisted deep learning reconstruction
- Final reconstructed clear image estimate
- Ground truth comparison
- Evaluation metrics: MAE, RMSE, PSNR, SSIM
- Interactive Gradio dashboard

## Tech Stack

- Python
- PyTorch
- NumPy
- SciPy
- Scikit-image
- Gradio
- Sentinel-1 SAR
- Sentinel-2 multispectral imagery

## Pipeline

Sentinel-1 SAR + Cloudy Sentinel-2  
→ Cloud Detection  
→ Cloud Mask Generation  
→ GeoSynth U-Net Reconstruction  
→ Selective Blending  
→ Clear Image Estimate  
→ Ground Truth Evaluation

## Run

```bash
pip install -r requirements.txt
python app_gradio.py
