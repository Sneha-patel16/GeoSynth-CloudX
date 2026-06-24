import os
import gradio as gr
import numpy as np
from PIL import Image

from src.reconstruction.reconstruct_pipeline import load_model, reconstruct

TEST_DIR = "/content/drive/MyDrive/GeoSynth-CloudX/data/dataset/test"
MODEL_PATH = "/content/drive/MyDrive/GeoSynth-CloudX/outputs/checkpoints/geosynth_final_best.pth"

model = load_model(MODEL_PATH)
files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(".npz")])

def run_from_dataset(sample, brightness, whiteness):
    path = os.path.join(TEST_DIR, sample)
    sar, cloudy, mask, raw, output, gt, metrics = reconstruct(path, model, brightness, whiteness)
    return sar, cloudy, mask, raw, output, gt, metrics

with gr.Blocks(theme=gr.themes.Soft(), title="GeoSynth CloudX") as demo:
    gr.Markdown("""
    # 🛰️ GeoSynth CloudX  
    ### AI-powered Cloud Detection & SAR-Assisted Satellite Image Reconstruction
    """)

    with gr.Row():
        sample = gr.Dropdown(files, label="Select Satellite Dataset Sample", value=files[0])

    with gr.Row():
        brightness = gr.Slider(80, 98, value=92, step=1, label="Cloud Brightness Threshold")
        whiteness = gr.Slider(50, 85, value=70, step=1, label="Cloud Whiteness Threshold")

    run_btn = gr.Button("🚀 Detect Cloud & Reconstruct Image")

    metrics = gr.Textbox(label="📊 Accuracy / Evaluation Metrics", lines=6)

    gr.Markdown("## Input & Cloud Detection")
    with gr.Row():
        sar = gr.Image(label="Sentinel-1 SAR Input")
        cloudy = gr.Image(label="Cloudy Sentinel-2 Input")
        mask = gr.Image(label="Detected Cloud Mask")

    gr.Markdown("## Reconstruction & Ground Truth Matching")
    with gr.Row():
        raw = gr.Image(label="Raw Model Reconstruction")
        output = gr.Image(label="Final Reconstructed Output")
        gt = gr.Image(label="Ground Truth Clear Image")

    run_btn.click(
        fn=run_from_dataset,
        inputs=[sample, brightness, whiteness],
        outputs=[sar, cloudy, mask, raw, output, gt, metrics]
    )

demo.launch(share=True, debug=True)
