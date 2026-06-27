import gradio as gr
from PIL import Image
import numpy as np

from src.reconstruction.reconstruct_pipeline import load_model, reconstruct

MODEL_PATH = "models/geosynth_final_best.pth"
model = load_model(MODEL_PATH)


def run_demo(uploaded_file, brightness, whiteness):
    if uploaded_file is None:
        raise gr.Error("Please upload a .npz satellite sample first.")

    file_path = uploaded_file.name

    sar, cloudy, mask, raw, final_output, gt, metrics = reconstruct(
        file_path,
        model,
        brightness,
        whiteness
    )

    save_path = "geosynth_reconstructed_output.png"
    Image.fromarray((final_output * 255).astype(np.uint8)).save(save_path)

    return cloudy, metrics, mask, raw, final_output, gt, save_path


css = """
.gradio-container {
    max-width: 1180px !important;
    margin: auto !important;
}
#title {
    text-align: center;
    padding: 20px;
    border-radius: 18px;
    background: linear-gradient(90deg, #0f172a, #1e3a8a, #065f46);
    color: white;
    margin-bottom: 22px;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css, title="GeoSynth CloudX") as demo:

    gr.HTML("""
    <div id="title">
        <h1>🛰️ GeoSynth CloudX</h1>
        <h3>Whole-Image Cloud Detection & SAR-Assisted Reconstruction</h3>
        <p>Upload NPZ → Detect Clouds → Reconstruct Detected Regions → Compare with Ground Truth</p>
    </div>
    """)

    upload = gr.File(
        label="📤 Upload NPZ Satellite Sample",
        file_types=[".npz"]
    )

    with gr.Row():
        brightness = gr.Slider(
            70, 95, value=76, step=1,
            label="☁️ Cloud Brightness Sensitivity"
        )
        whiteness = gr.Slider(
            25, 70, value=32, step=1,
            label="⚪ Cloud Whiteness Sensitivity"
        )

    run_btn = gr.Button("🚀 Detect Cloud & Reconstruct", variant="primary")

    gr.Markdown("## 📌 Uploaded Image & Accuracy Metrics")
    with gr.Row():
        cloudy_img = gr.Image(label="Uploaded / Cloudy Sentinel-2", height=360)
        metrics_box = gr.Textbox(label="📊 Accuracy and Quality Metrics", lines=13)

    gr.Markdown("## ☁️ Cloud Detection and Raw Reconstruction")
    with gr.Row():
        mask_img = gr.Image(label="Detected Cloud Mask", height=330)
        raw_img = gr.Image(label="Raw SAR-Assisted Reconstruction", height=330)

    gr.Markdown("## ✅ Final Output vs Ground Truth")
    with gr.Row():
        final_img = gr.Image(label="Final Reconstructed Output", height=360)
        gt_img = gr.Image(label="Original Ground Truth", height=360)

    download = gr.File(label="⬇️ Download Final Reconstructed Image")

    run_btn.click(
        fn=run_demo,
        inputs=[upload, brightness, whiteness],
        outputs=[
            cloudy_img,
            metrics_box,
            mask_img,
            raw_img,
            final_img,
            gt_img,
            download
        ]
    )

demo.launch()
