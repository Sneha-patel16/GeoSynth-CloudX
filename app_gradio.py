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

    sar, cloudy, mask, raw, final_output, ground_truth, metrics = reconstruct(
        file_path,
        model,
        brightness,
        whiteness,
    )

    save_path = "geosynth_reconstructed_output.png"
    Image.fromarray((final_output * 255).astype(np.uint8)).save(save_path)

    return cloudy, metrics, mask, raw, final_output, ground_truth, save_path


css = """
body {
    background: linear-gradient(-45deg, #020617, #0f172a, #1e1b4b, #0c4a6e);
    background-size: 400% 400%;
    animation: gradientMove 12s ease infinite;
}

@keyframes gradientMove {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.gradio-container {
    max-width: 1250px !important;
    margin: auto !important;
    background: transparent !important;
}

#hero {
    text-align: center;
    padding: 34px 24px;
    border-radius: 28px;
    color: white;
    background: rgba(15, 23, 42, 0.72);
    border: 1px solid rgba(148, 163, 184, 0.35);
    box-shadow: 0 20px 60px rgba(0,0,0,0.45);
    backdrop-filter: blur(18px);
    margin-bottom: 24px;
}

#hero h1 {
    font-size: 46px;
    margin-bottom: 6px;
}

#hero h3 {
    color: #bae6fd;
    margin-top: 4px;
}

#hero p {
    color: #cbd5e1;
    font-size: 16px;
}

.gr-box, .gr-panel, .block {
    border-radius: 22px !important;
}

label, .label-wrap span {
    color: #e5e7eb !important;
    font-weight: 700 !important;
}

textarea {
    font-size: 16px !important;
    font-weight: 600 !important;
    background: rgba(15, 23, 42, 0.92) !important;
    color: #d1fae5 !important;
    border-radius: 18px !important;
}

button {
    border-radius: 18px !important;
    font-size: 18px !important;
    font-weight: 800 !important;
    padding: 14px !important;
    background: linear-gradient(90deg, #2563eb, #06b6d4, #22c55e) !important;
    border: none !important;
    color: white !important;
    box-shadow: 0 12px 30px rgba(34,211,238,0.35) !important;
}

button:hover {
    transform: scale(1.01);
    transition: 0.2s ease;
}

.image-box {
    border-radius: 24px !important;
    overflow: hidden !important;
    border: 1px solid rgba(125, 211, 252, 0.35) !important;
    box-shadow: 0 18px 45px rgba(0,0,0,0.35) !important;
}

.image-box img {
    border-radius: 20px !important;
}

.section-title {
    color: white;
    padding: 14px 18px;
    border-left: 5px solid #38bdf8;
    background: rgba(15,23,42,0.65);
    border-radius: 16px;
    margin-top: 22px;
    margin-bottom: 12px;
    backdrop-filter: blur(10px);
}

.footer {
    text-align: center;
    color: #cbd5e1;
    padding: 20px;
    margin-top: 20px;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css, title="GeoSynth CloudX") as demo:

    gr.HTML("""
    <div id="hero">
        <h1>🛰️ GeoSynth CloudX</h1>
        <h3>AI-Based Cloud Detection & SAR-Assisted Satellite Image Reconstruction</h3>
        <p>
        Upload an NPZ satellite sample, detect cloud-obstructed regions, reconstruct affected areas,
        and compare the output with the original ground truth image.
        </p>
    </div>
    """)

    with gr.Row():
        upload = gr.File(
            label="📤 Upload NPZ Satellite Sample",
            file_types=[".npz"],
            scale=2
        )

    with gr.Row():
        brightness = gr.Slider(
            60, 95, value=78, step=1,
            label="☁️ Cloud Brightness Sensitivity"
        )
        whiteness = gr.Slider(
            20, 70, value=35, step=1,
            label="⚪ Cloud Whiteness Sensitivity"
        )

    run_btn = gr.Button("🚀 Run Cloud Detection & Reconstruction", variant="primary")

    gr.HTML('<div class="section-title"><h2>📌 Uploaded Image & Evaluation Metrics</h2></div>')

    with gr.Row():
        cloudy_img = gr.Image(
            label="Cloudy Sentinel-2 Input",
            elem_classes="image-box",
            height=390
        )
        metrics_box = gr.Textbox(
            label="📊 Reconstruction Metrics",
            lines=14,
            interactive=False
        )

    gr.HTML('<div class="section-title"><h2>☁️ Cloud Detection & Raw Reconstruction</h2></div>')

    with gr.Row():
        mask_img = gr.Image(
            label="Detected Cloud Mask",
            elem_classes="image-box",
            height=360
        )
        raw_img = gr.Image(
            label="SAR-Assisted Raw Reconstruction",
            elem_classes="image-box",
            height=360
        )

    gr.HTML('<div class="section-title"><h2>✅ Final Output vs Ground Truth</h2></div>')

    with gr.Row():
        final_img = gr.Image(
            label="Final Cloud-Free Reconstructed Output",
            elem_classes="image-box",
            height=390
        )
        gt_img = gr.Image(
            label="Original Ground Truth Clear Image",
            elem_classes="image-box",
            height=390
        )

    download = gr.File(label="⬇️ Download Final Reconstructed Image")

    gr.HTML("""
    <div class="footer">
        <b>GeoSynth CloudX</b> | Cloud Detection → SAR Reconstruction → Selective Blending → Evaluation
    </div>
    """)

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
        ],
    )

demo.launch()
