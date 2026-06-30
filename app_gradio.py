import re
import gradio as gr
from PIL import Image
import numpy as np

from src.reconstruction.reconstruct_pipeline import load_model, reconstruct

MODEL_PATH = "models/geosynth_final_best.pth"
model = load_model(MODEL_PATH)

def get_metric(metrics, name, default="--"):
    m = re.search(rf"{name}:\s*([0-9.]+)", metrics)
    return m.group(1) if m else default

def make_mask_visible(mask):
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.max() <= 1:
        mask = mask * 255
    return np.clip(mask, 0, 255).astype(np.uint8)

def metrics_dashboard(metrics):
    acc = get_metric(metrics, "Reconstruction Accuracy")
    cloud = get_metric(metrics, "Cloud Coverage Detected")
    mae = get_metric(metrics, "MAE")
    rmse = get_metric(metrics, "RMSE")
    psnr = get_metric(metrics, "PSNR")
    ssim = get_metric(metrics, "SSIM")

    return f"""
    <div class="metrics-grid">
        <div class="metric"><p>Accuracy</p><h2>{acc}%</h2><span>Final vs GT</span></div>
        <div class="metric"><p>Cloud Area</p><h2>{cloud}%</h2><span>Detected</span></div>
        <div class="metric"><p>PSNR</p><h2>{psnr} dB</h2><span>Quality</span></div>
        <div class="metric"><p>SSIM</p><h2>{ssim}</h2><span>Structure</span></div>
        <div class="metric"><p>RMSE</p><h2>{rmse}</h2><span>Error</span></div>
        <div class="metric"><p>MAE</p><h2>{mae}</h2><span>Error</span></div>
    </div>
    """

def run_demo(uploaded_file, brightness, whiteness):
    if uploaded_file is None:
        raise gr.Error("Please upload a valid .npz satellite sample first.")

    sar, cloudy, mask, raw, final_output, ground_truth, metrics = reconstruct(
        uploaded_file.name, model, brightness, whiteness
    )

    mask_visible = make_mask_visible(mask)

    save_path = "geosynth_reconstructed_output.png"
    Image.fromarray((np.clip(final_output, 0, 1) * 255).astype(np.uint8)).save(save_path)

    return (
        metrics_dashboard(metrics),
        cloudy,
        mask_visible,
        raw,
        final_output,
        ground_truth,
        metrics,
        save_path,
    )

css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

* {
    font-family: 'Inter', sans-serif !important;
}

body {
    background:
        radial-gradient(circle at 85% 0%, rgba(0, 80, 190, 0.28), transparent 30%),
        radial-gradient(circle at 10% 85%, rgba(0, 170, 210, 0.16), transparent 35%),
        linear-gradient(135deg, #eef6ff 0%, #ffffff 46%, #edf7ff 100%) !important;
}

.gradio-container {
    max-width: 1220px !important;
    margin: auto !important;
    padding: 18px !important;
    background: transparent !important;
}

footer {
    display: none !important;
}

#hero {
    position: relative;
    min-height: 250px;
    overflow: hidden;
    border-radius: 30px;
    padding: 34px 38px;
    background:
        linear-gradient(110deg, rgba(255,255,255,.97) 0%, rgba(237,247,255,.94) 55%, rgba(2,30,70,.23) 100%),
        radial-gradient(circle at 82% 30%, rgba(0,91,187,.30), transparent 36%);
    border: 1px solid rgba(147,197,253,.55);
    box-shadow: 0 25px 70px rgba(15,56,110,.14);
}

#hero::before {
    content: "";
    position: absolute;
    inset: 0;
    background:
        linear-gradient(rgba(15,85,170,.055) 1px, transparent 1px),
        linear-gradient(90deg, rgba(15,85,170,.055) 1px, transparent 1px);
    background-size: 36px 36px;
}

#hero::after {
    content: "🌍  🛰️";
    position: absolute;
    right: 55px;
    top: 48px;
    font-size: 72px;
    opacity: .9;
    filter: drop-shadow(0 18px 28px rgba(0,70,150,.25));
}

.hero-inner {
    position: relative;
    z-index: 2;
    max-width: 760px;
}

.kicker {
    color: #f97316;
    font-size: 12px;
    font-weight: 900;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

#hero h1 {
    margin: 0;
    font-size: 48px;
    line-height: 1;
    font-weight: 900;
    color: #061936;
    letter-spacing: -1.4px;
}

#hero h2 {
    margin: 10px 0;
    color: #075ca8;
    font-size: 20px;
    font-weight: 800;
}

#hero p {
    color: #334155;
    font-size: 14px;
    line-height: 1.65;
    max-width: 720px;
}

.badge {
    display: inline-block;
    margin: 7px 6px 0 0;
    padding: 7px 13px;
    border-radius: 999px;
    background: rgba(255,255,255,.9);
    border: 1px solid rgba(96,165,250,.45);
    color: #075985;
    font-size: 11px;
    font-weight: 900;
}

.pipeline {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
    margin: 18px 0 26px 0;
}

.pipe {
    padding: 13px;
    border-radius: 16px;
    background: rgba(255,255,255,.92);
    border: 1px solid rgba(147,197,253,.52);
    box-shadow: 0 12px 34px rgba(20,60,120,.07);
    text-align: center;
}

.pipe b {
    display: block;
    color: #004aad;
    font-size: 13px;
}

.pipe span {
    color: #334155;
    font-size: 11px;
    font-weight: 700;
}

.section-title {
    margin: 22px 0 12px;
    color: #061936;
    font-size: 22px;
    font-weight: 900;
}

.panel {
    padding: 14px;
    border-radius: 24px;
    background: rgba(255,255,255,.78);
    border: 1px solid rgba(147,197,253,.45);
    box-shadow: 0 20px 55px rgba(20,60,120,.09);
}

.upload-box {
    min-height: 155px;
    border-radius: 20px;
    overflow: hidden;
    background: linear-gradient(135deg, #0f766e, #0891b2);
}

.upload-box label {
    color: white !important;
    font-weight: 900 !important;
}

.slider-box {
    min-height: 155px;
    border-radius: 20px;
    padding: 12px;
    background: #061936;
}

.slider-box label {
    color: #eaf5ff !important;
    font-weight: 900 !important;
}

button {
    border-radius: 16px !important;
    min-height: 52px !important;
    background: linear-gradient(90deg, #ff6b00, #f59e0b) !important;
    border: none !important;
    color: white !important;
    font-size: 15px !important;
    font-weight: 900 !important;
    box-shadow: 0 18px 42px rgba(249,115,22,.28) !important;
}

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 10px;
    margin-bottom: 14px;
}

.metric {
    min-height: 86px;
    padding: 13px;
    border-radius: 18px;
    background: linear-gradient(145deg, #ffffff, #e8f4ff);
    border: 1px solid rgba(147,197,253,.55);
    box-shadow: 0 13px 34px rgba(20,60,120,.08);
}

.metric p {
    margin: 0;
    color: #52677f;
    text-transform: uppercase;
    font-size: 10px;
    font-weight: 900;
}

.metric h2 {
    margin: 6px 0 1px;
    color: #061936;
    font-size: 22px;
    font-weight: 900;
}

.metric span {
    color: #0b66b3;
    font-size: 11px;
    font-weight: 800;
}

.workspace {
    padding: 14px;
    border-radius: 26px;
    background: rgba(255,255,255,.72);
    border: 1px solid rgba(147,197,253,.45);
    box-shadow: 0 22px 60px rgba(20,60,120,.09);
}

.image-card {
    padding: 10px;
    border-radius: 22px;
    background: white;
    border: 1px solid rgba(147,197,253,.50);
    box-shadow: 0 15px 44px rgba(20,60,120,.08);
}

.image-title {
    margin: 0 0 8px 4px;
    color: #061936;
    font-size: 13px;
    font-weight: 900;
}

.image-box {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    border-radius: 18px !important;
    overflow: hidden !important;
}

.image-box img {
    border-radius: 16px !important;
    object-fit: contain !important;
    max-height: 250px !important;
}

.image-box label {
    display: none !important;
}

textarea {
    border-radius: 20px !important;
    background: #ffffff !important;
    color: #061936 !important;
    border: 1px solid #d8e8ff !important;
    font-size: 13px !important;
}

.export-box {
    padding: 14px;
    border-radius: 22px;
    background: #061936;
}

.export-box label {
    color: white !important;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css, title="GeoSynth CloudX") as demo:

    gr.HTML("""
    <div id="hero">
        <div class="hero-inner">
            <div class="kicker">Bharatiya Antariksh Hackathon 2026 · PS-02</div>
            <h1>GeoSynth CloudX</h1>
            <h2>Cloud-Free Satellite Image Generation Platform</h2>
            <p>
                AI-powered dashboard for detecting cloud-covered regions in Sentinel-2 imagery,
                reconstructing affected areas using SAR-assisted deep learning, preserving clear pixels,
                evaluating image quality, and exporting cloud-free satellite outputs.
            </p>
            <span class="badge">Sentinel-1 SAR</span>
            <span class="badge">Sentinel-2 Optical</span>
            <span class="badge">Cloud Mask</span>
            <span class="badge">GeoSynth U-Net</span>
            <span class="badge">PSNR / SSIM / RMSE</span>
        </div>
    </div>

    <div class="pipeline">
        <div class="pipe"><b>01 Upload</b><span>NPZ Sample</span></div>
        <div class="pipe"><b>02 Detect</b><span>Cloud Localization</span></div>
        <div class="pipe"><b>03 Mask</b><span>Cloud Region Map</span></div>
        <div class="pipe"><b>04 Reconstruct</b><span>SAR Output</span></div>
        <div class="pipe"><b>05 Evaluate</b><span>Metrics Export</span></div>
    </div>
    """)

    gr.HTML('<div class="section-title">Upload & Pipeline Controls</div>')

    with gr.Row(elem_classes="panel"):
        with gr.Column(scale=2, elem_classes="upload-box"):
            upload = gr.File(label="Upload NPZ Satellite Sample", file_types=[".npz"])
        with gr.Column(scale=1, elem_classes="slider-box"):
            brightness = gr.Slider(60, 95, value=78, step=1, label="Cloud Brightness Threshold")
            whiteness = gr.Slider(20, 70, value=35, step=1, label="Cloud Whiteness Threshold")

    run_btn = gr.Button("🚀 Run GeoSynth Reconstruction Pipeline")

    gr.HTML('<div class="section-title">Performance Overview</div>')
    metrics_cards = gr.HTML("""
    <div class="metrics-grid">
        <div class="metric"><p>Accuracy</p><h2>--</h2><span>Waiting</span></div>
        <div class="metric"><p>Cloud Area</p><h2>--</h2><span>Waiting</span></div>
        <div class="metric"><p>PSNR</p><h2>--</h2><span>Waiting</span></div>
        <div class="metric"><p>SSIM</p><h2>--</h2><span>Waiting</span></div>
        <div class="metric"><p>RMSE</p><h2>--</h2><span>Waiting</span></div>
        <div class="metric"><p>MAE</p><h2>--</h2><span>Waiting</span></div>
    </div>
    """)

    gr.HTML('<div class="section-title">Satellite Processing Workspace</div>')

    with gr.Column(elem_classes="workspace"):

        with gr.Row():
            with gr.Column(elem_classes="image-card"):
                gr.HTML('<div class="image-title">🌥️ Cloudy Sentinel-2 Input</div>')
                cloudy_img = gr.Image(show_label=False, elem_classes="image-box", height=280)

            with gr.Column(elem_classes="image-card"):
                gr.HTML('<div class="image-title">☁️ Detected Cloud Mask</div>')
                mask_img = gr.Image(show_label=False, elem_classes="image-box", height=280)

            with gr.Column(elem_classes="image-card"):
                gr.HTML('<div class="image-title">🎯 Ground Truth Clear Image</div>')
                gt_img = gr.Image(show_label=False, elem_classes="image-box", height=280)

        with gr.Row():
            with gr.Column(elem_classes="image-card"):
                gr.HTML('<div class="image-title">🛰️ Raw SAR-Assisted Reconstruction</div>')
                raw_img = gr.Image(show_label=False, elem_classes="image-box", height=280)

            with gr.Column(elem_classes="image-card"):
                gr.HTML('<div class="image-title">✅ Final Cloud-Free Output</div>')
                final_img = gr.Image(show_label=False, elem_classes="image-box", height=280)

    gr.HTML('<div class="section-title">Evaluation Report & Export</div>')

    with gr.Row():
        with gr.Column(scale=2):
            metrics_text = gr.Textbox(label="Detailed Metrics Report", lines=9, interactive=False)
        with gr.Column(scale=1, elem_classes="export-box"):
            download = gr.File(label="Download Final Reconstructed Image")

    run_btn.click(
        fn=run_demo,
        inputs=[upload, brightness, whiteness],
        outputs=[
            metrics_cards,
            cloudy_img,
            mask_img,
            raw_img,
            final_img,
            gt_img,
            metrics_text,
            download,
        ],
    )

demo.launch(server_name="0.0.0.0")
