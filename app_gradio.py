
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


def metrics_dashboard(metrics):
    acc = get_metric(metrics, "Reconstruction Accuracy")
    cloud = get_metric(metrics, "Cloud Coverage Detected")
    mae = get_metric(metrics, "MAE")
    rmse = get_metric(metrics, "RMSE")
    psnr = get_metric(metrics, "PSNR")
    ssim = get_metric(metrics, "SSIM")

    return f"""
    <div class="metric-strip">
        <div class="metric-tile"><span>Accuracy</span><strong>{acc}%</strong><small>Final vs GT</small></div>
        <div class="metric-tile"><span>Cloud Area</span><strong>{cloud}%</strong><small>Detected</small></div>
        <div class="metric-tile"><span>PSNR</span><strong>{psnr} dB</strong><small>Quality</small></div>
        <div class="metric-tile"><span>SSIM</span><strong>{ssim}</strong><small>Structure</small></div>
        <div class="metric-tile"><span>RMSE</span><strong>{rmse}</strong><small>Error</small></div>
        <div class="metric-tile"><span>MAE</span><strong>{mae}</strong><small>Error</small></div>
    </div>
    """


def status_ready():
    return """
    <div class="status-card ready">
        <b>System Ready</b>
        <span>Upload NPZ satellite sample and run reconstruction pipeline.</span>
    </div>
    """


def status_done():
    return """
    <div class="status-card done">
        <b>Processing Complete</b>
        <span>Cloud detection, SAR reconstruction, evaluation and export completed.</span>
    </div>
    """


def run_demo(uploaded_file, brightness, whiteness):
    if uploaded_file is None:
        raise gr.Error("Please upload a valid .npz satellite sample first.")

    sar, cloudy, mask, raw, final_output, ground_truth, metrics = reconstruct(
        uploaded_file.name,
        model,
        brightness,
        whiteness
    )

    save_path = "geosynth_reconstructed_output.png"
    Image.fromarray((final_output * 255).astype(np.uint8)).save(save_path)

    return (
        status_done(),
        metrics_dashboard(metrics),
        cloudy,
        mask,
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
    margin: 0 !important;
    background:
        radial-gradient(circle at 82% 8%, rgba(42, 125, 255, 0.26), transparent 26%),
        radial-gradient(circle at 10% 80%, rgba(0, 180, 216, 0.15), transparent 28%),
        linear-gradient(135deg, #eef5ff 0%, #f8fbff 48%, #edf6ff 100%) !important;
}

.gradio-container {
    max-width: 1320px !important;
    margin: auto !important;
    background: transparent !important;
    padding: 18px !important;
}

#top-hero {
    position: relative;
    min-height: 310px;
    border-radius: 34px;
    padding: 38px 42px;
    overflow: hidden;
    background:
        linear-gradient(115deg, rgba(255,255,255,.96) 0%, rgba(240,248,255,.90) 46%, rgba(7,28,65,.22) 100%),
        radial-gradient(circle at 78% 36%, rgba(12, 91, 190, .25), transparent 30%);
    border: 1px solid rgba(125, 170, 230, .35);
    box-shadow: 0 28px 90px rgba(20, 55, 100, .16);
}

#top-hero:before {
    content: "";
    position: absolute;
    inset: 0;
    background:
        linear-gradient(rgba(40,100,180,.06) 1px, transparent 1px),
        linear-gradient(90deg, rgba(40,100,180,.06) 1px, transparent 1px);
    background-size: 38px 38px;
    opacity: .45;
}

#top-hero:after {
    content: "🛰";
    position: absolute;
    right: 65px;
    top: 36px;
    font-size: 92px;
    opacity: .88;
    filter: drop-shadow(0 20px 30px rgba(0,70,160,.20));
}

.hero-content {
    position: relative;
    z-index: 2;
    max-width: 780px;
}

.hero-kicker {
    color: #ff6b00;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 900;
    margin-bottom: 10px;
}

#top-hero h1 {
    margin: 0;
    font-size: 48px;
    line-height: 1.03;
    color: #061936;
    font-weight: 900;
    letter-spacing: -1.4px;
}

#top-hero h2 {
    margin: 12px 0 12px 0;
    color: #075ca8;
    font-size: 19px;
    font-weight: 800;
}

#top-hero p {
    color: #334155;
    line-height: 1.7;
    font-size: 15px;
    max-width: 760px;
}

.hero-badges {
    margin-top: 18px;
}

.hero-badges span {
    display: inline-block;
    margin: 5px 8px 0 0;
    padding: 8px 14px;
    border-radius: 999px;
    background: rgba(255,255,255,.8);
    border: 1px solid rgba(96,165,250,.35);
    color: #075985;
    font-size: 12px;
    font-weight: 900;
}

.pipeline {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 14px;
    margin: 22px 0;
}

.pipe-step {
    padding: 15px 14px;
    border-radius: 20px;
    background: rgba(255,255,255,.84);
    border: 1px solid rgba(148, 190, 235, .42);
    box-shadow: 0 16px 45px rgba(20, 60, 120, .08);
    text-align: center;
}

.pipe-step b {
    display: block;
    color: #004aad;
    font-size: 14px;
    margin-bottom: 3px;
}

.pipe-step span {
    color: #334155;
    font-size: 12px;
    font-weight: 700;
}

.section-title {
    margin: 28px 0 14px 0;
    color: #061936;
    font-size: 25px;
    font-weight: 900;
    letter-spacing: -.3px;
}

.control-shell {
    padding: 18px;
    border-radius: 28px;
    background: rgba(255,255,255,.82);
    border: 1px solid rgba(148, 190, 235, .45);
    box-shadow: 0 24px 70px rgba(20, 60, 120, .10);
}

.upload-panel {
    min-height: 238px;
    border-radius: 24px;
    background:
        linear-gradient(135deg, #0057d9, #00a6c8);
    color: white;
    overflow: hidden;
}

.upload-panel label {
    color: white !important;
}

.slider-panel {
    min-height: 238px;
    border-radius: 24px;
    padding: 14px;
    background: #102033;
    border: 1px solid rgba(255,255,255,.08);
}

.slider-panel label {
    color: #eaf5ff !important;
}

.status-card {
    margin-top: 12px;
    padding: 14px 18px;
    border-radius: 18px;
    background: white;
    border: 1px solid #d8e8ff;
    box-shadow: 0 12px 30px rgba(20, 60, 120, .08);
}

.status-card b {
    display: block;
    color: #061936;
    font-size: 15px;
}

.status-card span {
    color: #52677f;
    font-size: 13px;
}

.status-card.done {
    border-left: 6px solid #10b981;
}

.status-card.ready {
    border-left: 6px solid #0ea5e9;
}

button {
    border-radius: 18px !important;
    min-height: 58px !important;
    background: linear-gradient(90deg, #004aad, #0097b2) !important;
    border: none !important;
    color: white !important;
    font-size: 16px !important;
    font-weight: 900 !important;
    box-shadow: 0 20px 50px rgba(0, 74, 173, .26) !important;
    transition: all .2s ease !important;
}

button:hover {
    transform: translateY(-2px);
    box-shadow: 0 26px 60px rgba(0, 151, 178, .32) !important;
}

.metric-strip {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 13px;
    margin-bottom: 18px;
}

.metric-tile {
    min-height: 104px;
    padding: 16px;
    border-radius: 22px;
    background:
        linear-gradient(145deg, rgba(255,255,255,.92), rgba(232,244,255,.88));
    border: 1px solid rgba(148, 190, 235, .55);
    box-shadow: 0 18px 50px rgba(20, 60, 120, .10);
}

.metric-tile span {
    display: block;
    color: #52677f;
    text-transform: uppercase;
    font-size: 11px;
    font-weight: 900;
    letter-spacing: .6px;
}

.metric-tile strong {
    display: block;
    margin-top: 7px;
    color: #061936;
    font-size: 25px;
    font-weight: 900;
}

.metric-tile small {
    display: block;
    margin-top: 3px;
    color: #0b66b3;
    font-size: 12px;
    font-weight: 800;
}

.result-grid {
    padding: 18px;
    border-radius: 30px;
    background: rgba(255,255,255,.66);
    border: 1px solid rgba(148,190,235,.40);
    box-shadow: 0 28px 80px rgba(20,60,120,.10);
}

.image-box {
    border-radius: 26px !important;
    overflow: hidden !important;
    background: rgba(255,255,255,.92) !important;
    border: 1px solid rgba(148,190,235,.55) !important;
    box-shadow: 0 22px 60px rgba(20,60,120,.11) !important;
}

.image-box img {
    border-radius: 18px !important;
    object-fit: contain !important;
}

.image-box label {
    background: transparent !important;
    color: #06315f !important;
    font-weight: 900 !important;
    font-size: 14px !important;
}

textarea {
    border-radius: 24px !important;
    background: rgba(255,255,255,.95) !important;
    color: #061936 !important;
    border: 1px solid #d8e8ff !important;
    font-size: 14px !important;
    line-height: 1.55 !important;
}

.export-panel {
    border-radius: 26px;
    padding: 18px;
    background: #061936;
    color: white;
    border: 1px solid rgba(255,255,255,.10);
    box-shadow: 0 24px 70px rgba(6,25,54,.22);
}

footer {
    display: none !important;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css, title="GeoSynth CloudX") as demo:

    gr.HTML("""
    <div id="top-hero">
        <div class="hero-content">
            <div class="hero-kicker">Bharatiya Antariksh Hackathon 2026 · PS-02</div>
            <h1>GeoSynth CloudX</h1>
            <h2>Cloud-Free Satellite Image Generation Platform</h2>
            <p>
                A scientific AI dashboard that detects cloud-covered regions in Sentinel-2 imagery,
                reconstructs affected areas using SAR-assisted deep learning, preserves clear pixels,
                evaluates output quality, and exports cloud-free satellite imagery.
            </p>
            <div class="hero-badges">
                <span>Sentinel-1 SAR</span>
                <span>Sentinel-2 Optical</span>
                <span>Cloud Mask</span>
                <span>GeoSynth U-Net</span>
                <span>PSNR / SSIM / RMSE</span>
            </div>
        </div>
    </div>

    <div class="pipeline">
        <div class="pipe-step"><b>01 Upload</b><span>NPZ Satellite Sample</span></div>
        <div class="pipe-step"><b>02 Detect</b><span>Cloud Localization</span></div>
        <div class="pipe-step"><b>03 Mask</b><span>Cloud Region Map</span></div>
        <div class="pipe-step"><b>04 Reconstruct</b><span>SAR-Assisted Output</span></div>
        <div class="pipe-step"><b>05 Evaluate</b><span>Metrics & Export</span></div>
    </div>
    """)

    gr.HTML('<div class="section-title">Upload & Pipeline Controls</div>')

    with gr.Row(elem_classes="control-shell"):
        with gr.Column(scale=2, elem_classes="upload-panel"):
            upload = gr.File(
                label="Upload NPZ Satellite Sample",
                file_types=[".npz"]
            )
        with gr.Column(scale=1, elem_classes="slider-panel"):
            brightness = gr.Slider(60, 95, value=78, step=1, label="Cloud Brightness Threshold")
            whiteness = gr.Slider(20, 70, value=35, step=1, label="Cloud Whiteness Threshold")

    run_btn = gr.Button("🚀 Run GeoSynth Reconstruction Pipeline")
    status_box = gr.HTML(status_ready())

    gr.HTML('<div class="section-title">Performance Overview</div>')
    metrics_cards = gr.HTML("""
    <div class="metric-strip">
        <div class="metric-tile"><span>Accuracy</span><strong>--</strong><small>Waiting</small></div>
        <div class="metric-tile"><span>Cloud Area</span><strong>--</strong><small>Waiting</small></div>
        <div class="metric-tile"><span>PSNR</span><strong>--</strong><small>Waiting</small></div>
        <div class="metric-tile"><span>SSIM</span><strong>--</strong><small>Waiting</small></div>
        <div class="metric-tile"><span>RMSE</span><strong>--</strong><small>Waiting</small></div>
        <div class="metric-tile"><span>MAE</span><strong>--</strong><small>Waiting</small></div>
    </div>
    """)

    gr.HTML('<div class="section-title">Satellite Processing Workspace</div>')

    with gr.Column(elem_classes="result-grid"):
        with gr.Row():
            cloudy_img = gr.Image(label="Cloudy Sentinel-2 Input", elem_classes="image-box", height=470)
            mask_img = gr.Image(label="Detected Cloud Mask", elem_classes="image-box", height=470)

        with gr.Row():
            raw_img = gr.Image(label="Raw SAR-Assisted Reconstruction", elem_classes="image-box", height=470)
            final_img = gr.Image(label="Final Cloud-Free Output", elem_classes="image-box", height=470)

        with gr.Row():
            gt_img = gr.Image(label="Ground Truth Clear Image", elem_classes="image-box", height=470)

    gr.HTML('<div class="section-title">Evaluation Report & Export</div>')

    with gr.Row():
        with gr.Column(scale=2):
            metrics_text = gr.Textbox(label="Detailed Metrics Report", lines=11, interactive=False)
        with gr.Column(scale=1, elem_classes="export-panel"):
            download = gr.File(label="Download Final Reconstructed Image")

    run_btn.click(
        fn=run_demo,
        inputs=[upload, brightness, whiteness],
        outputs=[
            status_box,
            metrics_cards,
            cloudy_img,
            mask_img,
            raw_img,
            final_img,
            gt_img,
            metrics_text,
            download
        ],
    )

demo.launch(server_name="0.0.0.0")
