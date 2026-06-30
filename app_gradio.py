
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

def metric_html(metrics):
    acc = get_metric(metrics, "Reconstruction Accuracy")
    cloud = get_metric(metrics, "Cloud Coverage Detected")
    psnr = get_metric(metrics, "PSNR")
    ssim = get_metric(metrics, "SSIM")
    rmse = get_metric(metrics, "RMSE")
    mae = get_metric(metrics, "MAE")

    return f"""
    <div class="metric-row">
        <div class="metric-card"><span>Accuracy</span><b>{acc}%</b></div>
        <div class="metric-card"><span>Cloud</span><b>{cloud}%</b></div>
        <div class="metric-card"><span>PSNR</span><b>{psnr} dB</b></div>
        <div class="metric-card"><span>SSIM</span><b>{ssim}</b></div>
        <div class="metric-card"><span>RMSE</span><b>{rmse}</b></div>
        <div class="metric-card"><span>MAE</span><b>{mae}</b></div>
    </div>
    """

def run_demo(uploaded_file, brightness, whiteness):
    if uploaded_file is None:
        raise gr.Error("Please upload a valid .npz satellite sample.")

    sar, cloudy, mask, raw, final_output, ground_truth, metrics = reconstruct(
        uploaded_file.name, model, brightness, whiteness
    )

    save_path = "geosynth_reconstructed_output.png"
    Image.fromarray((final_output * 255).astype(np.uint8)).save(save_path)

    return cloudy, metric_html(metrics), mask, raw, final_output, ground_truth, metrics, save_path

css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

* {
    font-family: Inter, sans-serif !important;
}

body {
    background:
        linear-gradient(180deg, rgba(255,255,255,0.92), rgba(241,247,255,0.95)),
        radial-gradient(circle at 85% 8%, rgba(0,83,179,0.30), transparent 30%),
        radial-gradient(circle at 92% 18%, rgba(14,165,233,0.22), transparent 34%) !important;
}

.gradio-container {
    max-width: 1280px !important;
    margin: auto !important;
    background: transparent !important;
}

/* top black strip like PPT */
#topbar {
    height: 72px;
    background: #030712;
    border-radius: 0 0 0 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 34px;
    color: white;
    margin-bottom: 0;
}

.logo-left {
    font-size: 17px;
    font-weight: 900;
    color: #38bdf8;
}

.logo-right {
    font-size: 26px;
    font-weight: 900;
    letter-spacing: 3px;
    color: white;
}

/* hero with earth/satellite style */
#hero {
    min-height: 250px;
    padding: 34px 42px;
    border-radius: 0 0 28px 28px;
    background:
        radial-gradient(circle at 82% 55%, rgba(59,130,246,0.32), transparent 27%),
        radial-gradient(circle at 88% 70%, rgba(125,211,252,0.25), transparent 34%),
        linear-gradient(105deg, #ffffff 0%, #f8fbff 45%, #dbeafe 100%);
    border: 1px solid #d8e8ff;
    box-shadow: 0 24px 70px rgba(15, 23, 42, 0.10);
    position: relative;
    overflow: hidden;
}

#hero::after {
    content: "🛰";
    position: absolute;
    right: 70px;
    top: 40px;
    font-size: 96px;
    opacity: 0.95;
    transform: rotate(-18deg);
}

#hero::before {
    content: "";
    position: absolute;
    right: -120px;
    bottom: -190px;
    width: 520px;
    height: 520px;
    border-radius: 50%;
    background:
        radial-gradient(circle, rgba(255,255,255,0.35), rgba(59,130,246,0.18) 35%, rgba(30,64,175,0.20) 60%, transparent 70%);
}

#hero h1 {
    margin: 0;
    font-size: 46px;
    color: #081b3d;
    font-weight: 900;
    letter-spacing: -1px;
}

#hero h3 {
    margin: 8px 0 10px;
    font-size: 21px;
    color: #0057b8;
    font-weight: 800;
}

#hero p {
    max-width: 760px;
    color: #40536d;
    font-size: 15px;
    line-height: 1.6;
}

.badge {
    display: inline-block;
    padding: 8px 14px;
    margin: 8px 6px 0 0;
    background: white;
    color: #075985;
    border: 1px solid #bfdbfe;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 800;
}

/* steps like mock diagram */
.steps {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 14px;
    margin: 26px 0;
}

.step {
    background: white;
    border: 1px solid #d7e8ff;
    border-radius: 18px;
    padding: 14px;
    text-align: center;
    box-shadow: 0 12px 28px rgba(15, 23, 42, 0.07);
    color: #0b3b80;
    font-weight: 900;
}

.step small {
    display: block;
    color: #64748b;
    font-weight: 700;
    margin-top: 3px;
}

/* upload panel */
.panel {
    background: white;
    border: 1px solid #d7e8ff;
    border-radius: 24px;
    padding: 20px;
    box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
}

.section-title {
    margin: 32px 0 16px;
    color: #081b3d;
    font-size: 25px;
    font-weight: 900;
}

.section-title::after {
    content: "";
    display: block;
    width: 160px;
    height: 4px;
    margin-top: 8px;
    border-radius: 10px;
    background: linear-gradient(90deg, #ff5a00, #0057ff);
}

button {
    border-radius: 16px !important;
    min-height: 54px !important;
    background: linear-gradient(90deg, #004aad, #0097b2) !important;
    color: white !important;
    border: none !important;
    font-size: 16px !important;
    font-weight: 900 !important;
    box-shadow: 0 18px 42px rgba(0, 74, 173, 0.25) !important;
}

button:hover {
    transform: translateY(-2px);
}

/* metric mini cards */
.metric-row {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 12px;
}

.metric-card {
    background: linear-gradient(135deg, #061a3a, #0057b8);
    color: white;
    border-radius: 18px;
    padding: 16px;
    min-height: 88px;
    box-shadow: 0 15px 34px rgba(0, 74, 173, 0.18);
}

.metric-card span {
    display: block;
    font-size: 11px;
    text-transform: uppercase;
    opacity: 0.85;
    font-weight: 800;
    letter-spacing: 0.5px;
}

.metric-card b {
    display: block;
    font-size: 22px;
    margin-top: 8px;
}

/* images bigger */
.image-box {
    background: white !important;
    border: 1px solid #d7e8ff !important;
    border-radius: 24px !important;
    overflow: hidden !important;
    box-shadow: 0 18px 45px rgba(15, 23, 42, 0.09) !important;
}

.image-box img {
    border-radius: 16px !important;
    object-fit: contain !important;
}

label {
    color: #0b3b80 !important;
    font-weight: 900 !important;
}

textarea {
    background: white !important;
    border-radius: 20px !important;
    color: #0f172a !important;
}

footer {
    display: none !important;
}

#footer {
    margin: 34px 0 20px;
    padding: 18px;
    background: white;
    border: 1px solid #d7e8ff;
    border-radius: 20px;
    text-align: center;
    color: #334155;
    font-weight: 700;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css, title="GeoSynth CloudX") as demo:

    gr.HTML("""
    <div id="topbar">
        <div class="logo-left">ISRO · GeoSynth CloudX</div>
        <div class="logo-right">H2S</div>
    </div>

    <div id="hero">
        <h1>GeoSynth CloudX</h1>
        <h3>AI-Based Cloud Detection & SAR-Assisted Satellite Image Reconstruction</h3>
        <p>
            A scientific dashboard for converting cloud-contaminated Sentinel satellite imagery into
            cloud-free, analysis-ready outputs using cloud masking, SAR-optical fusion and GeoSynth U-Net reconstruction.
        </p>
        <span class="badge">Sentinel-1 SAR</span>
        <span class="badge">Sentinel-2 Optical</span>
        <span class="badge">Cloud Mask</span>
        <span class="badge">GeoSynth U-Net</span>
        <span class="badge">PSNR / SSIM / RMSE</span>
    </div>

    <div class="steps">
        <div class="step">01 Upload<small>NPZ satellite sample</small></div>
        <div class="step">02 Detect<small>Cloud localization</small></div>
        <div class="step">03 Mask<small>Binary cloud region</small></div>
        <div class="step">04 Reconstruct<small>SAR-assisted output</small></div>
        <div class="step">05 Evaluate<small>Metrics & export</small></div>
    </div>
    """)

    gr.HTML('<div class="section-title">Upload & Control Panel</div>')

    with gr.Row():
        with gr.Column(scale=2, elem_classes="panel"):
            upload = gr.File(label="Upload NPZ Satellite Sample", file_types=[".npz"])
        with gr.Column(scale=1, elem_classes="panel"):
            brightness = gr.Slider(60, 95, value=78, step=1, label="Cloud Brightness Threshold")
            whiteness = gr.Slider(20, 70, value=35, step=1, label="Cloud Whiteness Threshold")

    run_btn = gr.Button("🚀 Run Cloud Detection & Reconstruction")

    gr.HTML('<div class="section-title">Input Overview & Quantitative Evaluation</div>')

    with gr.Row():
        with gr.Column(scale=3):
            cloudy_img = gr.Image(label="Cloudy Sentinel-2 Input", elem_classes="image-box", height=500)
        with gr.Column(scale=2):
            metrics_cards = gr.HTML()

    gr.HTML('<div class="section-title">Cloud Localization & SAR-Assisted Reconstruction</div>')

    with gr.Row():
        mask_img = gr.Image(label="Detected Cloud Mask", elem_classes="image-box", height=460)
        raw_img = gr.Image(label="Raw SAR-Assisted Reconstruction", elem_classes="image-box", height=460)

    gr.HTML('<div class="section-title">Final Cloud-Free Output vs Ground Truth</div>')

    with gr.Row():
        final_img = gr.Image(label="Final Reconstructed Output", elem_classes="image-box", height=500)
        gt_img = gr.Image(label="Ground Truth Clear Image", elem_classes="image-box", height=500)

    gr.HTML('<div class="section-title">Evaluation Report & Download</div>')

    metrics_text = gr.Textbox(label="Detailed Metrics Report", lines=10, interactive=False)
    download = gr.File(label="Download Final Reconstructed Image")

    gr.HTML("""
    <div id="footer">
        GeoSynth CloudX · Cloud Detection · SAR + Optical Fusion · Cloud-Free Satellite Reconstruction · ISRO Hackathon Prototype
    </div>
    """)

    run_btn.click(
        fn=run_demo,
        inputs=[upload, brightness, whiteness],
        outputs=[
            cloudy_img,
            metrics_cards,
            mask_img,
            raw_img,
            final_img,
            gt_img,
            metrics_text,
            download,
        ],
    )

demo.launch(server_name="0.0.0.0")
