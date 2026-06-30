import re
import gradio as gr
from PIL import Image
import numpy as np

from src.reconstruction.reconstruct_pipeline import load_model, reconstruct

MODEL_PATH = "models/geosynth_final_best.pth"
model = load_model(MODEL_PATH)


def extract_metric(metrics, name, default="--"):
    match = re.search(rf"{name}:\s*([0-9.]+)", metrics)
    return match.group(1) if match else default


def dashboard_metrics(metrics):
    accuracy = extract_metric(metrics, "Reconstruction Accuracy")
    cloud = extract_metric(metrics, "Cloud Coverage Detected")
    mae = extract_metric(metrics, "MAE")
    rmse = extract_metric(metrics, "RMSE")
    psnr = extract_metric(metrics, "PSNR")
    ssim = extract_metric(metrics, "SSIM")

    return f"""
    <div class="metric-grid">
        <div class="metric-card blue"><span>Accuracy</span><h2>{accuracy}%</h2><p>Final vs Ground Truth</p></div>
        <div class="metric-card green"><span>Cloud Coverage</span><h2>{cloud}%</h2><p>Detected cloud region</p></div>
        <div class="metric-card purple"><span>PSNR</span><h2>{psnr} dB</h2><p>Signal quality</p></div>
        <div class="metric-card orange"><span>SSIM</span><h2>{ssim}</h2><p>Structural similarity</p></div>
        <div class="metric-card teal"><span>RMSE</span><h2>{rmse}</h2><p>Pixel-level error</p></div>
        <div class="metric-card gray"><span>MAE</span><h2>{mae}</h2><p>Absolute error</p></div>
    </div>
    """


def run_demo(uploaded_file, brightness, whiteness):
    if uploaded_file is None:
        raise gr.Error("Please upload a valid .npz satellite sample first.")

    sar, cloudy, mask, raw, final_output, ground_truth, metrics = reconstruct(
        uploaded_file.name,
        model,
        brightness,
        whiteness,
    )

    save_path = "geosynth_reconstructed_output.png"
    Image.fromarray((final_output * 255).astype(np.uint8)).save(save_path)

    return (
        dashboard_metrics(metrics),
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
    background:
        linear-gradient(120deg, rgba(255,255,255,0.95), rgba(239,246,255,0.90)),
        radial-gradient(circle at 80% 0%, rgba(37,99,235,0.22), transparent 35%),
        radial-gradient(circle at 100% 20%, rgba(14,165,233,0.18), transparent 35%) !important;
}

.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
    background: transparent !important;
}

/* MAIN LAYOUT */
#main-shell {
    min-height: 100vh;
}

/* SIDEBAR */
#sidebar {
    background: linear-gradient(180deg, #020617, #0f172a 60%, #111827);
    min-height: 100vh;
    padding: 28px 20px;
    border-right: 1px solid rgba(148,163,184,0.18);
    box-shadow: 10px 0 35px rgba(15,23,42,0.18);
}

#brand {
    color: white;
    margin-bottom: 34px;
}

#brand h1 {
    margin: 0;
    font-size: 24px;
    font-weight: 900;
    letter-spacing: -0.5px;
}

#brand p {
    margin: 8px 0 0 0;
    color: #93c5fd;
    font-size: 12px;
    line-height: 1.5;
}

.nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 13px 14px;
    margin-bottom: 10px;
    color: #cbd5e1;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(148,163,184,0.12);
    border-radius: 14px;
    font-size: 14px;
    font-weight: 700;
}

.nav-item.active {
    background: linear-gradient(90deg, #0369a1, #0f766e);
    color: white;
    box-shadow: 0 12px 30px rgba(14,165,233,0.22);
}

.sidebar-footer {
    margin-top: 30px;
    padding: 15px;
    border-radius: 14px;
    background: rgba(255,255,255,0.05);
    color: #94a3b8;
    font-size: 12px;
    line-height: 1.55;
}

/* CONTENT */
#content {
    padding: 26px 34px 36px 34px;
}

/* HERO */
#hero {
    position: relative;
    overflow: hidden;
    border-radius: 26px;
    padding: 34px 36px;
    margin-bottom: 24px;
    background:
        linear-gradient(100deg, rgba(255,255,255,0.96), rgba(239,246,255,0.88)),
        radial-gradient(circle at 85% 10%, rgba(14,165,233,0.24), transparent 38%);
    border: 1px solid rgba(148,163,184,0.28);
    box-shadow: 0 24px 70px rgba(15,23,42,0.10);
}

#hero::after {
    content: "";
    position: absolute;
    inset: 0;
    background:
        radial-gradient(circle at 85% 5%, rgba(59,130,246,0.20), transparent 30%),
        linear-gradient(135deg, transparent 40%, rgba(2,6,23,0.05));
    pointer-events: none;
}

#hero h1 {
    position: relative;
    z-index: 1;
    margin: 0;
    color: #0f172a;
    font-size: 42px;
    font-weight: 900;
    letter-spacing: -1px;
}

#hero h3 {
    position: relative;
    z-index: 1;
    margin: 10px 0 12px 0;
    color: #0369a1;
    font-weight: 800;
}

#hero p {
    position: relative;
    z-index: 1;
    max-width: 780px;
    color: #475569;
    line-height: 1.6;
    font-size: 15px;
}

.badge-row {
    position: relative;
    z-index: 1;
    margin-top: 16px;
}

.badge {
    display: inline-block;
    padding: 8px 13px;
    margin-right: 8px;
    margin-bottom: 8px;
    border-radius: 999px;
    background: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
    font-size: 12px;
    font-weight: 800;
}

/* SECTION TITLES */
.section-title {
    margin: 24px 0 14px 0;
    padding: 14px 18px;
    border-radius: 16px;
    background: white;
    border-left: 6px solid #0284c7;
    box-shadow: 0 14px 34px rgba(15,23,42,0.07);
    color: #0f172a;
    font-size: 21px;
    font-weight: 900;
}

/* CONTROL PANEL */
.control-panel {
    background: white;
    border: 1px solid rgba(148,163,184,0.25);
    border-radius: 22px;
    padding: 18px;
    box-shadow: 0 18px 50px rgba(15,23,42,0.08);
}

/* BUTTON */
button {
    border-radius: 14px !important;
    min-height: 52px !important;
    font-size: 15px !important;
    font-weight: 900 !important;
    background: linear-gradient(90deg, #0f766e, #0369a1) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 16px 35px rgba(3,105,161,0.28) !important;
    transition: 0.2s ease !important;
}

button:hover {
    transform: translateY(-2px);
    box-shadow: 0 20px 45px rgba(14,165,233,0.35) !important;
}

/* METRICS */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px;
}

.metric-card {
    min-height: 112px;
    border-radius: 20px;
    padding: 18px;
    color: white;
    box-shadow: 0 18px 42px rgba(15,23,42,0.18);
}

.metric-card span {
    font-size: 12px;
    text-transform: uppercase;
    font-weight: 900;
    letter-spacing: 0.5px;
    opacity: 0.9;
}

.metric-card h2 {
    margin: 7px 0 4px 0;
    font-size: 30px;
    font-weight: 900;
}

.metric-card p {
    margin: 0;
    font-size: 12px;
    opacity: 0.88;
}

.blue { background: linear-gradient(135deg, #1d4ed8, #0284c7); }
.green { background: linear-gradient(135deg, #047857, #0f766e); }
.purple { background: linear-gradient(135deg, #6d28d9, #7c3aed); }
.orange { background: linear-gradient(135deg, #ea580c, #f97316); }
.teal { background: linear-gradient(135deg, #0891b2, #0e7490); }
.gray { background: linear-gradient(135deg, #334155, #475569); }

/* IMAGE CARDS */
.image-box {
    border-radius: 22px !important;
    overflow: hidden !important;
    background: white !important;
    border: 1px solid rgba(148,163,184,0.25) !important;
    box-shadow: 0 18px 48px rgba(15,23,42,0.12) !important;
}

.image-box img {
    border-radius: 16px !important;
}

/* WORKFLOW */
.workflow {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
    margin: 16px 0 24px 0;
}

.workflow-card {
    padding: 15px;
    border-radius: 17px;
    background: white;
    border: 1px solid rgba(148,163,184,0.24);
    box-shadow: 0 14px 32px rgba(15,23,42,0.07);
    text-align: center;
}

.workflow-card h4 {
    margin: 5px 0;
    color: #0f172a;
    font-size: 14px;
}

.workflow-card p {
    margin: 0;
    color: #64748b;
    font-size: 12px;
}

/* TEXTBOX */
textarea {
    border-radius: 16px !important;
    background: white !important;
    color: #0f172a !important;
    font-size: 14px !important;
    line-height: 1.55 !important;
}

/* FOOTER */
#footer {
    margin-top: 26px;
    padding: 18px;
    border-radius: 18px;
    background: #0f172a;
    color: #cbd5e1;
    text-align: center;
    font-size: 13px;
}
"""


with gr.Blocks(theme=gr.themes.Soft(), css=css, title="GeoSynth CloudX") as demo:

    with gr.Row(elem_id="main-shell"):
        with gr.Column(scale=1, min_width=230, elem_id="sidebar"):
            gr.HTML("""
            <div id="brand">
                <h1>🛰 GeoSynth CloudX</h1>
                <p>AI Satellite Intelligence Platform<br>SAR-Assisted Cloud Reconstruction</p>
            </div>

            <div class="nav-item active">🏠 Dashboard</div>
            <div class="nav-item">📤 Upload Data</div>
            <div class="nav-item">☁ Cloud Detection</div>
            <div class="nav-item">🛰 SAR Reconstruction</div>
            <div class="nav-item">📊 Evaluation</div>
            <div class="nav-item">📁 Results</div>
            <div class="nav-item">⚙ Settings</div>

            <div class="sidebar-footer">
                <b>Pipeline</b><br>
                Sentinel-1 SAR + Sentinel-2 Optical → Cloud Mask → U-Net Reconstruction → Metrics
            </div>
            """)

        with gr.Column(scale=5, elem_id="content"):

            gr.HTML("""
            <div id="hero">
                <h1>GeoSynth CloudX Dashboard</h1>
                <h3>AI-Based Cloud Detection & SAR-Assisted Satellite Image Reconstruction</h3>
                <p>
                    Upload a satellite sample, detect cloud-covered regions, reconstruct affected pixels using SAR-assisted deep learning,
                    compare results with ground truth, and download cloud-free analysis-ready imagery.
                </p>
                <div class="badge-row">
                    <span class="badge">Sentinel-1 SAR</span>
                    <span class="badge">Sentinel-2 Optical</span>
                    <span class="badge">Cloud Masking</span>
                    <span class="badge">GeoSynth U-Net</span>
                    <span class="badge">PSNR / SSIM / RMSE</span>
                </div>
            </div>
            """)

            gr.HTML("""
            <div class="workflow">
                <div class="workflow-card"><h4>01 Upload</h4><p>NPZ satellite sample</p></div>
                <div class="workflow-card"><h4>02 Detect</h4><p>Cloud localization</p></div>
                <div class="workflow-card"><h4>03 Mask</h4><p>Binary cloud mask</p></div>
                <div class="workflow-card"><h4>04 Reconstruct</h4><p>SAR-assisted output</p></div>
                <div class="workflow-card"><h4>05 Evaluate</h4><p>Metrics & download</p></div>
            </div>
            """)

            gr.HTML('<div class="section-title">Upload Data & Processing Controls</div>')

            with gr.Row():
                with gr.Column(scale=2, elem_classes="control-panel"):
                    upload = gr.File(
                        label="Upload NPZ Satellite Sample",
                        file_types=[".npz"],
                    )
                with gr.Column(scale=2, elem_classes="control-panel"):
                    brightness = gr.Slider(
                        60, 95, value=78, step=1,
                        label="Cloud Brightness Threshold"
                    )
                    whiteness = gr.Slider(
                        20, 70, value=35, step=1,
                        label="Cloud Whiteness Threshold"
                    )

            run_btn = gr.Button("🚀 Run Cloud Detection & Reconstruction", variant="primary")

            gr.HTML('<div class="section-title">Dashboard Metrics</div>')
            metrics_cards = gr.HTML()

            gr.HTML('<div class="section-title">Input Overview</div>')
            cloudy_img = gr.Image(
                label="Cloudy Sentinel-2 Input",
                elem_classes="image-box",
                height=430
            )

            gr.HTML('<div class="section-title">Cloud Detection & SAR-Assisted Reconstruction</div>')
            with gr.Row():
                mask_img = gr.Image(
                    label="Detected Cloud Mask",
                    elem_classes="image-box",
                    height=390
                )
                raw_img = gr.Image(
                    label="Raw SAR-Assisted Reconstruction",
                    elem_classes="image-box",
                    height=390
                )

            gr.HTML('<div class="section-title">Final Output Comparison</div>')
            with gr.Row():
                final_img = gr.Image(
                    label="Final Reconstructed Output",
                    elem_classes="image-box",
                    height=420
                )
                gt_img = gr.Image(
                    label="Ground Truth Clear Image",
                    elem_classes="image-box",
                    height=420
                )

            gr.HTML('<div class="section-title">Evaluation Report & Export</div>')
            metrics_text = gr.Textbox(
                label="Detailed Metrics Report",
                lines=11,
                interactive=False
            )
            download = gr.File(label="Download Final Reconstructed Image")

            gr.HTML("""
            <div id="footer">
                GeoSynth CloudX · ISRO Hackathon Prototype · AI Remote Sensing · SAR + Optical Fusion · Cloud-Free Satellite Reconstruction
            </div>
            """)

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
