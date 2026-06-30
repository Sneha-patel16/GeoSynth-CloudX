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
    mae = get_metric(metrics, "MAE")
    rmse = get_metric(metrics, "RMSE")
    psnr = get_metric(metrics, "PSNR")
    ssim = get_metric(metrics, "SSIM")

    return f"""
    <div class="metrics">
      <div class="metric"><p>Accuracy</p><h2>{acc}%</h2></div>
      <div class="metric"><p>Cloud Coverage</p><h2>{cloud}%</h2></div>
      <div class="metric"><p>PSNR</p><h2>{psnr} dB</h2></div>
      <div class="metric"><p>SSIM</p><h2>{ssim}</h2></div>
      <div class="metric"><p>RMSE</p><h2>{rmse}</h2></div>
      <div class="metric"><p>MAE</p><h2>{mae}</h2></div>
    </div>
    """

def run_demo(uploaded_file, brightness, whiteness):
    if uploaded_file is None:
        raise gr.Error("Upload .npz satellite sample first.")

    sar, cloudy, mask, raw, final_output, ground_truth, metrics = reconstruct(
        uploaded_file.name, model, brightness, whiteness
    )

    save_path = "geosynth_reconstructed_output.png"
    Image.fromarray((final_output * 255).astype(np.uint8)).save(save_path)

    return cloudy, metric_html(metrics), mask, raw, final_output, ground_truth, metrics, save_path

css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

*{font-family:Inter,sans-serif!important}

body{
  background:#f4f8ff!important;
}

.gradio-container{
  max-width:1180px!important;
  margin:auto!important;
  background:#f4f8ff!important;
}

#hero{
  margin-top:20px;
  border-radius:30px;
  padding:38px;
  background:
    linear-gradient(110deg,rgba(255,255,255,.96),rgba(236,246,255,.88)),
    radial-gradient(circle at 90% 10%,rgba(0,91,187,.22),transparent 35%);
  border:1px solid #d8e8ff;
  box-shadow:0 25px 70px rgba(20,60,120,.12);
}

#hero h1{
  font-size:42px;
  margin:0;
  color:#071b3a;
  font-weight:900;
}

#hero h3{
  color:#0b66b3;
  margin:8px 0;
  font-weight:800;
}

#hero p{
  max-width:850px;
  color:#40536d;
  font-size:15px;
  line-height:1.6;
}

.badge{
  display:inline-block;
  padding:8px 14px;
  margin:6px 6px 0 0;
  background:#eaf5ff;
  color:#075985;
  border:1px solid #b9ddff;
  border-radius:999px;
  font-weight:800;
  font-size:12px;
}

.stepbar{
  display:grid;
  grid-template-columns:repeat(5,1fr);
  gap:14px;
  margin:24px 0;
}

.step{
  background:white;
  border:1px solid #d8e8ff;
  border-radius:18px;
  padding:16px;
  text-align:center;
  box-shadow:0 12px 30px rgba(20,60,120,.08);
}

.step b{
  color:#0b3b80;
}

.card{
  background:white;
  border:1px solid #d8e8ff;
  border-radius:24px;
  padding:20px;
  box-shadow:0 18px 50px rgba(20,60,120,.09);
}

.section{
  margin:30px 0 14px;
  font-size:24px;
  font-weight:900;
  color:#071b3a;
}

button{
  border-radius:16px!important;
  min-height:54px!important;
  background:linear-gradient(90deg,#004aad,#0097b2)!important;
  color:white!important;
  border:none!important;
  font-weight:900!important;
  font-size:16px!important;
  box-shadow:0 18px 42px rgba(0,74,173,.25)!important;
}

button:hover{
  transform:translateY(-2px);
}

.metrics{
  display:grid;
  grid-template-columns:repeat(3,1fr);
  gap:14px;
}

.metric{
  background:linear-gradient(135deg,#071b3a,#0b66b3);
  border-radius:20px;
  padding:18px;
  color:white;
  min-height:105px;
}

.metric p{
  margin:0;
  font-size:12px;
  text-transform:uppercase;
  letter-spacing:.5px;
  opacity:.85;
  font-weight:800;
}

.metric h2{
  margin:8px 0 0;
  font-size:30px;
  font-weight:900;
}

.image-box{
  background:white!important;
  border:1px solid #d8e8ff!important;
  border-radius:24px!important;
  box-shadow:0 18px 50px rgba(20,60,120,.09)!important;
  overflow:hidden!important;
}

.image-box img{
  border-radius:18px!important;
}

label{
  color:#0b3b80!important;
  font-weight:800!important;
}

textarea{
  border-radius:20px!important;
  background:white!important;
  color:#071b3a!important;
}

footer{display:none!important}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css, title="GeoSynth CloudX") as demo:

    gr.HTML("""
    <div id="hero">
      <h1>🛰 GeoSynth CloudX</h1>
      <h3>Scientific Dashboard for Cloud-Free Satellite Image Generation</h3>
      <p>
        Upload Sentinel satellite sample, detect cloud-covered regions, reconstruct affected areas using
        SAR-assisted deep learning, compare with ground truth, and export the cloud-free output.
      </p>
      <span class="badge">Sentinel-1 SAR</span>
      <span class="badge">Sentinel-2 Optical</span>
      <span class="badge">GeoSynth U-Net</span>
      <span class="badge">Cloud Mask</span>
      <span class="badge">PSNR / SSIM / RMSE</span>
    </div>

    <div class="stepbar">
      <div class="step"><b>01</b><br>Upload</div>
      <div class="step"><b>02</b><br>Detect Cloud</div>
      <div class="step"><b>03</b><br>Generate Mask</div>
      <div class="step"><b>04</b><br>Reconstruct</div>
      <div class="step"><b>05</b><br>Evaluate</div>
    </div>
    """)

    gr.HTML('<div class="section">Upload & Controls</div>')

    with gr.Row():
        with gr.Column(scale=2, elem_classes="card"):
            upload = gr.File(label="Upload NPZ Satellite Sample", file_types=[".npz"])
        with gr.Column(scale=1, elem_classes="card"):
            brightness = gr.Slider(60, 95, value=78, step=1, label="Cloud Brightness Threshold")
            whiteness = gr.Slider(20, 70, value=35, step=1, label="Cloud Whiteness Threshold")

    run_btn = gr.Button("🚀 Run GeoSynth Reconstruction")

    gr.HTML('<div class="section">Input Overview & Evaluation</div>')

    with gr.Row():
        cloudy_img = gr.Image(label="Cloudy Sentinel-2 Input", elem_classes="image-box", height=390)
        metrics_cards = gr.HTML()

    gr.HTML('<div class="section">Cloud Detection & Reconstruction</div>')

    with gr.Row():
        mask_img = gr.Image(label="Detected Cloud Mask", elem_classes="image-box", height=360)
        raw_img = gr.Image(label="Raw SAR-Assisted Reconstruction", elem_classes="image-box", height=360)

    gr.HTML('<div class="section">Final Output Comparison</div>')

    with gr.Row():
        final_img = gr.Image(label="Final Reconstructed Output", elem_classes="image-box", height=390)
        gt_img = gr.Image(label="Ground Truth Clear Image", elem_classes="image-box", height=390)

    gr.HTML('<div class="section">Detailed Evaluation & Export</div>')

    metrics_text = gr.Textbox(label="Metrics Report", lines=10, interactive=False)
    download = gr.File(label="Download Final Reconstructed Image")

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
