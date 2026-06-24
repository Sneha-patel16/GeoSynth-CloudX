import os
import matplotlib.pyplot as plt

from src.reconstruction.reconstruct_pipeline import load_model, reconstruct

TEST_DIR = "/content/drive/MyDrive/GeoSynth-CloudX/data/dataset/test"
MODEL_PATH = "/content/drive/MyDrive/GeoSynth-CloudX/outputs/checkpoints/geosynth_final_best.pth"
SAVE_DIR = "/content/drive/MyDrive/GeoSynth-CloudX/outputs/isro_final_demo"

os.makedirs(SAVE_DIR, exist_ok=True)

model = load_model(MODEL_PATH)

files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(".npz")])

for i, file in enumerate(files[:30]):
    path = os.path.join(TEST_DIR, file)
    out = reconstruct(path, model)

    plt.figure(figsize=(18, 4))

    titles = [
        "Sentinel-1 SAR",
        "Cloudy Sentinel-2",
        "Detected Cloud Mask",
        "Raw Model Reconstruction",
        "Final Blended Output",
        "Ground Truth",
    ]

    images = [
        out["sar"],
        out["cloudy"],
        out["mask"],
        out["raw_model"],
        out["output"],
        out["ground_truth"],
    ]

    for j in range(6):
        plt.subplot(1, 6, j + 1)
        if j == 2:
            plt.imshow(images[j], cmap="gray")
        else:
            plt.imshow(images[j])
        plt.title(titles[j], fontsize=9)
        plt.axis("off")

    plt.suptitle(file)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"isro_final_{i+1}.png"), dpi=200)
    plt.close()

print("Saved:", SAVE_DIR)
