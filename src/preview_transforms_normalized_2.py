# Use a non-interactive matplotlib backend (must be set before importing pyplot)
import matplotlib
matplotlib.use("Agg")

import os
from PIL import Image, ImageOps, UnidentifiedImageError
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import numpy as np

# --- Config ---
img_dir = "/home/teaching/earthwatch-cnn/data/Great_UK_water_blitz/"
target_size = (224, 224)

# ImageNet mean/std (typical for pretrained CNNs like ResNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

model_transform = transforms.Compose([
    transforms.Resize(target_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def unnormalize(tensor, mean, std):
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    tensor = tensor * std_t + mean_t
    tensor = torch.clamp(tensor, 0.0, 1.0)
    return tensor

# --- Load file list ---
image_files = [
    f for f in os.listdir(img_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]
image_files = sorted(image_files)

image_files = image_files[2000:]
image_files = image_files[:-6500]

num_images = len(image_files)
if num_images == 0:
    raise SystemExit("No images selected after slicing — check your indices")

cols = 3
rows = (num_images + cols - 1) // cols
fig = plt.figure(figsize=(10, 4 * rows))

# Precompute checkpoints once
n_files = num_images
delta_checkpoints = max(1, n_files // 20)
checkpoints = set(np.arange(0, n_files, delta_checkpoints))

plot_index = 1
for i, filename in enumerate(image_files):
    if i not in checkpoints:
        continue

    path = os.path.join(img_dir, filename)
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        print(f"Skipping invalid or unreadable image: {path} -> {e}")
        continue

    # Apply model transforms and unnormalize for plotting
    t = model_transform(img)              # (C, H, W)
    t_vis = unnormalize(t, IMAGENET_MEAN, IMAGENET_STD)
    np_img = t_vis.permute(1, 2, 0).numpy()

    ax = fig.add_subplot(rows, cols, plot_index)
    ax.imshow(np_img)
    ax.set_title(filename)
    ax.axis("off")
    plot_index += 1

# Save combined figure (no X required)
out_preview = "/home/teaching/earthwatch-cnn/results/preview_normalized_sample.png"
os.makedirs(os.path.dirname(out_preview), exist_ok=True)
plt.tight_layout()
plt.savefig(out_preview, bbox_inches="tight")
plt.close(fig)  # free memory
print(f"Saved preview to: {out_preview}")

