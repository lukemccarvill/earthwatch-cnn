#  to see what the normalized photos looks like

import os
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

# --- Config ---
img_dir = "data/images"
target_size = (224, 224)

# ImageNet mean/std (typical for pretrained CNNs like ResNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Transform used for the *model*
model_transform = transforms.Compose([
    transforms.Resize(target_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def unnormalize(tensor, mean, std):
    """
    Undo torchvision.transforms.Normalize for visualisation.
    tensor: (C, H, W) torch tensor
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)

    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    tensor = tensor * std_t + mean_t  # de-standardise

    # clamp to [0,1] so matplotlib can display nicely
    tensor = torch.clamp(tensor, 0.0, 1.0)
    return tensor

# --- Load file list ---
image_files = [
    f for f in os.listdir(img_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

num_images = len(image_files)
if num_images == 0:
    print("No images found in data/images")
    raise SystemExit

cols = 3
rows = (num_images + cols - 1) // cols

plt.figure(figsize=(10, 4 * rows))

for i, filename in enumerate(image_files):
    path = os.path.join(img_dir, filename)

    # 1. Open + fix EXIF orientation
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")

    # 2. Apply the *same* transform your model will see
    t = model_transform(img)              # shape (C, H, W), normalised

    # 3. Un-normalise for plotting
    t_vis = unnormalize(t, IMAGENET_MEAN, IMAGENET_STD)

    # t_vis = t

    # 4. Convert to HWC NumPy for matplotlib
    np_img = t_vis.permute(1, 2, 0).numpy()

    plt.subplot(rows, cols, i + 1)
    plt.imshow(np_img)
    plt.title(filename)
    plt.axis("off")

plt.tight_layout()
plt.show()
