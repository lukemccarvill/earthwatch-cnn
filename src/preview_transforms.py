import os
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# Path to your images
img_dir = "data/images"

target_size = (224, 224)

# Get a list of image files
image_files = [
    f for f in os.listdir(img_dir)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
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

    img = Image.open(path).convert("RGB")
    img = ImageOps.exif_transpose(img)   # apply EXIF orientation. need to do this later with pytorch too
    resized = img.resize(target_size, Image.Resampling.LANCZOS)

    plt.subplot(rows, cols, i + 1)
    plt.imshow(resized)
    plt.title(filename)
    plt.axis("off")

plt.tight_layout()
plt.show()
