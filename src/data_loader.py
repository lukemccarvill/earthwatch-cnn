import torch
import torchvision
import torchvision.transforms as transforms
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
import pandas as pd

classes = ('Poor', 'Moderate', 'Good')
labels_csv_path = "data/labels.csv"


# --- transforms:

transform = transforms.Compose([
    transforms.Resize((224, 224)), # resolution of images used in the CNN!
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # normalize around 0.5???

batch_size = 4 # may need to update this?

# --- tiny custom dataset; before i have all photos

class SimpleImageDataset(Dataset):
    def __init__(self, img_dir, labels_csv, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        df = pd.read_csv(labels_csv)
        self.labels =  dict(zip(df["filename"], df['label']))

        # keep only labelled image files in this folder
        self.image_files = [
            f for f in os.listdir(img_dir)
            # if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            if f in self.labels
        ]

        if len(self.image_files) ==0:
            raise RuntimeError(f"No images found in {img_dir}")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        filename = self.image_files[idx]
        path = os.path.join(self.img_dir, filename)

        img = Image.open(path)
        img = ImageOps.exif_transpose(img) # this it to deal with 90deg rotation EXIF
        img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        label_name = self.labels[filename]
        label_idx = classes.index(label_name)

        return img, label_idx

# datasets:
trainset = SimpleImageDataset("data/images/training", labels_csv_path, transform=transform)
trainloader = DataLoader(
    trainset,
    batch_size=min(batch_size, len(trainset)), # handle tiny datasets
    shuffle=True,
    num_workers=0  #apparently 0 is simpler on windows?
)

testset  = SimpleImageDataset("data/images/testing", labels_csv_path, transform=transform)
testloader = DataLoader(
    testset,
    batch_size=min(batch_size, len(testset)), # handle tiny datasets
    shuffle=False,
    num_workers=0  #apparently 0 is simpler on windows?
)

# --- plotting helper:

def imshow(img):
    # unnormalize
    img = img /2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1, 2, 0)))
    plt.show()

# --- get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

imshow(torchvision.utils.make_grid(images)) # show images

print(' '.join(classes[l] for l in labels))


