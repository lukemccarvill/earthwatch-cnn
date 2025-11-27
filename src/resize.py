# Images are from smartphones, of arbitrary size, so need to get to a uniform image dimension, e.g. 224x224px
# This script doe sit using torchvision.transforms module

from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224,224)), # distorts to size 224x224 pixel image
    transforms.ToTensor(), # convert to tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # standard ImageNet normalisation
        std=[0.229, 0.224, 0.225]
    ),
])

