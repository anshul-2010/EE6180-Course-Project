import torchvision.transforms as transforms
from PIL import Image
import torch

def load_image(image_path, img_size=256):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image)

def invert_transform(tensor):
    return (tensor * 0.5) + 0.5  # Invert normalization to view image