import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loader(dataset_path, batch_size=32, img_size=256):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False), dataset.classes