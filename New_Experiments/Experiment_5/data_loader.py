from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random

def get_data_loaders(data_path, batch_size=32, subset_size=2000):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = datasets.ImageFolder(data_path, transform=transform)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    subset = Subset(dataset, indices[:subset_size])
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    return loader