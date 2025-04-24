import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from torchvision.models import resnet18
from NPRDeepfakeDetect.util import Logger, printSet
from NPRDeepfakeDetect.validate import validate
from NPRDeepfakeDetect.networks.resnet import resnet50
from NPRDeepfakeDetect.options.test_options import TestOptions
import NPRDeepfakeDetect.networks.resnet as resnet
import time
import numpy as np
import os

# Load pretrained model (modify path accordingly)

DetectionTests = {
        'UniversalFakeDetect': { 'dataroot'   : 'C:/Users/Dell/Desktop/Courses/Sem_VIII/EE6180/Project/EE6180-Course-Project/UniversalFakeDetect_test/',
                                 'no_resize'  : False, # Due to the different shapes of images in the dataset, resizing is required during batch detection.
                                 'no_crop'    : True,
                               },
                 }

opt = TestOptions().parse(print_options=False)
opt.model_path = "C:/Users/Dell/Desktop/Courses/Sem_VIII/EE6180/Project/EE6180-Course-Project/NPRDeepfakeDetect/model_epoch_last_3090.pth"
print(f'Model_path {opt.model_path}')

# Load your trained model (assumed binary classification: 0=real, 1=fake)
model = resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 2)  # Binary classifier
model.load_state_dict(torch.load("npr_model.pth", map_location='cpu'))
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def occlusion_sensitivity(image, model, label, occ_size=32, stride=16):
    image_tensor = transform(image).unsqueeze(0)
    _, _, H, W = image_tensor.shape
    heatmap = np.zeros((H, W))

    for y in range(0, H - occ_size + 1, stride):
        for x in range(0, W - occ_size + 1, stride):
            occluded = image_tensor.clone()
            occluded[:, :, y:y+occ_size, x:x+occ_size] = 0  # Occlude with black
            with torch.no_grad():
                out = model(occluded)
                prob = F.softmax(out, dim=1)[0, label].item()
            heatmap[y:y+occ_size, x:x+occ_size] += prob

    # Normalize heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap

# Example usage:
image_path = "sample_fake.png"
image = Image.open(image_path).convert("RGB")
label = 1  # Fake image

heatmap = occlusion_sensitivity(image, model, label)

# Visualization
plt.imshow(image)
plt.imshow(heatmap, cmap='jet', alpha=0.5)
plt.colorbar(label="Model Confidence for True Class")
plt.title("Occlusion Sensitivity Heatmap")
plt.axis("off")
plt.show()