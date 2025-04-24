import matplotlib.pyplot as plt
from utils import load_image, invert_transform
from lime_explainer import explain_with_lime
import torch
from NPRDeepfakeDetect.util import Logger, printSet
from NPRDeepfakeDetect.validate import validate
from NPRDeepfakeDetect.networks.resnet import resnet50
from NPRDeepfakeDetect.options.test_options import TestOptions
import NPRDeepfakeDetect.networks.resnet as resnet
import time
from skimage.segmentation import mark_boundaries
import numpy as np
import os

DetectionTests = {
        'UniversalFakeDetect': { 'dataroot'   : 'C:/Users/Dell/Desktop/Courses/Sem_VIII/EE6180/Project/EE6180-Course-Project/UniversalFakeDetect_test/',
                                 'no_resize'  : False, # Due to the different shapes of images in the dataset, resizing is required during batch detection.
                                 'no_crop'    : True,
                               },
                 }

opt = TestOptions().parse(print_options=False)
opt.model_path = "C:/Users/Dell/Desktop/Courses/Sem_VIII/EE6180/Project/EE6180-Course-Project/NPRDeepfakeDetect/model_epoch_last_3090.pth"
print(f'Model_path {opt.model_path}')

# Load model
# get model
model = resnet50(num_classes=1)
model.load_state_dict(torch.load(opt.model_path, map_location='cpu'), strict=True)
model.eval()

# Load image
image_path = "C:/Users/Dell/Desktop/Courses/Sem_VIII/EE6180/Project/EE6180-Course-Project/UniversalFakeDetect_test/dalle/0_real/aadlygmazf.jpg"
image_tensor = load_image(image_path)

# Explain with LIME
explanation = explain_with_lime(model, image_tensor, lambda x: load_image(image_path))

# Get image and mask
temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0],
    positive_only=False,
    num_features=10,
    hide_rest=False
)

# Plot result
plt.imshow(mark_boundaries(temp / 255.0, mask))
plt.title("LIME Explanation on Fake Sample")
plt.axis('off')
plt.show()