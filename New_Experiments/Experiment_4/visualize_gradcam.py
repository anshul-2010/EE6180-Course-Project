import torch
import torchvision.models as models
from gradcam_utils import GradCAM, preprocess_image
import cv2
import os
from NPRDeepfakeDetect.util import Logger, printSet
from NPRDeepfakeDetect.networks.resnet import resnet50
from NPRDeepfakeDetect.options.test_options import TestOptions
import NPRDeepfakeDetect.networks.resnet as resnet
import numpy as np
import random
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from NPRDeepfakeDetect.options.test_options import TestOptions
from NPRDeepfakeDetect.data import create_dataloader

def validate(model, opt):
    data_loader = create_dataloader(opt)

    with torch.no_grad():
        y_true, y_pred = [], []
        for img, label in data_loader:
            in_tens = img
            y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    return acc, ap, r_acc, f_acc, y_true, y_pred

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
seed_torch(100)
DetectionTests = {
           'GANGen-Detection': { 'dataroot'   : '/NPRDeepfakeDetect/dataset/GANGen-Detection/',
                                 'no_resize'  : True,
                                 'no_crop'    : True,
                               },

        'UniversalFakeDetect': { 'dataroot'   : '/NPRDeepfakeDetect/dataset/UniversalFakeDetect/',
                                 'no_resize'  : False, # Due to the different shapes of images in the dataset, resizing is required during batch detection.
                                 'no_crop'    : True,
                               },
                 }


opt = TestOptions().parse(print_options=False)
print(f'Model_path {opt.model_path}')

# Load model (you can replace this with your custom-trained NPR model)
model = models.resnet18(pretrained=True)
model.load_state_dict(torch.load(opt.model_path, map_location='cpu'), strict=True)
model.eval()

# Select the last convolutional layer
target_layer = model.layer4[1].conv2
cam_generator = GradCAM(model, target_layer)

# Load image
image_path = "example_images/fake_1.jpg"  # Replace with actual path
input_tensor, original_image = preprocess_image(image_path)

# Generate Grad-CAM
heatmap = cam_generator.generate(input_tensor)

# Overlay heatmap
heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
overlayed = cv2.addWeighted(heatmap_color, 0.5, original_image, 0.5, 0)

# Save or display
cv2.imwrite("outputs/fake_1_gradcam.jpg", overlayed)
print("Grad-CAM saved to outputs/fake_1_gradcam.jpg")
