from dataset_loader import get_data_loader
from model_loader import load_model
from evaluate import evaluate_model
import torch
from NPRDeepfakeDetect.networks.resnet import resnet50
from NPRDeepfakeDetect.options.test_options import TestOptions

# Paths (replace with real paths)
ffpp_path = "datasets/ffpp_test"
celebdf_path = "datasets/celebdf_test"

# Load model
DetectionTests = {
        'UniversalFakeDetect': { 'dataroot'   : 'C:/Users/Dell/Desktop/Courses/Sem_VIII/EE6180/Project/EE6180-Course-Project/UniversalFakeDetect_test/',
                                 'no_resize'  : False, # Due to the different shapes of images in the dataset, resizing is required during batch detection.
                                 'no_crop'    : True,
                               },
                 }

opt = TestOptions().parse(print_options=False)
opt.model_path = "C:/Users/Dell/Desktop/Courses/Sem_VIII/EE6180/Project/EE6180-Course-Project/NPRDeepfakeDetect/model_epoch_last_3090.pth"
print(f'Model_path {opt.model_path}')

# get model
model = resnet50(num_classes=1)
model.load_state_dict(torch.load(opt.model_path, map_location='cpu'), strict=True)
model.eval()

# Evaluate on FF++ (in-distribution)
ffpp_loader, _ = get_data_loader(ffpp_path)
ffpp_acc, ffpp_auroc = evaluate_model(model, ffpp_loader)
print(f"FF++ Test Accuracy: {ffpp_acc:.4f}, AUROC: {ffpp_auroc:.4f}")

# Evaluate on Celeb-DF (out-of-distribution)
celebdf_loader, _ = get_data_loader(celebdf_path)
celebdf_acc, celebdf_auroc = evaluate_model(model, celebdf_loader)
print(f"Celeb-DF Test Accuracy: {celebdf_acc:.4f}, AUROC: {celebdf_auroc:.4f}")