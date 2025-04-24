import torch
from data_loader import get_data_loaders
from fgsm_attack import fgsm_attack
from evaluate import evaluate
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

# get model
model = resnet50(num_classes=1)
model.load_state_dict(torch.load(opt.model_path, map_location='cpu'), strict=True)
model.eval()

# for testSet in DetectionTests.keys():
#     dataroot = DetectionTests[testSet]['dataroot']
#     printSet(testSet)

#     accs = [];aps = []
#     print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
#     for v_id, val in enumerate(os.listdir(dataroot)):
#         opt.dataroot = '{}/{}'.format(dataroot, val)
#         opt.classes  = '' #os.listdir(opt.dataroot) if multiclass[v_id] else ['']
#         opt.no_resize = DetectionTests[testSet]['no_resize']
#         opt.no_crop   = DetectionTests[testSet]['no_crop']
#         acc, ap, _, _, _, _ = validate(model, opt)
#         accs.append(acc);aps.append(ap)
#         print("({} {:12}) acc: {:.1f}; ap: {:.1f}".format(v_id, val, acc*100, ap*100))
#     print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id+1,'Mean', np.array(accs).mean()*100, np.array(aps).mean()*100));print('*'*25) 

# Load data
dataloader = get_data_loaders('C:/Users/Dell/Desktop/Courses/Sem_VIII/EE6180/Project/EE6180-Course-Project/UniversalFakeDetect_test/dalle', batch_size=4)

# Evaluate on clean data
acc, auroc = evaluate(model, dataloader)
print(f'Clean Accuracy: {acc:.4f}, AUROC: {auroc:.4f}')

# Evaluate on adversarial data
epsilons = [0.001, 0.01, 0.03, 0.05]
for eps in epsilons:
    acc, auroc = evaluate(model, dataloader, attack_fn=fgsm_attack, epsilon=eps)
    print(f'FGSM Attack (ε={eps}): Accuracy: {acc:.4f}, AUROC: {auroc:.4f}')