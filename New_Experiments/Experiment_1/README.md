# Evaluating Adversarial Robustness of the NPR Model with FGSM Attack

## Overview
This experiment aims to evaluate the adversarial robustness of the trained NPR (Neural Portrait Recognition) model using the Fast Gradient Sign Method (FGSM) attack. FGSM is a popular method for generating adversarial examples by introducing small perturbations to the input data, designed to mislead the model. By testing the model's performance on these adversarial examples, we can measure its vulnerability to such attacks.

## Requirements
- Python 3.x
- PyTorch 1.x or later
- NumPy
- Matplotlib
- Pre-trained NPR model

## Steps
1. **Load the Trained NPR Model:** We start by loading the pre-trained NPR model.
2. **Generate Adversarial Examples:** Using the FGSM attack, we create adversarial examples by modifying the input images slightly.
3. **Evaluate Performance:** We evaluate the model's performance on the adversarial examples and compare the results to the performance on the clean (non-adversarial) images.

## Code
The code involves:
1. Loading the model and input data.
2. Implementing the FGSM attack to perturb the input images.
3. Testing the model's performance on the perturbed data.

## Results
The results are expected to show a significant drop in accuracy when the model is tested on adversarial examples. This will demonstrate the need for improving the model's robustness to adversarial attacks.

## Output
- Accuracy metrics on clean vs adversarial images.
- Visualizations of adversarial examples and their effects on model predictions.

## Usage
```bash
python fgsm_attack.py --model npr_model.pth --data adversarial_data/
