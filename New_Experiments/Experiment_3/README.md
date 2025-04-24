# Interpretability of the NPR Model Using LIME

## Overview
This experiment uses LIME (Local Interpretable Model-Agnostic Explanations) to explain the predictions made by the NPR model. LIME helps in understanding what features of the input image are being utilized by the model to make a prediction, improving the interpretability and trustworthiness of the deepfake detection model.

## Requirements
- Python 3.x
- PyTorch 1.x or later
- NumPy
- Matplotlib
- LIME
- Pre-trained NPR model

## Steps
1. **Load the Trained NPR Model:** We start by loading the pre-trained NPR model.
2. **Apply LIME:** For each prediction, we apply LIME to generate an explanation that highlights the important regions of the image.
3. **Visualize Results:** Visualize the regions of the image that are most influential in the model’s decision-making process.

## Code
The code involves:
1. Loading the model and data.
2. Using the LIME library to explain model predictions.
3. Displaying visual explanations alongside the input image.

## Results
LIME should reveal regions of the image, such as facial features or other artifacts, that contribute most to the model's decision. This helps in understanding the model's decision-making process and checking for overfitting to irrelevant features.

## Output
- Visual explanation for each prediction.
- Summary of the most important features for different classes (real vs fake).

## Usage
```bash
python lime_explainability.py --model npr_model.pth --data images/
