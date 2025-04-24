# Occlusion Sensitivity Analysis

## Overview
This experiment analyzes the impact of occluding different regions of an input image on the model’s performance. By occluding parts of the image and observing the changes in classification accuracy, we can determine which regions of the image are most critical for the model’s decision-making process. This can be particularly useful for detecting biases or unintended dependencies in the model.

## Requirements
- Python 3.x
- PyTorch 1.x or later
- NumPy
- Matplotlib
- Pre-trained NPR model

## Steps
1. **Load the Trained NPR Model:** We load the pre-trained model.
2. **Occlusion Strategy:** A sliding window is used to occlude different parts of the image.
3. **Evaluate Performance:** We evaluate the model’s performance after occluding each region and visualize the impact.

## Code
The code involves:
1. Loading the model and input data.
2. Sliding a occlusion window over the image.
3. Recording the performance of the model after each occlusion.
4. Visualizing a heatmap of sensitivity.

## Results
The results will show which regions of the image the model depends on most for classification. Regions that cause a large drop in performance upon occlusion are considered the most important for decision-making.

## Output
- A heatmap showing occlusion sensitivity.
- Comparison of model performance after occlusion.

## Usage
```bash
python occlusion_sensitivity.py --model npr_model.pth --data images/
