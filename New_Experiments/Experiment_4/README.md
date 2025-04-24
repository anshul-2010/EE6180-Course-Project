# Saliency Map-Based Explainability Test

## Overview
This experiment utilizes saliency maps to visualize which pixels in the input image most affect the output decision of the NPR model. By calculating the gradients of the output with respect to the input image, saliency maps highlight the most influential regions, thus improving the interpretability of the model.

## Requirements
- Python 3.x
- PyTorch 1.x or later
- NumPy
- Matplotlib
- Pre-trained NPR model

## Steps
1. **Load the Trained NPR Model:** Start by loading the pre-trained model.
2. **Generate Saliency Map:** Compute the gradients of the output with respect to the input image.
3. **Visualize Saliency Map:** Display the saliency map overlaid on the original image to highlight important regions.

## Code
The code involves:
1. Loading the model and image data.
2. Computing the gradients to generate the saliency map.
3. Displaying the map on top of the image.

## Results
Saliency maps will help to identify the important regions used by the model for decision-making. In deepfake detection, we expect the saliency map to highlight regions around the eyes, mouth, or other facial features.

## Output
- Saliency map showing the influence of different parts of the image on the model’s prediction.

## Usage
```bash
python saliency_map_explainability.py --model npr_model.pth --data images/
