# Cross-Dataset Generalization of the NPR Model

## Overview
This experiment evaluates how well the NPR model generalizes across different datasets. The goal is to assess if the model trained on one dataset performs well when tested on images from a completely different dataset, particularly in the context of detecting deepfakes. Cross-dataset generalization is an essential property for models to be deployed in real-world settings where training and testing data may vary.

## Requirements
- Python 3.x
- PyTorch 1.x or later
- NumPy
- Matplotlib
- Pre-trained NPR model
- A source dataset (e.g., FakeFace) and a target dataset (e.g., CelebA)

## Steps
1. **Load Pre-trained Model:** Use the pre-trained model trained on the source dataset.
2. **Test on Target Dataset:** Evaluate the model’s performance on a different target dataset.
3. **Compare Performance:** Compare the accuracy, precision, recall, and F1 score of the model on both datasets.

## Code
The code involves:
1. Loading the source and target datasets.
2. Using the pre-trained model to make predictions on the target dataset.
3. Computing performance metrics on the target dataset.

## Results
We expect the model to show reduced performance on the target dataset, highlighting the need for domain adaptation strategies to improve cross-dataset generalization.

## Output
- Performance metrics comparison between the source and target datasets.
- A report highlighting how much the model's performance deteriorates when switching datasets.

## Usage
```bash
python cross_dataset_generalization.py --model npr_model.pth --source_data FakeFace --target_data CelebA
