import torch
from lime import lime_image
from skimage.segmentation import mark_boundaries
import numpy as np

def explain_with_lime(model, image_tensor, transform_fn, batch_size=1):
    model.eval()
    model.cpu()

    def predict(images):
        images = [transform_fn(img) for img in images]
        batch = torch.stack(images).cpu()
        with torch.no_grad():
            outputs = model(batch)
            return outputs.softmax(1).numpy()

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image_tensor.permute(1, 2, 0).numpy(),
        predict,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )
    
    return explanation