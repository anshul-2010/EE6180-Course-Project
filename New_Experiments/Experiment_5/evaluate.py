import torch
from sklearn.metrics import roc_auc_score

def evaluate(model, dataloader, attack_fn=None, epsilon=0.0):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in dataloader:
        images, labels = images, labels
        if attack_fn is not None:
            images = attack_fn(model, images, labels, epsilon)
        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = correct / total
    try:
        auroc = roc_auc_score(all_labels, all_preds)
    except:
        auroc = 0.0
    return acc, auroc