import torch
import numpy as np
from tqdm import tqdm
from single_dqn_agent import SingleDQNAgent
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix

# Set device for computations
device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "mps:0" if torch.backends.mps.is_available() else "cpu"
)


def g_mean(all_labels, all_preds):
    """
    Calculate the G-mean metric.

    Args:
        all_labels (list or np.ndarray): True labels.
        all_preds (list or np.ndarray): Predicted labels.

    Returns:
        float: G-mean score.
    """
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return np.sqrt(sensitivity * specificity)


def evaluate(single_dqn_agent_, test_loader_):
    """
    Evaluate the agent using the test data.

    Args:
        single_dqn_agent_ (SingleDQNAgent): The agent to be evaluated.
        test_loader_ (DataLoader): DataLoader for the test data.
    """
    single_dqn_agent_.qnetwork.eval()  # Set network to evaluation mode
    correct = 0  # Initialize correct predictions count
    total = 0  # Initialize total predictions count
    all_labels = []
    all_preds = []
    test_loader_tqdm = tqdm(test_loader_, desc="Evaluating", unit="batch")  # Progress bar
    with torch.no_grad():
        for texts, labels in test_loader_tqdm:
            texts, labels = texts.to(device).long(), labels.to(device)  # Move data to device
            outputs = single_dqn_agent_.act(texts)  # Get predictions
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(outputs.cpu().numpy())
            correct += (outputs == labels).sum().item()  # Count correct predictions
            total += labels.size(0)  # Count total predictions

    accuracy = 100 * correct / total
    f_score = f1_score(all_labels, all_preds)
    g_mean_score = g_mean(all_labels, all_preds)

    return accuracy, f_score, g_mean_score
