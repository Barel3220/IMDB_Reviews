import torch
from tqdm import tqdm
from agent import Agent
from torch.utils.data import DataLoader

# Set device for computations
device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "mps:0" if torch.backends.mps.is_available() else "cpu")


def evaluate(agent_, test_loader_):
    """
    Evaluate the agent using the test data.

    Args:
        agent_ (Agent): The agent to be evaluated.
        test_loader_ (DataLoader): DataLoader for the test data.
    """
    agent_.qnetwork_local.eval()  # Set network to evaluation mode
    correct = 0  # Initialize correct predictions count
    total = 0  # Initialize total predictions count
    test_loader_tqdm = tqdm(test_loader_, desc="Evaluating", unit="batch")  # Progress bar
    with torch.no_grad():
        for texts, labels in test_loader_tqdm:
            texts, labels = texts.to(device).long(), labels.to(device)  # Move data to device
            outputs = agent_.act(texts)  # Get predictions
            correct += (outputs == labels).sum().item()  # Count correct predictions
            total += labels.size(0)  # Count total predictions

    print(f'Accuracy: {100 * correct / total:.2f}%')  # Print accuracy
