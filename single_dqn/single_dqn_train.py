import torch
from tqdm import tqdm
from single_dqn_agent import SingleDQNAgent
from torch.utils.data import DataLoader

# Set device for computations
device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "mps:0" if torch.backends.mps.is_available() else "cpu")


def train(single_dqn_agent_, train_loader_, num_epochs=2):
    """
    Train the agent using the training data.

    Args:
        single_dqn_agent_ (SingleDQNAgent): The agent to be trained.
        train_loader_ (DataLoader): DataLoader for the training data.
        num_epochs (int): Number of epochs to train the agent.
    """
    for epoch in range(num_epochs):
        epoch_loss = 0  # Initialize epoch loss
        num_batches = 0  # Initialize number of batches
        train_loader_tqdm = tqdm(train_loader_, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")  # Progress bar
        for texts, labels in train_loader_tqdm:
            texts, labels = texts.to(device).long(), labels.to(device)  # Move data to device
            loss = single_dqn_agent_.step(texts, labels)  # Perform a step of training
            if loss is not None:
                epoch_loss += loss.item()  # Accumulate loss
                num_batches += 1  # Increment batch count
        if num_batches > 0:
            avg_epoch_loss = epoch_loss / num_batches  # Calculate average loss
        else:
            avg_epoch_loss = 0
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}')  # Print epoch loss
