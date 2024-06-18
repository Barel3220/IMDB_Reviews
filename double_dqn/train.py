import torch
from tqdm import tqdm
from agent import DoubleDQNAgent
from torch.utils.data import DataLoader

# Set device for computations
device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "mps:0" if torch.backends.mps.is_available() else "cpu")


def train(agent_, train_loader_, num_epochs=20):
    """
    Train the agent using the training data.

    Args:
        agent_ (DoubleDQNAgent): The agent to be trained.
        train_loader_ (DataLoader): DataLoader for the training data.
        num_epochs (int): Number of epochs to train the agent.
    """
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        train_loader_tqdm = tqdm(train_loader_, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")
        for texts, labels in train_loader_tqdm:
            texts, labels = texts.to(device).long(), labels.to(device)
            loss = agent_.step(texts, labels)

            if loss is not None:
                epoch_loss += loss.item()
                num_batches += 1

        if num_batches > 0:
            avg_epoch_loss = epoch_loss / num_batches
        else:
            avg_epoch_loss = 0
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}')

    return agent_