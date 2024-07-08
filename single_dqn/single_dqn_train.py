import torch
from tqdm import tqdm
from single_dqn_agent import SingleDQNAgent
from torch.utils.data import DataLoader
from plotter import Plotter
from single_dqn_evaluate import evaluate

# Set device for computations
device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "mps:0" if torch.backends.mps.is_available() else "cpu"
)


def train(single_dqn_agent_, train_loader_, test_loader_, num_epochs=20, percent_string=''):
    """
    Train the agent using the training data.

    Args:
        single_dqn_agent_ (SingleDQNAgent): The agent to be trained.
        train_loader_ (DataLoader): DataLoader for the training data.
        test_loader_ (DataLoader): DataLoader for the test data.
        num_epochs (int): Number of epochs to train the agent.
        percent_string (str): Which Dataset is running and which to save.
    """
    plotter = Plotter()

    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        train_loader_tqdm = tqdm(train_loader_, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")
        for texts, labels in train_loader_tqdm:
            texts, labels = texts.to(device).long(), labels.to(device)

            # Pass texts and labels to the agent's step function, which handles reward calculation and memory update
            loss = single_dqn_agent_.step(texts, labels)

            if loss is not None:
                epoch_loss += loss.item()
                num_batches += 1

        if num_batches > 0:
            avg_epoch_loss = epoch_loss / num_batches
        else:
            avg_epoch_loss = 0

        is_last_epoch = False

        if (epoch + 1) == num_epochs:
            is_last_epoch = True

        # Evaluate the agent on the test set
        accuracy, f_score, g_mean_score = evaluate(single_dqn_agent_, test_loader_, is_last_epoch, percent_string)

        # Log metrics
        plotter.log_epoch(epoch + 1, avg_epoch_loss, accuracy, f_score, g_mean_score)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}, Accuracy: {accuracy:.4f}, \
        F-score: {f_score:.4f}, G-mean: {g_mean_score:.4f}')

    plotter.plot_metrics('../plots/single_dqn_training_metrics_' + percent_string + '.png')
