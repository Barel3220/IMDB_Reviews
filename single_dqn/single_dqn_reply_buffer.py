import random
from collections import deque, namedtuple
import torch
from torch.nn.utils.rnn import pad_sequence


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed):
        """
        Initialize the ReplayBuffer.

        Args:
            buffer_size (int): Maximum size of the buffer.
            batch_size (int): Size of each training batch.
            seed (int): Random seed for reproducibility.
        """
        self.memory = deque(maxlen=buffer_size)  # Buffer to store experiences
        self.batch_size = batch_size  # Batch size for sampling
        self.experience = namedtuple("Experience", field_names=["text", "label"])  # Define experience tuple
        self.seed = random.seed(seed)  # Set random seed

    def add(self, text, label):
        """
        Add a new experience to memory.

        Args:
            text (torch.Tensor): Tensor containing token ids.
            label (int): Sentiment label.
        """
        e = self.experience(text.to(device), int(label))  # Ensure label is an integer and text is on the correct device
        self.memory.append(e)  # Add experience to memory

    def sample(self):
        """
        Sample a batch of experiences from memory.

        Returns:
            tuple: A tuple containing batches of texts and labels.
        """
        experiences = random.sample(self.memory, k=self.batch_size)  # Sample experiences

        texts = [e.text for e in experiences if e is not None]  # Extract texts
        labels = torch.tensor([e.label for e in experiences if e is not None], dtype=torch.long).to(device)
        max_len = max([len(text) for text in texts])  # Find the maximum length of texts
        texts_padded = pad_sequence(
            [torch.cat([text, torch.tensor([1] * (max_len - len(text))).to(device)]).unsqueeze(0) for text in texts],
            batch_first=True, padding_value=1
        ).squeeze(1)  # Pad texts and move to device
        return texts_padded, labels  # Return padded texts and labels

    def __len__(self):
        """
        Return the current size of internal memory.
        """
        return len(self.memory)


# Set the device for tensors
device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "mps:0" if torch.backends.mps.is_available() else "cpu")