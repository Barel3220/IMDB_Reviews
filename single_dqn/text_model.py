import torch.nn as nn
import torch.nn.functional as F


class TextModel(nn.Module):
    def __init__(self, vocab_size, max_words, output_size):
        """
        Initialize the TextModel.

        Args:
            vocab_size (int): Size of the vocabulary.
            max_words (int): Maximum number of words in a sequence.
            output_size (int): Number of output classes (positive/negative sentiment).
        """
        super(TextModel, self).__init__()
        self.embedding_dim = 128
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.embedding_dim * max_words, 250)
        self.fc2 = nn.Linear(250, output_size)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor containing token ids.

        Returns:
            torch.Tensor: Output tensor containing class scores.
        """
        x = x.long()
        x = self.embedding(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
