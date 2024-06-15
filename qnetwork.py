import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        """
        Initialize the QNetwork.

        Args:
            vocab_size (int): Size of the vocabulary.
            embed_dim (int): Dimension of the embedding vectors.
            num_classes (int): Number of output classes (positive/negative sentiment).
        """
        super(QNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=1)  # Embedding layer
        self.fc = nn.Linear(embed_dim, num_classes)  # Fully connected layer

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor containing token ids.

        Returns:
            torch.Tensor: Output tensor containing class scores.
        """
        x = x.long()  # Ensure input is of type long
        x = self.embedding(x)  # Pass through embedding layer
        x = x.mean(dim=1)  # Average pooling along the sequence length dimension
        x = self.fc(x)  # Pass through fully connected layer
        return x
