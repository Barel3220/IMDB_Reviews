import torch.nn as nn
import torch.nn.functional as F


class TextModel(nn.Module):
    def __init__(self, vocab_size, max_words, output_size):
        super(TextModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 128)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * max_words, 250)
        self.fc2 = nn.Linear(250, output_size)

    def forward(self, x):
        x = self.embedding(x.long())  # Ensure input is of type long
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_text_model(input_shape, output_size):
    top_words, max_words = input_shape
    return TextModel(top_words, max_words, output_size)
