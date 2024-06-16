import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from imdb_dataset import IMDBDataset
from agent import Agent
from predict import predict_sentiments
from train import train
from evaluate import evaluate

# Load the IMDb dataset
file_path = 'IMDB-Dataset-Edited.csv'
df = pd.read_csv(file_path)

# Encode the sentiments (positive/negative) as integers
le = LabelEncoder()
df['sentiment'] = le.fit_transform(df['sentiment'])

# Calculate the imbalance ratio
counts = df['sentiment'].value_counts()
imbalance_ratio = counts.max() / counts.min()
print(f"Imbalance Ratio: {imbalance_ratio}")

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Create datasets
train_dataset = IMDBDataset(train_df, max_tokens=5000, max_len=500)
test_dataset = IMDBDataset(test_df, vocab=train_dataset.vocab, max_len=500)


def collate_fn(batch, vocab):
    """
    Collate function to pad sequences to the same length.

    Args:
        batch (list): List of samples.
        vocab (dict): Vocabulary mapping tokens to indices.

    Returns:
        tuple: Padded sequences and labels.
    """
    texts, labels = zip(*batch)
    lengths = [len(text) for text in texts]
    max_len = max(lengths)
    texts_padded = pad_sequence(
        [torch.cat([text, torch.tensor([vocab['<pad>']] * (max_len - len(text)))]) for text in texts],
        batch_first=True, padding_value=vocab['<pad>']
    )
    labels = torch.tensor(labels, dtype=torch.long)
    return texts_padded, labels


# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                          collate_fn=lambda x: collate_fn(x, train_dataset.vocab))
test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=lambda x: collate_fn(x, test_dataset.vocab))

if __name__ == "__main__":
    # Training the agent
    vocab_size = len(train_dataset.vocab)  # Vocabulary size
    embed_dim = 100  # Embedding dimension
    num_classes = 2  # Number of output classes
    agent = Agent(vocab_size, embed_dim, num_classes)  # Initialize agent

    train(agent, train_loader)  # Train agent

    evaluate(agent, test_loader)    # Evaluate the agent

    # Predict sentiment of the last 10 reviews
    predict_sentiments('last_10_reviews.csv', agent, train_dataset.vocab)
