import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from imdb_dataset import IMDBDataset
from single_dqn_agent import SingleDQNAgent
from single_dqn_predict import predict_sentiment
from single_dqn_train import train
from single_dqn_evaluate import evaluate

# Load the IMDb dataset
# file_path = '../imbalanced_datasets/IMDB_Dataset_Imbalance_0.01.csv'
file_path = '../IMDB-Dataset-Edited.csv'
small_path = '../last_10_reviews.csv'

# Load the datasets
df = pd.read_csv(file_path)
small_df = pd.read_csv(small_path)

# Encode sentiments as integers
le = LabelEncoder()
df['sentiment'] = le.fit_transform(df['sentiment'])
small_df['sentiment'] = le.fit_transform(small_df['sentiment'])

# Calculate the imbalance ratio
counts = df['sentiment'].value_counts()
imbalance_ratio = counts.max() / counts.min()
print(f"Imbalance Ratio: {imbalance_ratio:.2f}")

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Create datasets
train_dataset = IMDBDataset(train_df, max_tokens=5000, max_len=500)
test_dataset = IMDBDataset(test_df, vocab=train_dataset.vocab, max_len=500)
small_dataset = IMDBDataset(small_df, max_tokens=5000, max_len=500, vocab=train_dataset.vocab)


def collate_fn(batch, vocab, max_len=500):
    """
    Collate function to pad sequences to the same length.

    Args:
        batch (list): List of samples.
        vocab (dict): Vocabulary mapping tokens to indices.
        max_len (int): Maximum length of the tokenized sequence.

    Returns:
        tuple: Padded sequences and labels.
    """
    texts, labels = zip(*batch)
    texts_padded = pad_sequence(
        [torch.cat([text, torch.tensor([vocab['<pad>']] * (max_len - len(text)))]) for text in texts],
        batch_first=True, padding_value=vocab['<pad>']
    ).long()
    labels = torch.tensor(labels, dtype=torch.long)
    return texts_padded, labels


# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                          collate_fn=lambda x: collate_fn(x, train_dataset.vocab))
test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=lambda x: collate_fn(x, test_dataset.vocab))
small_loader = DataLoader(small_dataset, batch_size=1, collate_fn=lambda x: collate_fn(x, train_dataset.vocab))

if __name__ == "__main__":
    # Training the agent
    vocab_size = len(train_dataset.vocab)  # Vocabulary size
    num_classes = 2  # Number of output classes
    agent = SingleDQNAgent(vocab_size, max_words=500, num_classes=num_classes)

    agent = train(agent, train_loader)  # Train agent

    evaluate(agent, test_loader)  # Evaluate the agent

    # Predict sentiment of the last 10 reviews
    print(f"{'Review':<80} {'True':<10} {'Predicted':<10}")
    print("=" * 100)

    for text, label in small_loader:
        review = " ".join([train_dataset.reverse_vocab[token.item()] for token in text[0]
                           if token.item() != train_dataset.vocab['<pad>']])
        true_sentiment = label.item()
        predicted_sentiment = predict_sentiment(review, agent, train_dataset.vocab, max_len=500)
        print(f"{review[:77]:<80} {true_sentiment:<10} {predicted_sentiment:<10}")