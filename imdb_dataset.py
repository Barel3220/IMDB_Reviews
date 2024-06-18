import torch
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from collections import Counter

# Initialize tokenizer for tokenizing text data
tokenizer = get_tokenizer('basic_english')


class IMDBDataset(Dataset):
    def __init__(self, dataframe, vocab=None, max_tokens=5000, max_len=500):
        """
        Initialize the IMDBDataset.

        Args:
            dataframe (pd.DataFrame): DataFrame containing the dataset.
            vocab (dict): Vocabulary mapping tokens to indices. If None, vocab will be built from the dataset.
            max_tokens (int): Maximum number of tokens in the vocabulary.
            max_len (int): Maximum length of tokenized sequences.
        """
        self.reverse_vocab = None
        self.dataframe = dataframe
        self.reviews = dataframe['review'].tolist()
        self.sentiments = dataframe['sentiment'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Build vocabulary if not provided
        if vocab is None:
            self.vocab = self.build_vocab(max_tokens)
        else:
            self.vocab = vocab

    def build_vocab(self, max_tokens):
        """
        Build a vocabulary from the dataset.

        Args:
            max_tokens (int): Maximum number of tokens in the vocabulary.

        Returns:
            dict: Vocabulary mapping tokens to indices.
        """
        counter = Counter()
        for text in self.reviews:
            counter.update(self.tokenizer(text))
        vocab = {word: idx for idx, (word, _) in enumerate(counter.most_common(max_tokens), start=2)}
        vocab['<unk>'] = 0
        vocab['<pad>'] = 1
        # Create reverse mapping for index-to-string
        self.reverse_vocab = {idx: word for word, idx in vocab.items()}
        return vocab

    def process_data(self, text):
        """
        Tokenize and convert a text to its corresponding token ids.

        Args:
            text (str): The input text.

        Returns:
            torch.Tensor: Tensor containing token ids.
        """
        tokens = self.tokenizer(text)
        token_ids = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
        return torch.tensor(token_ids, dtype=torch.long)

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing token ids and the corresponding sentiment label.
        """
        text, label = self.reviews[idx], self.sentiments[idx]
        token_ids = self.process_data(text)
        token_ids = token_ids[:self.max_len]  # Trim to max length
        return token_ids, label
