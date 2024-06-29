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
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["text", "action", "reward", "next_text", "done"])
        self.seed = random.seed(seed)

    def add(self, text, action, reward, next_text, done):
        """
         Add a new experience to memory.

         Args:
             text (torch.Tensor): Tensor containing token ids.
             action (int): Action taken.
             reward (float): Reward received.
             next_text (torch.Tensor): Next state.
             done (bool): Whether the episode is done.
         """
        e = self.experience(text.to(device), action, reward, next_text.to(device), done)
        self.memory.append(e)

    def sample(self):
        """
        Sample a batch of experiences from memory.

        Returns:
            tuple: A tuple containing batches of texts, actions, rewards, next_texts, and dones.
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        texts = [e.text for e in experiences if e is not None]
        actions = [e.action for e in experiences if e is not None]
        rewards = [e.reward for e in experiences if e is not None]
        next_texts = [e.next_text for e in experiences if e is not None]
        dones = [e.done for e in experiences if e is not None]

        max_len = max(max([len(text) for text in texts]), max([len(text) for text in next_texts]))

        texts_padded = pad_sequence(
            [torch.cat([text, torch.tensor([1] * (max_len - len(text))).to(device)]).unsqueeze(0) for text in texts],
            batch_first=True, padding_value=1
        ).squeeze(1)

        next_texts_padded = pad_sequence(
            [torch.cat([text, torch.tensor([1] * (max_len - len(text))).to(device)]).unsqueeze(0) for text in
             next_texts],
            batch_first=True, padding_value=1
        ).squeeze(1)

        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        dones = torch.tensor(dones, dtype=torch.float).to(device)

        return texts_padded, actions, rewards, next_texts_padded, dones

    def __len__(self):
        return len(self.memory)


device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "mps:0" if torch.backends.mps.is_available() else "cpu")
