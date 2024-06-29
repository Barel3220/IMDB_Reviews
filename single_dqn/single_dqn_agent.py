import torch
import torch.nn.functional as F
import torch.optim as optim
from text_model import TextModel
from single_dqn_replay_buffer import ReplayBuffer, device


class SingleDQNAgent:
    def __init__(self, vocab_size, max_words, num_classes, buffer_size=int(1e5), batch_size=128,
                 lr=5e-4, update_every=1, gamma=0.5):
        """
        Initialize the SingleDQNAgent.

        Args:
            vocab_size (int): Size of the vocabulary.
            max_words (int): Maximum number of words in a sequence.
            num_classes (int): Number of output classes (positive/negative sentiment).
            buffer_size (int): Maximum size of the replay buffer.
            batch_size (int): Size of each training batch.
            lr (float): Learning rate for the optimizer.
            update_every (int): Frequency of updating the network.
            gamma (float): Discount factor for future rewards.
        """
        self.vocab_size = vocab_size
        self.max_words = max_words
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.update_every = update_every
        self.gamma = gamma

        # Initialize Q-Network
        self.qnetwork = TextModel(vocab_size, max_words, num_classes).to(device)
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=lr)

        # Initialize replay memory
        self.memory = ReplayBuffer(buffer_size, batch_size, seed=0)
        self.t_step = 0  # Initialize time step for updating network

    def step(self, texts, actions, rewards, next_texts, dones):
        """
        Add experience to memory and possibly learn from it.

        Args:
            texts (torch.Tensor): Batch of token ids.
            actions (torch.Tensor): Batch of actions taken.
            rewards (torch.Tensor): Batch of rewards received.
            next_texts (torch.Tensor): Batch of next states.
            dones (torch.Tensor): Batch of done signals.
        """
        texts, actions, rewards, next_texts, dones = texts.to(device), actions.to(device), rewards.to(
            device), next_texts.to(device), dones.to(device)
        for i in range(len(texts)):
            self.memory.add(texts[i], actions[i].item(), rewards[i].item(), next_texts[i], dones[i].item())

        loss = None
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                loss = self.learn(experiences)
        return loss

    def act(self, text):
        """
        Get action values from the Q-Network.

        Args:
            text (torch.Tensor): Tensor containing token ids.

        Returns:
            torch.Tensor: Tensor containing action values.
        """
        text = text.to(device)
        self.qnetwork.eval()
        with torch.no_grad():
            action_values = self.qnetwork(text)
        self.qnetwork.train()
        return torch.argmax(action_values, dim=1)

    def learn(self, experiences):
        """
        Update value parameters using given batch of experience tuples.

        Args:
            experiences (tuple): Batch of experiences.
        """
        texts, actions, rewards, next_texts, dones = experiences

        # Get the Q value from the Q-Network
        q_values = self.qnetwork(texts).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_values_next = self.qnetwork(next_texts).max(1)[0]
            q_targets = rewards + (self.gamma * q_values_next * (1 - dones.float()))

        loss = F.mse_loss(q_values, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
