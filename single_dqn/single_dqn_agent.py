import torch
import torch.nn.functional as F
import torch.optim as optim
from text_model import TextModel
from single_dqn_replay_buffer import ReplayBuffer, device


class SingleDQNAgent:
    def __init__(self, vocab_size, max_words, num_classes, buffer_size=int(5e4), batch_size=64,
                 lr=5e-4, update_every=1, gamma=0.1):
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

    def step(self, texts, labels):
        """
        Add experience to memory and possibly learn from it.

        Args:
            texts (torch.Tensor): Batch of token ids.
            labels (torch.Tensor): Batch of sentiment labels.
        """
        texts, labels = texts.to(device), labels.to(device)
        rewards = []

        for text, label in zip(texts, labels):
            action = self.act(text.unsqueeze(0)).item()
            reward = self.calculate_reward(action, label)
            done = True  # Each step is a terminal step in this setup
            self.memory.add(text, action, reward, text, done)
            rewards.append(reward)  # Collect rewards for logging or debugging if needed

        loss = None
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                loss = self.learn(experiences)
        return loss

    def calculate_reward(self, action, label):
        minority_class = 1  # Assuming minority class is labeled as 1
        majority_class = 0  # Assuming majority class is labeled as 0
        if label == minority_class:
            return 1 if action == label else -1
        else:
            return self.gamma if action == label else -self.gamma

    def act(self, text):
        """
        Get action values from the Q-Network using ε-greedy policy.

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
