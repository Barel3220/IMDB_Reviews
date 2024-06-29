import torch
import torch.nn.functional as F
import torch.optim as optim
from text_model import TextModel
from replay_buffer import ReplayBuffer, device


def soft_update(local_model, target_model, tau):
    """
    Used to slowly update the target network parameters to be closer to the local network parameters using the
    parameter tau.
    θ_target = τ*θ_local + (1 - τ)*θ_target

    Args:
        local_model (nn.Module): Local Q-Network.
        target_model (nn.Module): Target Q-Network.
        tau (float): Interpolation parameter.
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * local_param.data)


class DoubleDQNAgent:
    def __init__(self, vocab_size, max_words, num_classes, buffer_size=int(1e5), batch_size=128, lr=5e-4,
                 update_every=1, gamma=0.5, tau=1e-2):
        """
        Initialize the DoubleDQNAgent.

        Args:
            vocab_size (int): Size of the vocabulary.
            max_words (int): Maximum number of words in a sequence.
            num_classes (int): Number of output classes (positive/negative sentiment).
            buffer_size (int): Maximum size of the replay buffer.
            batch_size (int): Size of each training batch.
            lr (float): Learning rate for the optimizer.
            update_every (int): Frequency of updating the network.
            gamma (float): Discount factor for future rewards.
            tau (float): Soft update parameter for target network.
        """
        self.vocab_size = vocab_size
        self.max_words = max_words
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.update_every = update_every
        self.tau = tau
        self.gamma = gamma

        # Initialize Q-Network and target Q-Network
        self.qnetwork_local = TextModel(vocab_size, max_words, num_classes).to(device)
        self.qnetwork_target = TextModel(vocab_size, max_words, num_classes).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

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
        Get action values from the local Q-Network.

        Args:
            text (torch.Tensor): Tensor containing token ids.

        Returns:
            torch.Tensor: Tensor containing action values.
        """
        text = text.to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(text)
        self.qnetwork_local.train()
        return torch.argmax(action_values, dim=1)

    def learn(self, experiences):
        """
        Update value parameters using given batch of experience tuples.

        Args:
            experiences (tuple): Batch of experiences.
        """
        texts, actions, rewards, next_texts, dones = experiences

        q_values_local = self.qnetwork_local(texts).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_values_next_local = self.qnetwork_local(next_texts)
            q_values_next_target = self.qnetwork_target(next_texts)
            max_actions = q_values_next_local.argmax(dim=1)
            q_values_next = q_values_next_target.gather(1, max_actions.unsqueeze(1)).squeeze(1)

        q_targets = rewards + (self.gamma * q_values_next * (1 - dones.float()))

        loss = F.mse_loss(q_values_local, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

        return loss
