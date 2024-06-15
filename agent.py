import torch
import torch.nn.functional as F
import torch.optim as optim
from qnetwork import QNetwork
from replay_buffer import ReplayBuffer, device


def soft_update(local_model, target_model, tau):
    """
    Used to slowly update the target network parameters to be closer to the local network parameters using the
    parameter tau

    Args:
        local_model (nn.Module): Local Q-Network.
        target_model (nn.Module): Target Q-Network.
        tau (float): Interpolation parameter.
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * local_param.data)


class Agent:
    def __init__(self, vocab_size, embed_dim, num_classes, buffer_size=int(1e5), batch_size=64, gamma=0.99, tau=1e-3,
                 lr=5e-4, update_every=4):
        """
        Initialize the Agent.

        Args:
            vocab_size (int): Size of the vocabulary.
            embed_dim (int): Dimension of the embedding vectors.
            num_classes (int): Number of output classes (positive/negative sentiment).
            buffer_size (int): Maximum size of the replay buffer.
            batch_size (int): Size of each training batch.
            gamma (float): Discount factor for future rewards.
            tau (float): Parameter for soft update of target network.
            lr (float): Learning rate for the optimizer.
            update_every (int): Frequency of updating the network.
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.update_every = update_every

        # Q-Network
        self.qnetwork_local = QNetwork(vocab_size, embed_dim, num_classes).to(device)  # Local Q-Network
        self.qnetwork_target = QNetwork(vocab_size, embed_dim, num_classes).to(device)  # Target Q-Network
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)  # Optimizer

        # Replay memory
        self.memory = ReplayBuffer(buffer_size, batch_size, seed=0)  # Replay buffer
        self.t_step = 0  # Initialize time step for updating network

    def step(self, texts, labels):
        """
        Add experience to memory and possibly learn from it.

        Args:
            texts (torch.Tensor): Batch of token ids.
            labels (torch.Tensor): Batch of sentiment labels.
        """
        texts, labels = texts.to(device), labels.to(device)  # Ensure texts and labels are on the correct device
        for text, label in zip(texts, labels):
            self.memory.add(text, label.item())  # Add experience to memory

        loss = None
        self.t_step = (self.t_step + 1) % self.update_every  # Increment time step
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()  # Sample experiences
                loss = self.learn(experiences)  # Learn from experiences
        return loss

    def act(self, text):
        """
        Get action values from the local Q-Network.

        Args:
            text (torch.Tensor): Tensor containing token ids.

        Returns:
            torch.Tensor: Tensor containing action values.
        """
        text = text.to(device)  # Ensure text is on the correct device
        self.qnetwork_local.eval()  # Set network to evaluation mode
        with torch.no_grad():
            action_values = self.qnetwork_local(text)  # Get action values
        self.qnetwork_local.train()  # Set network back to training mode
        return torch.argmax(action_values, dim=1)  # Return the action with the highest value

    def learn(self, experiences):
        """
        Update value parameters using given batch of experience tuples.

        Args:
            experiences (tuple): Batch of experiences.
        """
        texts, labels = experiences  # Unpack experiences

        # Double DQN: Get the best action using the local model
        local_actions = self.qnetwork_local(texts).detach()
        local_actions = torch.argmax(local_actions, dim=1, keepdim=True)

        # Get the Q value from the target network for the next action
        target_q_values = self.qnetwork_target(texts).gather(1, local_actions).detach()

        # Compute Q targets for current states
        Q_targets = labels.float().view(-1, 1)

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(texts).gather(1, local_actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

        return loss
