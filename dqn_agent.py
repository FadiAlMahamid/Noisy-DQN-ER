import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from q_network import QNetwork
from replay_buffer import ReplayBuffer

# ===================================================================
# --- The Noisy DQN Agent Class ---
# ===================================================================
class DQNAgent:
    """
    The Noisy DQN agent with Experience Replay and a Target Network.
    Uses NoisyLinear layers for exploration instead of epsilon-greedy,
    and learns from batches sampled from a Replay Buffer to stabilize training.

    Noisy DQN replaces the epsilon-greedy policy with parametric noise
    injected into the network weights (Fortunato et al., 2018). The noise
    magnitude is learned via gradient descent, allowing the agent to
    automatically adjust its exploration level — exploring more in
    uncertain states and less in well-learned ones.

    Combined with Double DQN target calculation to reduce overestimation:
        target = rₜ + γ · Q_target(sₜ₊₁, argmaxₐ' Q_policy(sₜ₊₁, a'))
    """
    def __init__(self, state_shape, action_size, learning_rate, gamma,
                 device, buffer_capacity, batch_size, learning_starts,
                 target_update_freq, loss_function="huber", optimizer="adam",
                 grad_clip_norm=1.0, clip_rewards=True, sigma_init=0.5):
        """
        Initializes the Noisy DQN Agent.

        Note: No epsilon parameters are needed — exploration is handled
        entirely by the NoisyLinear layers in the Q-network.
        """
        self.state_shape = state_shape
        self.action_size = action_size
        self.gamma = gamma
        self.device = device
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.target_update_freq = target_update_freq
        self.grad_clip_norm = grad_clip_norm
        self.clip_rewards = clip_rewards

        # Total steps taken across all training episodes
        self.total_steps = 0

        # Counter for tracking when to update the target network
        self.learn_step_counter = 0

        # Initialize the Policy Network (the noisy network we train).
        # NoisyLinear layers in the network handle exploration automatically.
        self.policy_network = QNetwork(state_shape, action_size, sigma_init=sigma_init).to(self.device)

        # Initialize the Target Network (a frozen copy used for stable TD targets).
        # The target network is NOT trained directly — its weights are periodically
        # copied from the policy network to provide stable Q-value targets.
        self.target_network = QNetwork(state_shape, action_size, sigma_init=sigma_init).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

        # Initialize the Experience Replay Buffer.
        # Stores individual frames and reconstructs stacked states on demand
        # to avoid storing redundant overlapping frames.
        frame_shape = state_shape[1:]  # (H, W) from (stack_size, H, W)
        stack_size = state_shape[0]
        self.replay_buffer = ReplayBuffer(buffer_capacity, frame_shape=frame_shape, stack_size=stack_size)

        # Set up the optimizer to update the policy network weights
        if optimizer == "rmsprop":
            self.optimizer = optim.RMSprop(self.policy_network.parameters(), lr=learning_rate)
        else:
            self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

        # Set up the loss function.
        if loss_function == "huber":
            self.loss_fn = nn.SmoothL1Loss()
        else:
            self.loss_fn = nn.MSELoss()

    def choose_action(self, state, training=True):
        """
        Chooses an action using the noisy network (no epsilon-greedy needed).

        The NoisyLinear layers inject learned noise into the forward pass,
        causing the network to output slightly different Q-values each time.
        This naturally drives exploration — the agent tries different actions
        because the noisy weights perturb its Q-value estimates. As training
        progresses, the learned sigma parameters shrink, and the agent
        automatically transitions from exploration to exploitation.

        Args:
            state (np.ndarray): The current state observation.
            training (bool): If True (training mode), increment step counter
                and use noisy forward pass. If False (deployment mode),
                use eval mode for deterministic (noise-free) actions.

        Returns:
            int: The selected action index.
        """
        # No random action selection — the noise in the network weights
        # handles exploration. The agent always picks the action with the
        # highest (noisy) Q-value, but noise makes this stochastic.
        # In deployment mode (training=False), eval mode disables noise
        # for deterministic action selection.
        if training:
            self.total_steps += 1
        else:
            self.policy_network.eval()

        # torch.no_grad() disables gradient tracking since we're only
        # doing inference here, not training — saves memory and compute.
        with torch.no_grad():
            # unsqueeze(0) adds a batch dimension: (4,84,84) -> (1,4,84,84)
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Forward pass returns Q-values for all actions, shape: (1, num_actions)
            q_values = self.policy_network(state_tensor)

            # argmax returns the action index with the highest Q-value
            action = q_values.argmax().item()

        if not training:
            self.policy_network.train()

        return action

    def store_transition(self, state, action, reward, next_state, done):
        """
        Stores a transition in the replay buffer.
        Applies reward clipping if enabled.

        Args:
            state (np.ndarray): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (np.ndarray): The next state.
            done (bool): Whether the episode terminated.
        """
        # Reward clipping normalizes the learning signal so that games with
        # large score differences (e.g., 1 vs 100 points) train at similar scales.
        # All positive rewards become +1, negatives become -1.
        if self.clip_rewards:
            reward = np.clip(reward, -1.0, 1.0)

        self.replay_buffer.push(state, action, reward, next_state, done)

    def learn(self):
        """
        Performs a gradient descent step using a mini-batch from the replay buffer.
        Uses Double DQN target calculation: the POLICY network selects the best
        action for the next state, but the TARGET network evaluates that action's
        Q-value. After each learning step, noise is resampled in both networks
        so that the next action selection uses fresh exploration noise.

            target = rₜ + γ · Q_target(sₜ₊₁, argmaxₐ' Q_policy(sₜ₊₁, a'))

        Also handles periodic target network updates and noise resampling.

        Returns:
            float: The loss value for this gradient step.
        """
        # 1. Sample a random mini-batch from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # 2. Convert numpy arrays to PyTorch tensors and move to GPU/device.
        #    unsqueeze(1) adds a column dimension so shapes align for element-wise
        #    operations: actions (32,) -> (32,1), rewards (32,) -> (32,1), etc.
        states_t = torch.tensor(states, dtype=torch.float32).to(self.device)        # (batch, 4, 84, 84)
        actions_t = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)  # (batch, 1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)  # (batch, 1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32).to(self.device)  # (batch, 4, 84, 84)
        dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)  # (batch, 1)

        # 3. Calculate Q(s, a) - The "Prediction"
        #    policy_network(states_t) returns noisy Q-values for ALL actions: (batch, num_actions)
        #    .gather(1, actions_t) selects only the Q-value for the action that was
        #    actually taken in each transition, resulting in shape (batch, 1).
        q_prediction = self.policy_network(states_t).gather(1, actions_t)

        # 4. Calculate the TD Target using Double DQN
        #
        #   Standard DQN:  target = rₜ + γ · maxₐ' Q_target(sₜ₊₁, a')
        #   Double DQN:    target = rₜ + γ · Q_target(sₜ₊₁, argmaxₐ' Q_policy(sₜ₊₁, a'))
        #
        # The POLICY network selects the best action (argmax),
        # but the TARGET network evaluates that action's Q-value.
        # This decoupling reduces overestimation bias because the policy
        # network's greedy action may not be the highest-valued action
        # according to the target network, avoiding the positive bias
        # that occurs when the same network both selects and evaluates.
        with torch.no_grad():
            # Step A: Policy network selects the best action for each next state.
            # .argmax(1) returns the action index with the highest Q-value.
            # keepdim=True preserves shape (batch, 1) for use with .gather().
            best_actions = self.policy_network(next_states_t).argmax(1, keepdim=True)

            # Step B: Target network evaluates the Q-value of those selected actions.
            # .gather(1, best_actions) picks the Q-value at the action index chosen above.
            q_next = self.target_network(next_states_t).gather(1, best_actions)

            # Bellman Equation: rₜ + γ · Q_target(sₜ₊₁, best_action)
            # (1 - dones_t) zeroes out future reward for terminal states,
            # since there is no "next state" after the episode ends.
            q_target = rewards_t + (1 - dones_t) * self.gamma * q_next

        # 5. Calculate the Loss
        loss = self.loss_fn(q_prediction, q_target)

        # 6. Perform Gradient Descent
        self.optimizer.zero_grad()  # Clear gradients from the previous step
        loss.backward()             # Backpropagate the loss to compute gradients
        # Clip gradients to prevent exploding gradients from large TD errors
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.grad_clip_norm)
        self.optimizer.step()       # Update weights using the computed gradients

        # 7. Resample noise in the policy network after each learning step.
        # Fresh noise ensures the agent explores differently on the next action
        # selection.
        self.policy_network.reset_noise()

        # 8. Periodically update the target network
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()

    def update_target_network(self):
        """
        Hard update: copies policy network weights to the target network.
        Called periodically (every target_update_freq learning steps) to keep
        the target network's Q-values stable between updates.
        """
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def save_model(self, filepath):
        """Saves a full training checkpoint to file."""
        torch.save({
            'policy_network': self.policy_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'learn_step_counter': self.learn_step_counter,
        }, filepath)

    def load_model(self, filepath):
        """Loads a training checkpoint from file into agent components."""
        # map_location ensures tensors are loaded onto the correct device (CPU/GPU/MPS).
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        # Support both new checkpoint format (dict with keys) and legacy format
        # (raw state_dict) for backward compatibility with older saved models.
        if isinstance(checkpoint, dict) and 'policy_network' in checkpoint:
            self.policy_network.load_state_dict(checkpoint['policy_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.learn_step_counter = checkpoint['learn_step_counter']
        else:
            # Legacy format: checkpoint is just the policy network state_dict
            self.policy_network.load_state_dict(checkpoint)
            self.target_network.load_state_dict(self.policy_network.state_dict())

        self.target_network.eval()
        self.policy_network.train()
