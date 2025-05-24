"""
Critic Network (DoubleCritic) for the Actor-Critic Algorithm.

This module defines the DoubleCritic network architecture, which takes the
environment observation and a structured action (from the actor) as input,
and outputs two Q-value estimates.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import config # Import shared configurations

class DoubleCritic(nn.Module):
    """
    Double Critic network that evaluates state-action pairs.

    It takes the state (environment observation) and a structured action dictionary
    as input. The action dictionary is processed into a flat tensor before being
    combined with the processed state to estimate Q-values. Two separate Q-networks
    (q1_net, q2_net) are used to mitigate overestimation bias.
    """
    def __init__(self,
                 state_dim: int = config.FLATTENED_OBS_DIM,
                 # action_dim is implicitly defined by config.PROCESSED_ACTION_DIM_FOR_CRITIC
                 hidden_dim: int = 256): # Example hidden_dim, can be configured
        super().__init__()

        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), nn.Mish()
        )
        self.action_mlp = nn.Sequential(
            nn.Linear(config.PROCESSED_ACTION_DIM_FOR_CRITIC, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), nn.Mish()
        )
        
        # Q1 network
        self.q1_net = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim * 2), nn.Mish(),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, 1)
        )
        # Q2 network
        self.q2_net = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim * 2), nn.Mish(),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, 1)
        )

    def _process_structured_action(self, action_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Processes a structured action dictionary into a single flat tensor.

        Applies softmax to logits for discrete action components and concatenates
        them with continuous action components (e.g., priority weights).

        Args:
            action_dict: A dictionary containing action components:
                         'dim_bucket_logits', 'batch_size_logits', 'priority_weights'.

        Returns:
            A flat tensor representing the processed action.
        """
        dim_bucket_logits = action_dict["dim_bucket_logits"]
        batch_size_logits = action_dict["batch_size_logits"]
        priority_weights = action_dict["priority_weights"]

        dim_bucket_probs = F.softmax(dim_bucket_logits.float(), dim=-1)
        batch_size_probs = F.softmax(batch_size_logits.float(), dim=-1)
        
        processed_action_tensor = torch.cat([dim_bucket_probs, batch_size_probs, priority_weights], dim=-1)
        
        if processed_action_tensor.shape[-1] != config.PROCESSED_ACTION_DIM_FOR_CRITIC:
            raise ValueError(
                f"Processed action tensor has wrong dimension. "
                f"Expected {config.PROCESSED_ACTION_DIM_FOR_CRITIC}, got {processed_action_tensor.shape[-1]}"
            )
        return processed_action_tensor

    def forward(self, state: torch.Tensor, action_dict: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the critic.

        Args:
            state: The environment observation (flattened).
            action_dict: The structured action dictionary from the policy.

        Returns:
            A tuple containing the two Q-value estimates (q1, q2).
        """
        processed_state = self.state_mlp(state)
        processed_action_flat = self._process_structured_action(action_dict)
        processed_action_embedding = self.action_mlp(processed_action_flat)
        
        # Handle cases where state or action might be single sample vs batch
        if processed_state.ndim == 1 and processed_action_embedding.ndim > 1 and processed_action_embedding.size(0) > 0 :
            processed_state = processed_state.unsqueeze(0).expand(processed_action_embedding.size(0), -1)
        elif processed_action_embedding.ndim == 1 and processed_state.ndim > 1 and processed_state.size(0) > 0:
            processed_action_embedding = processed_action_embedding.unsqueeze(0).expand(processed_state.size(0), -1)

        x = torch.cat([processed_state, processed_action_embedding], dim=-1)
        return self.q1_net(x), self.q2_net(x)

    def q_min(self, state: torch.Tensor, action_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Returns the minimum of the two Q-values, used for target Q calculation."""
        q1, q2 = self.forward(state, action_dict)
        return torch.min(q1, q2)

if __name__ == '__main__':
    BATCH_S = 16
    critic_network = DoubleCritic(state_dim=config.FLATTENED_OBS_DIM, hidden_dim=128)

    dummy_state_obs = torch.randn(BATCH_S, config.FLATTENED_OBS_DIM)
    dummy_action_dict = {
        "dim_bucket_logits": torch.randn(BATCH_S, config.K_DIM_BUCKETS),
        "batch_size_logits": torch.randn(BATCH_S, config.M_BATCH_SIZE_OPTIONS),
        "priority_weights": torch.tanh(torch.randn(BATCH_S, config.N_PRIORITY_WEIGHTS)) * config.MAX_ACTION_CONTINUOUS_PART
    }

    q1_vals, q2_vals = critic_network(dummy_state_obs, dummy_action_dict)
    print(f"Critic Q1 values shape: {q1_vals.shape}")
    print(f"Critic Q2 values shape: {q2_vals.shape}")
    assert q1_vals.shape == (BATCH_S, 1)
    assert q2_vals.shape == (BATCH_S, 1)

    min_q_vals = critic_network.q_min(dummy_state_obs, dummy_action_dict)
    print(f"Critic Min Q values shape: {min_q_vals.shape}")
    assert min_q_vals.shape == (BATCH_S, 1)
    
    # Test with single instance
    dummy_state_single = torch.randn(config.FLATTENED_OBS_DIM)
    dummy_action_dict_single = {
        "dim_bucket_logits": torch.randn(config.K_DIM_BUCKETS),
        "batch_size_logits": torch.randn(config.M_BATCH_SIZE_OPTIONS),
        "priority_weights": torch.tanh(torch.randn(config.N_PRIORITY_WEIGHTS)) * config.MAX_ACTION_CONTINUOUS_PART
    }
    q1_single, _ = critic_network(dummy_state_single, dummy_action_dict_single)
    print(f"Critic Q1 single instance shape: {q1_single.shape}")
    assert q1_single.shape == (1,1) or q1_single.shape == (1,)

    print("\nDoubleCritic basic tests with config dimensions seem OK.")
