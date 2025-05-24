"""
Diffusion-based Optimization Policy (DiffusionOPT).

This module defines the DiffusionOPT policy, which uses a diffusion model as
an actor and a double critic for Q-value estimation. It handles the interaction
between the actor, critic, and environment, including action parsing,
observation preprocessing, and learning updates.
"""
import torch
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
from typing import Any, Dict, Optional, Union, Tuple
from tianshou.data import Batch, ReplayBuffer, to_torch # Assuming Tianshou components
from tianshou.policy import BasePolicy # Assuming Tianshou BasePolicy
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataclasses import dataclass
from tianshou.policy.base import TrainingStats
import config

@dataclass(kw_only=True)
class DiffusionOPTStats(TrainingStats): # Custom TrainingStats for Tianshou
    """Statistics for DiffusionOPT training."""
    critic_loss: float = 0.0
    actor_loss: float = 0.0
    actor_lr: float = 0.0
    critic_lr: float = 0.0

class GaussianNoise: # Helper class for exploration noise
    """Generates Gaussian noise for action exploration."""
    def __init__(self, mu: float = 0.0, sigma: float = 0.1):
        self.mu = mu
        self.sigma = sigma
    def __call__(self, shape: tuple) -> np.ndarray:
        return np.random.normal(self.mu, self.sigma, shape).astype(np.float32)

class DiffusionOPT(BasePolicy):
    """
    Diffusion-based Optimization Policy.

    Implements an actor-critic style policy where the actor is a diffusion model
    that generates parameters for a structured action.
    """
    def __init__(
            self,
            actor: torch.nn.Module, # Diffusion model (e.g., Diffusion class wrapping MLPUNet)
            actor_optim: torch.optim.Optimizer,
            critic: torch.nn.Module, # DoubleCritic network
            critic_optim: torch.optim.Optimizer,
            device: Union[str, int, torch.device] = "cpu",
            tau: float = config.TAU,
            gamma: float = config.GAMMA,
            exploration_noise_std: float = config.EXPLORATION_NOISE_STD,
            reward_normalization: bool = False,
            n_step: int = 1, # For n-step returns
            # Learning rate decay (optional, from original code)
            lr_decay: bool = False,
            lr_maxt: int = 1000, # T_max for CosineAnnealingLR
            # bc_coef_in_actor: Matches Diffusion model's bc_coef flag for loss calculation
            bc_coef_in_actor_diffusion_model: bool = False,
            action_space = None, # Tianshou BasePolicy might require this
            **kwargs: Any
    ) -> None:
        super().__init__(action_space=action_space, **kwargs) # Pass action_space if needed by BasePolicy
        
        self._actor: torch.nn.Module = actor.to(device)
        self._target_actor = deepcopy(self._actor).eval()
        self._actor_optim: torch.optim.Optimizer = actor_optim

        self._critic: torch.nn.Module = critic.to(device)
        self._target_critic = deepcopy(self._critic).eval()
        self._critic_optim: torch.optim.Optimizer = critic_optim
        
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        self._tau = tau
        self._gamma = gamma
        self._noise = GaussianNoise(sigma=exploration_noise_std) if exploration_noise_std > 0 else None
        self._rew_norm = reward_normalization
        self._n_step = n_step
        self.bc_coef_in_actor_diffusion_model = bc_coef_in_actor_diffusion_model

        self._actor_lr_scheduler: Optional[CosineAnnealingLR] = None
        self._critic_lr_scheduler: Optional[CosineAnnealingLR] = None
        if lr_decay:
            self._actor_lr_scheduler = CosineAnnealingLR(self._actor_optim, T_max=lr_maxt, eta_min=0.)
            self._critic_lr_scheduler = CosineAnnealingLR(self._critic_optim, T_max=lr_maxt, eta_min=0.)
        
        # Heuristic to check if actor is a diffusion model (has .sample and .loss methods)
        self._actor_is_diffusion_model = hasattr(self._actor, 'sample') and hasattr(self._actor, 'loss')


    def _parse_flat_action_to_dict(self, flat_action_params: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Converts the actor's flat output vector into a structured action dictionary."""
        if flat_action_params.shape[-1] != config.STRUCTURED_ACTION_FLAT_DIM:
            raise ValueError(
                f"Actor output dim {flat_action_params.shape[-1]} != expected {config.STRUCTURED_ACTION_FLAT_DIM}"
            )
        k_logits = flat_action_params[..., :config.K_DIM_BUCKETS]
        m_logits = flat_action_params[..., config.K_DIM_BUCKETS : config.K_DIM_BUCKETS + config.M_BATCH_SIZE_OPTIONS]
        priority_w = flat_action_params[..., config.K_DIM_BUCKETS + config.M_BATCH_SIZE_OPTIONS:]
        return {
            "dim_bucket_logits": k_logits,
            "batch_size_logits": m_logits,
            "priority_weights": priority_w
        }

    def _preprocess_obs(self, obs: Union[np.ndarray, torch.Tensor, Dict, Batch]) -> torch.Tensor:
        """Preprocesses observations to the flat tensor format expected by networks."""
        if isinstance(obs, Batch) and hasattr(obs, 'obs'): # Tianshou Batch
             # Recursively call with the actual observation content
            return self._preprocess_obs(obs.obs)
        elif isinstance(obs, dict): # From custom Env
            server_state = to_torch(obs['server_state'], device=self._device, dtype=torch.float32)
            request_queue = to_torch(obs['request_queue'], device=self._device, dtype=torch.float32)
            
            # Ensure batch dimension
            if server_state.ndim == 1: server_state = server_state.unsqueeze(0)
            if request_queue.ndim == 2: request_queue = request_queue.unsqueeze(0) # (N_req, F_req) -> (B, N_req, F_req)
            
            batch_size = server_state.shape[0]
            request_queue_flat = request_queue.reshape(batch_size, -1)
            flat_obs = torch.cat([server_state, request_queue_flat], dim=1)
            return flat_obs
        else: # Assuming obs is already a flat tensor or ndarray
            processed_obs = to_torch(obs, device=self._device, dtype=torch.float32)
            if processed_obs.ndim == 1: # Ensure batch dim for single observation
                processed_obs = processed_obs.unsqueeze(0)
            return processed_obs

    def forward(self, batch: Batch, state: Optional[Any] = None, model: str = "_actor", **kwargs: Any) -> Batch:
        """
        Computes actions for the given batch of observations.
        'model' arg specifies whether to use self._actor or self._target_actor.
        """
        current_obs_flat = self._preprocess_obs(batch) # batch.obs is handled by _preprocess_obs
        actor_nn = getattr(self, model) # self._actor or self._target_actor
        
        # Actor (Diffusion model) outputs a flat vector of action parameters
        flat_action_params = actor_nn(current_obs_flat) # Diffusion.sample(state_condition)
        
        if self.is_training and self._noise is not None and model == "_actor": # Add noise only from main actor during training
            noise_val = self._noise(flat_action_params.shape) # type: ignore
            noise_val_torch = to_torch(noise_val, device=flat_action_params.device, dtype=flat_action_params.dtype)
            flat_action_params = flat_action_params + noise_val_torch
            
            # Clamp the continuous part (priority_weights) after adding noise
            pw_start_idx = config.K_DIM_BUCKETS + config.M_BATCH_SIZE_OPTIONS
            if hasattr(actor_nn, 'max_action_clamp'): # If diffusion model has max_action_clamp
                 flat_action_params[..., pw_start_idx:] = \
                     flat_action_params[..., pw_start_idx:].clamp_(-actor_nn.max_action_clamp, actor_nn.max_action_clamp)
            else: # Fallback if max_action_clamp is not on actor (e.g. not diffusion)
                 flat_action_params[..., pw_start_idx:] = \
                     flat_action_params[..., pw_start_idx:].clamp_(-config.MAX_ACTION_CONTINUOUS_PART, config.MAX_ACTION_CONTINUOUS_PART)


        structured_action_dict = self._parse_flat_action_to_dict(flat_action_params)
        
        # Tianshou Batch expects 'act' and 'logits' (optional, but good for storage)
        # 'logits' here stores the raw flat parameters from the actor.
        # 'act' stores the structured dictionary to be used by the environment.
        return Batch(act=structured_action_dict, logits=flat_action_params, state=state)


    def _get_target_q_values(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        """Helper function to compute target Q-values for n-step returns."""
        batch = buffer[indices]
        obs_next_flat = self._preprocess_obs(batch.obs_next)
        
        # Target actor computes structured action for next_obs
        # The forward method is called with model='_target_actor'
        target_actor_output = self(Batch(obs=obs_next_flat), model='_target_actor') # Pass Batch to forward
        structured_target_actions_dict = target_actor_output.act
        
        # Target critic evaluates Q(s', target_actor(s'))
        with torch.no_grad():
            target_q_val = self._target_critic.q_min(obs_next_flat, structured_target_actions_dict)
        return target_q_val.flatten()


    def process_fn(self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray) -> Batch:
        """Process batch for n-step returns if n_step > 0."""
        if self._n_step > 0:
            # Tianshou's `compute_nstep_return` is usually used here.
            # We provide our own `_get_target_q_values` as the `target_q_fn`.
            return self.compute_nstep_return(
                batch, buffer, indices, self._get_target_q_values,
                self._gamma, self._n_step, self._rew_norm
            )
        # If n_step is 0, Tianshou might expect 'returns' to be pre-calculated or uses 1-step.
        # If 'returns' isn't in batch, calculate 1-step return.
        if not hasattr(batch, 'returns') or batch.returns is None:
             obs_next_flat = self._preprocess_obs(batch.obs_next)
             with torch.no_grad():
                target_actor_output = self(Batch(obs=obs_next_flat), model='_target_actor')
                next_q_values = self._target_critic.q_min(obs_next_flat, target_actor_output.act)
             batch.returns = batch.rew + self._gamma * (1.0 - batch.done) * next_q_values.flatten()
        return batch

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]: # type: ignore
        """Update actor and critic networks."""
        current_obs_flat = self._preprocess_obs(batch.obs)
        
        # --- Critic Update ---
        # Actions from buffer are flat parameters. Parse them for critic.
        # batch.act should be the flat action parameters stored from actor's output (logits field of forward())
        action_from_buffer_flat = to_torch(batch.act, device=self._device, dtype=torch.float32)
        action_for_critic_dict = self._parse_flat_action_to_dict(action_from_buffer_flat)

        target_q = to_torch(batch.returns, device=self._device, dtype=torch.float32).flatten()
        
        current_q1, current_q2 = self._critic(current_obs_flat, action_for_critic_dict)
        current_q1 = current_q1.flatten()
        current_q2 = current_q2.flatten()
        
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self._critic_optim.zero_grad()
        critic_loss.backward()
        self._critic_optim.step()

        # --- Actor Update ---
        # Actor aims to maximize Q(s, actor(s)) or minimize diffusion loss for BC.
        # self(Batch(obs=current_obs_flat), model='_actor') gets current actor's action for s.
        current_actor_output = self(Batch(obs=current_obs_flat), model='_actor')
        structured_current_actor_actions_dict = current_actor_output.act
        
        actor_loss: torch.Tensor
        if self._actor_is_diffusion_model and self.bc_coef_in_actor_diffusion_model:
            # Behavior Cloning loss for diffusion actor (predicts x0)
            if hasattr(batch, 'info') and "expert_action_flat" in batch.info and batch.info.expert_action_flat is not None:
                 expert_actions_flat = to_torch(batch.info.expert_action_flat, device=self._device, dtype=torch.float32)
                 # The actor (Diffusion model) has its own loss method
                 actor_loss = self._actor.loss(x_0_target_action=expert_actions_flat, state_condition=current_obs_flat) # type: ignore
            else: # Fallback if no expert action for BC, use PG-like loss
                 q_val_for_actor = self._critic.q_min(current_obs_flat, structured_current_actor_actions_dict).flatten()
                 actor_loss = -q_val_for_actor.mean()
        else: # Standard policy gradient for actor, or diffusion actor predicting noise (needs careful target for noise loss)
            q_val_for_actor = self._critic.q_min(current_obs_flat, structured_current_actor_actions_dict).flatten()
            actor_loss = -q_val_for_actor.mean() # Simplistic PG for now

        self._actor_optim.zero_grad()
        actor_loss.backward()
        self._actor_optim.step()

        self.sync_weight() # Soft update target networks

        if self._actor_lr_scheduler: self._actor_lr_scheduler.step()
        if self._critic_lr_scheduler: self._critic_lr_scheduler.step()
        
        actor_lr = self._actor_optim.param_groups[0]['lr']
        critic_lr = self._critic_optim.param_groups[0]['lr']

        return { # Return losses for Tianshou TrainingStats
            "loss/critic": critic_loss.item(),
            "loss/actor": actor_loss.item(),
            "lr/actor": actor_lr,
            "lr/critic": critic_lr,
        }
    
    def sync_weight(self) -> None:
        """Soft update target network weights."""
        self.soft_update(self._target_actor, self._actor, self._tau)
        self.soft_update(self._target_critic, self._critic, self._tau)

    # Tianshou's update method signature (if not inheriting from a specific Tianshou policy that defines it)
    def update(self, sample_size: int, buffer: Optional[ReplayBuffer], **kwargs: Any) -> DiffusionOPTStats:
        """Main update entry point called by Tianshou trainer."""
        if buffer is None or len(buffer) < sample_size:
            return DiffusionOPTStats() # Return empty/default stats

        batch, indices = buffer.sample(sample_size)
        # Ensure batch is on the correct device before processing
        batch.to_torch(device=self._device, dtype=torch.float32) # Move non-tensor data as well if needed
        
        # Process batch (e.g., for n-step returns)
        # This should populate batch.returns
        batch = self.process_fn(batch, buffer, indices)
        
        losses = self.learn(batch, **kwargs)
        
        return DiffusionOPTStats(
            critic_loss=losses["loss/critic"],
            actor_loss=losses["loss/actor"],
            actor_lr=losses["lr/actor"],
            critic_lr=losses["lr/critic"]
        )

if __name__ == '__main__':
    from net.critic import DoubleCritic # Assuming this is defined in net.critic
    # This section is for basic structural testing.
    # Full testing requires Tianshou environment and data pipeline.
    print("DiffusionOPT class definition using config for structured actions.")

    # Dummy Actor (Diffusion model) - simplified
    class DummyDiffusionActor(torch.nn.Module):
        def __init__(self, obs_dim, act_dim):
            super().__init__()
            self.fc = torch.nn.Linear(obs_dim, act_dim)
            self.action_dim = act_dim
            self.output_dim = act_dim # For Diffusion class if it infers
            self.max_action_clamp = config.MAX_ACTION_CONTINUOUS_PART
        def forward(self, obs_flat): # Alias for sample in this dummy
            return torch.tanh(self.fc(obs_flat)) * self.max_action_clamp
        def sample(self, obs_flat): return self.forward(obs_flat)
        def loss(self, x_0_target_action, state_condition): # Dummy loss
            pred_x0 = self.forward(state_condition)
            return F.mse_loss(pred_x0, x_0_target_action)

    actor_instance = DummyDiffusionActor(config.FLATTENED_OBS_DIM, config.STRUCTURED_ACTION_FLAT_DIM)
    critic_instance = DoubleCritic(state_dim=config.FLATTENED_OBS_DIM, hidden_dim=64) # From net.critic

    actor_optim_ex = torch.optim.Adam(actor_instance.parameters(), lr=config.LEARNING_RATE_ACTOR)
    critic_optim_ex = torch.optim.Adam(critic_instance.parameters(), lr=config.LEARNING_RATE_CRITIC)

    policy = DiffusionOPT(
        actor=actor_instance, actor_optim=actor_optim_ex,
        critic=critic_instance, critic_optim=critic_optim_ex,
        device='cpu', exploration_noise_std=config.EXPLORATION_NOISE_STD,
        bc_coef_in_actor_diffusion_model=True # Test with BC true
    )

    batch_size_ex = 4
    # Create a Batch object similar to what Tianshou's ReplayBuffer might provide
    dummy_obs_data_dict = {
        'server_state': np.random.rand(batch_size_ex, config.SERVER_STATE_DIM).astype(np.float32),
        'request_queue': np.random.rand(batch_size_ex, config.MAX_REQUESTS_IN_QUEUE_OBS, config.REQUEST_FEATURE_SIZE).astype(np.float32)
    }
    # For Tianshou, obs is often a key in the Batch, e.g. batch.obs
    tianshou_like_batch_obs = Batch(obs=dummy_obs_data_dict)


    policy_output = policy(tianshou_like_batch_obs) # Pass the Batch object
    print("\nPolicy forward output (act - structured dict):")
    for key, val_tensor in policy_output.act.items(): # type: ignore
        print(f"  {key}: shape {val_tensor.shape}")
    print("Policy forward output (logits - flat vector):", policy_output.logits.shape) # type: ignore
    assert policy_output.logits.shape == (batch_size_ex, config.STRUCTURED_ACTION_FLAT_DIM) # type: ignore
    assert policy_output.act["dim_bucket_logits"].shape == (batch_size_ex, config.K_DIM_BUCKETS) # type: ignore

    # Dummy batch for learning
    # In Tianshou, batch.act would typically be the 'logits' (flat action params)
    # and batch.info might contain expert actions.
    # batch.returns would be calculated by process_fn.
    dummy_learn_batch_data = {
        'obs': dummy_obs_data_dict, # This will be preprocessed
        'act': np.random.rand(batch_size_ex, config.STRUCTURED_ACTION_FLAT_DIM).astype(np.float32), # Flat actions
        'rew': np.random.rand(batch_size_ex).astype(np.float32),
        'done': np.zeros(batch_size_ex, dtype=np.bool_),
        'obs_next': dummy_obs_data_dict, # Simplified
        'info': {'expert_action_flat': np.random.rand(batch_size_ex, config.STRUCTURED_ACTION_FLAT_DIM).astype(np.float32)},
        # 'returns' should be populated by process_fn if n_step > 0, or calculated in learn if not.
    }
    # Manually create a Batch object
    # Note: Tianshou's ReplayBuffer.sample() returns a Batch instance.
    # Here, we construct one directly for testing.
    # The `returns` field is critical and usually computed by `process_fn`.
    # For this test, we'll let `process_fn` (with n_step=0 fallback) compute it.
    
    # Constructing a Tianshou Batch object:
    # Tianshou's Batch can be created from a dictionary of lists/arrays.
    # For simplicity, let's assume we have lists of individual experiences.
    # This is a bit contrived for a direct test of `learn` without a buffer.
    
    # Simplified test of learn - assuming `process_fn` populates `returns`
    # In a real setup, `update` would be called, which calls `process_fn` then `learn`.
    
    # To test `learn` directly, we need `batch.returns`. Let's simulate `process_fn` for 1-step.
    test_batch_for_learn = Batch(dummy_learn_batch_data)
    test_batch_for_learn.to_torch(device='cpu') # Move data to device
    
    # Simulate what process_fn (with n_step=0) would do if 'returns' is not present
    if not hasattr(test_batch_for_learn, 'returns') or test_batch_for_learn.returns is None:
        obs_next_flat_learn = policy._preprocess_obs(test_batch_for_learn.obs_next)
        with torch.no_grad():
            target_actor_output_learn = policy(Batch(obs=obs_next_flat_learn), model='_target_actor')
            next_q_values_learn = policy._target_critic.q_min(obs_next_flat_learn, target_actor_output_learn.act) # type: ignore
        test_batch_for_learn.returns = test_batch_for_learn.rew + policy._gamma * (1.0 - test_batch_for_learn.done) * next_q_values_learn.flatten()


    losses = policy.learn(test_batch_for_learn)
    print("\nLosses from learn step:", losses)
    assert "loss/critic" in losses
    assert "loss/actor" in losses
    
    print("\nBasic structure and learn path of DiffusionOPT with config seems plausible.")
