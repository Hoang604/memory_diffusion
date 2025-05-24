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
from tianshou.data import Batch, ReplayBuffer, to_torch, to_numpy # Added to_numpy
from tianshou.policy import BasePolicy # Assuming Tianshou BasePolicy
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataclasses import dataclass, field
from gymnasium import spaces # Import gymnasium.spaces
from tianshou.policy.base import TrainingStats


import config # Import shared configurations

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
            lr_decay: bool = False,
            lr_maxt: int = 1000, # T_max for CosineAnnealingLR
            bc_coef_in_actor_diffusion_model: bool = False,
            action_space = None, 
            **kwargs: Any
    ) -> None:
        dummy_super_action_space = spaces.Box(
            low=-config.MAX_ACTION_CONTINUOUS_PART, 
            high=config.MAX_ACTION_CONTINUOUS_PART, 
            shape=(config.STRUCTURED_ACTION_FLAT_DIM,),
            dtype=np.float32
        )
        super().__init__(action_space=dummy_super_action_space, **kwargs)
        
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
            if hasattr(self, '_actor_optim') and self._actor_optim is not None:
                self._actor_lr_scheduler = CosineAnnealingLR(self._actor_optim, T_max=lr_maxt, eta_min=0.)
            if hasattr(self, '_critic_optim') and self._critic_optim is not None:
                self._critic_lr_scheduler = CosineAnnealingLR(self._critic_optim, T_max=lr_maxt, eta_min=0.)
        
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

    def _preprocess_obs(self, obs_input: Union[np.ndarray, torch.Tensor, Dict, Batch]) -> torch.Tensor:
        """Preprocesses observations to the flat tensor format expected by networks."""
        
        actual_obs_payload: Union[Dict, Batch, np.ndarray, torch.Tensor]
        
        if isinstance(obs_input, Batch) and hasattr(obs_input, 'obs'):
            actual_obs_payload = obs_input.obs
        else:
            actual_obs_payload = obs_input

        if isinstance(actual_obs_payload, dict):
            server_state = to_torch(actual_obs_payload['server_state'], device=self._device, dtype=torch.float32)
            request_queue = to_torch(actual_obs_payload['request_queue'], device=self._device, dtype=torch.float32)
            
            if server_state.ndim == 1: server_state = server_state.unsqueeze(0) 
            if request_queue.ndim == 2: request_queue = request_queue.unsqueeze(0) 
            
            batch_size = server_state.shape[0]
            request_queue_flat = request_queue.reshape(batch_size, -1)
            flat_obs = torch.cat([server_state, request_queue_flat], dim=1)
            return flat_obs
        
        elif isinstance(actual_obs_payload, Batch):
            server_state = to_torch(actual_obs_payload.server_state, device=self._device, dtype=torch.float32)
            request_queue = to_torch(actual_obs_payload.request_queue, device=self._device, dtype=torch.float32)
            
            batch_size = server_state.shape[0]
            request_queue_flat = request_queue.reshape(batch_size, -1) 
            flat_obs = torch.cat([server_state, request_queue_flat], dim=1)
            return flat_obs
            
        elif torch.is_tensor(actual_obs_payload) or isinstance(actual_obs_payload, np.ndarray):
            processed_obs = to_torch(actual_obs_payload, device=self._device, dtype=torch.float32)
            if processed_obs.ndim == 1: 
                processed_obs = processed_obs.unsqueeze(0)
            return processed_obs
        else:
            raise TypeError(f"Unsupported actual_obs_payload type after initial processing: {type(actual_obs_payload)}")


    def forward(self, batch: Batch, state: Optional[Any] = None, model: str = "_actor", **kwargs: Any) -> Batch:
        """
        Computes actions for the given batch of observations.
        'model' arg specifies whether to use self._actor or self._target_actor.
        """
        current_obs_flat = self._preprocess_obs(batch) 
        actor_nn = getattr(self, model) 
        
        flat_action_params = actor_nn(current_obs_flat) 
        
        if self.training and self._noise is not None and model == "_actor": 
            noise_val = self._noise(flat_action_params.shape) 
            noise_val_torch = to_torch(noise_val, device=flat_action_params.device, dtype=flat_action_params.dtype)
            flat_action_params = flat_action_params + noise_val_torch
            
            pw_start_idx = config.K_DIM_BUCKETS + config.M_BATCH_SIZE_OPTIONS
            clamp_val = config.MAX_ACTION_CONTINUOUS_PART
            if hasattr(actor_nn, 'max_action_clamp'): 
                 clamp_val = actor_nn.max_action_clamp
            
            flat_action_params[..., pw_start_idx:] = \
                flat_action_params[..., pw_start_idx:].clamp_(-clamp_val, clamp_val)

        structured_action_dict = self._parse_flat_action_to_dict(flat_action_params)
        
        return Batch(act=structured_action_dict, logits=flat_action_params, state=state)

    # --- Overriding BasePolicy methods for Dict action space ---
    def map_action(self, act: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Map raw network output to action for the environment.
        For our Dict action space, the 'act' from forward() is already the
        structured dictionary of tensors. The environment expects this dictionary.
        No further transformation is typically needed here, but values could be
        scaled or clipped if necessary (though clamping is done in forward for noise).
        The main purpose here is to ensure it's passed through correctly.
        """
        # 'act' is the dictionary of tensors:
        # {'dim_bucket_logits': ..., 'batch_size_logits': ..., 'priority_weights': ...}
        # Our environment's step() method expects this dictionary.
        return act

    def _action_to_numpy(self, act: Any) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Convert action to numpy. Handles our structured dictionary action.
        This is called by the Collector before passing the action to env.step().
        """
        if isinstance(act, dict):
            # Convert tensor values in the dictionary to numpy arrays
            return {k: to_numpy(v) for k, v in act.items()}
        # Fallback to parent's implementation for standard Box/Discrete actions
        # (though not expected if map_action always returns our dict for this policy)
        return super()._action_to_numpy(act)
    # --- End of Overrides ---


    def _get_target_q_values(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        """Helper function to compute target Q-values for n-step returns."""
        batch = buffer[indices]
        obs_next_flat = self._preprocess_obs(batch.obs_next) 
        
        target_actor_output = self(Batch(obs=obs_next_flat), model='_target_actor') 
        structured_target_actions_dict = target_actor_output.act
        
        with torch.no_grad():
            target_q_val = self._target_critic.q_min(obs_next_flat, structured_target_actions_dict)
        return target_q_val.flatten()


    def process_fn(self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray) -> Batch:
        """Process batch for n-step returns if n_step > 0."""
        if self._n_step > 0:
            return self.compute_nstep_return(
                batch, buffer, indices, self._get_target_q_values,
                self._gamma, self._n_step, self._rew_norm
            )
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
        
        action_for_critic_dict = {}
        # batch.act is the structured dictionary from the replay buffer
        for key, val in batch.act.items(): 
            action_for_critic_dict[key] = to_torch(val, device=self._device, dtype=torch.float32)

        target_q = to_torch(batch.returns, device=self._device, dtype=torch.float32).flatten()
        
        current_q1, current_q2 = self._critic(current_obs_flat, action_for_critic_dict)
        current_q1 = current_q1.flatten()
        current_q2 = current_q2.flatten()
        
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self._critic_optim.zero_grad()
        critic_loss.backward()
        self._critic_optim.step()

        current_actor_output = self(Batch(obs=current_obs_flat), model='_actor')
        structured_current_actor_actions_dict = current_actor_output.act
        
        actor_loss: torch.Tensor
        if self._actor_is_diffusion_model and self.bc_coef_in_actor_diffusion_model:
            expert_action_flat_data = batch.info.get("expert_action_flat", None) if hasattr(batch, 'info') else None
            if expert_action_flat_data is not None:
                 expert_actions_flat = to_torch(expert_action_flat_data, device=self._device, dtype=torch.float32)
                 actor_loss = self._actor.loss(x_0_target_action=expert_actions_flat, state_condition=current_obs_flat) # type: ignore
            else: 
                 q_val_for_actor = self._critic.q_min(current_obs_flat, structured_current_actor_actions_dict).flatten()
                 actor_loss = -q_val_for_actor.mean()
        else: 
            q_val_for_actor = self._critic.q_min(current_obs_flat, structured_current_actor_actions_dict).flatten()
            actor_loss = -q_val_for_actor.mean()

        self._actor_optim.zero_grad()
        actor_loss.backward()
        self._actor_optim.step()

        self.sync_weight() 

        if self._actor_lr_scheduler: self._actor_lr_scheduler.step()
        if self._critic_lr_scheduler: self._critic_lr_scheduler.step()
        
        actor_lr = self._actor_optim.param_groups[0]['lr'] if self._actor_optim else 0.0
        critic_lr = self._critic_optim.param_groups[0]['lr'] if self._critic_optim else 0.0

        return { 
            "loss/critic": critic_loss.item(),
            "loss/actor": actor_loss.item(),
            "lr/actor": actor_lr,
            "lr/critic": critic_lr,
        }
    
    def sync_weight(self) -> None:
        """Soft update target network weights."""
        if hasattr(self, '_target_actor') and self._target_actor is not None:
            self.soft_update(self._target_actor, self._actor, self._tau)
        if hasattr(self, '_target_critic') and self._target_critic is not None:
            self.soft_update(self._target_critic, self._critic, self._tau)

    def update(self, sample_size: int, buffer: Optional[ReplayBuffer], **kwargs: Any) -> DiffusionOPTStats:
        """Main update entry point called by Tianshou trainer."""
        if buffer is None or len(buffer) == 0 or len(buffer) < sample_size : 
            return DiffusionOPTStats() 

        batch, indices = buffer.sample(sample_size)
        batch.to_torch(device=self._device) 
        
        batch = self.process_fn(batch, buffer, indices)
        
        losses = self.learn(batch, **kwargs)
        
        return DiffusionOPTStats(
            critic_loss=losses["loss/critic"],
            actor_loss=losses["loss/actor"],
            actor_lr=losses["lr/actor"],
            critic_lr=losses["lr/critic"]
        )

if __name__ == '__main__':
    print("DiffusionOPT class definition using config for structured actions.")
    class DummyDiffusionActor(torch.nn.Module):
        def __init__(self, obs_dim, act_dim):
            super().__init__()
            self.fc = torch.nn.Linear(obs_dim, act_dim)
            self.action_dim = act_dim
            self.output_dim = act_dim 
            self.max_action_clamp = config.MAX_ACTION_CONTINUOUS_PART
        def forward(self, obs_flat): 
            return torch.tanh(self.fc(obs_flat)) * self.max_action_clamp
        def sample(self, obs_flat): return self.forward(obs_flat)
        def loss(self, x_0_target_action, state_condition): 
            pred_x0 = self.forward(state_condition)
            return F.mse_loss(pred_x0, x_0_target_action)

    try:
        from net.critic import DoubleCritic
    except ImportError:
        print("Warning: Could not import DoubleCritic from net.critic. Using a local dummy for test.")
        class DoubleCritic(torch.nn.Module): 
            def __init__(self, state_dim, hidden_dim=64):
                super().__init__()
                self.fc1 = torch.nn.Linear(state_dim + config.PROCESSED_ACTION_DIM_FOR_CRITIC, hidden_dim)
                self.fc2 = torch.nn.Linear(hidden_dim, 1)
            def forward(self, s, a_dict): 
                # This dummy doesn't correctly process a_dict, it's for structural test
                dummy_action_flat = torch.randn(s.shape[0], config.PROCESSED_ACTION_DIM_FOR_CRITIC, device=s.device)
                x = torch.cat([s, dummy_action_flat], dim=-1)
                return self.fc2(F.relu(self.fc1(x))), self.fc2(F.relu(self.fc1(x)))
            def q_min(self, s, a_dict):
                q1, _ = self.forward(s,a_dict)
                return q1

    actor_instance = DummyDiffusionActor(config.FLATTENED_OBS_DIM, config.STRUCTURED_ACTION_FLAT_DIM)
    critic_instance = DoubleCritic(state_dim=config.FLATTENED_OBS_DIM, hidden_dim=64) 

    actor_optim_ex = torch.optim.Adam(actor_instance.parameters(), lr=config.LEARNING_RATE_ACTOR)
    critic_optim_ex = torch.optim.Adam(critic_instance.parameters(), lr=config.LEARNING_RATE_CRITIC)

    policy = DiffusionOPT(
        actor=actor_instance, actor_optim=actor_optim_ex,
        critic=critic_instance, critic_optim=critic_optim_ex,
        device='cpu', exploration_noise_std=config.EXPLORATION_NOISE_STD,
        bc_coef_in_actor_diffusion_model=True 
    )

    batch_size_ex = 4
    dummy_obs_for_batch = { 
        'server_state': np.random.rand(batch_size_ex, config.SERVER_STATE_DIM).astype(np.float32),
        'request_queue': np.random.rand(batch_size_ex, config.MAX_REQUESTS_IN_QUEUE_OBS, config.REQUEST_FEATURE_SIZE).astype(np.float32)
    }
    
    tianshou_input_batch = Batch(obs=dummy_obs_for_batch) 

    policy.train() 
    policy_output_train = policy(tianshou_input_batch) 
    
    # Test the overridden map_action and _action_to_numpy
    # policy_output_train.act is a dict of Tensors
    mapped_action = policy.map_action(policy_output_train.act) # Should return dict of Tensors
    numpy_action = policy._action_to_numpy(mapped_action) # Should return dict of NumPy arrays

    print("\nOutput of map_action (should be dict of Tensors):")
    for k,v in mapped_action.items(): print(f"  {k}: type {type(v)}")
    print("Output of _action_to_numpy (should be dict of NumPy arrays):")
    for k,v in numpy_action.items(): print(f"  {k}: type {type(v)}")
    assert isinstance(numpy_action["dim_bucket_logits"], np.ndarray)


    policy.eval() 
    policy_output_eval = policy(tianshou_input_batch)
    
    print("\nPolicy forward output (act - structured dict):")
    for key, val_tensor in policy_output_eval.act.items(): 
        print(f"  {key}: shape {val_tensor.shape}")
    print("Policy forward output (logits - flat vector):", policy_output_eval.logits.shape) 
    assert policy_output_eval.logits.shape == (batch_size_ex, config.STRUCTURED_ACTION_FLAT_DIM) 
    assert policy_output_eval.act["dim_bucket_logits"].shape == (batch_size_ex, config.K_DIM_BUCKETS) 

    act_dict_for_buffer = {
        "dim_bucket_logits": np.random.randn(batch_size_ex, config.K_DIM_BUCKETS).astype(np.float32),
        "batch_size_logits": np.random.randn(batch_size_ex, config.M_BATCH_SIZE_OPTIONS).astype(np.float32),
        "priority_weights": np.tanh(np.random.randn(batch_size_ex, config.N_PRIORITY_WEIGHTS)).astype(np.float32) * config.MAX_ACTION_CONTINUOUS_PART
    }
    learn_batch_data_dict = {
        'obs': dummy_obs_for_batch, 
        'act': act_dict_for_buffer, 
        'rew': np.random.rand(batch_size_ex).astype(np.float32),
        'done': np.zeros(batch_size_ex, dtype=np.bool_),
        'obs_next': dummy_obs_for_batch, 
        'info': Batch(expert_action_flat=np.random.rand(batch_size_ex, config.STRUCTURED_ACTION_FLAT_DIM).astype(np.float32)),
        'returns': np.random.rand(batch_size_ex).astype(np.float32) 
    }
    test_batch_for_learn = Batch(learn_batch_data_dict)
    
    test_batch_for_learn.to_torch(device='cpu') 
    
    losses = policy.learn(test_batch_for_learn)
    print("\nLosses from learn step:", losses)
    assert "loss/critic" in losses
    assert "loss/actor" in losses
    
    print("\nBasic structure and learn path of DiffusionOPT with config seems plausible.")

