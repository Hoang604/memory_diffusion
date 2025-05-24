"""
Diffusion Model for Action Generation.

This module defines the Diffusion class, which wraps a core model (e.g., MLPUNet)
to perform the diffusion and reverse diffusion (sampling) processes for generating
action parameters.
"""
import numpy as np
import torch
import torch.nn as nn

from .helpers import ( # Assuming these are in the same directory or accessible
    cosine_beta_schedule, linear_beta_schedule, vp_beta_schedule,
    extract, Losses
)
from .utils import Progress, Silent # Assuming these are utility classes

import config # Import shared configurations

class Diffusion(nn.Module):
    """
    Diffusion model for generating action parameters.

    This class implements the diffusion process (q_sample) and the reverse
    sampling process (p_sample_loop) to generate action parameters based on
    a conditioning state (environment observation) and a learned model.
    """
    def __init__(self,
                 model: nn.Module, # The core network, e.g., MLPUNet
                 beta_schedule: str = 'vp',
                 n_timesteps: int = config.N_DIFFUSION_TIMESTEPS,
                 action_dim: int = config.STRUCTURED_ACTION_FLAT_DIM, # Dimension of the flat action vector
                 max_action_clamp: float = config.MAX_ACTION_CONTINUOUS_PART, # For clamping continuous action parts
                 loss_type: str = 'l2',
                 clip_denoised: bool = True,
                 # bc_coef: If True, model predicts x0 (action) directly. Else, predicts noise.
                 bc_coef: bool = False):
        super().__init__()
        self.model = model
        self.action_dim = action_dim
        self.max_action_clamp = max_action_clamp # Renamed from max_action to avoid confusion with action_dim

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(n_timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(n_timesteps)
        elif beta_schedule == 'vp':
            betas = vp_beta_schedule(n_timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_shifted_left = torch.cat([torch.ones(1, dtype=betas.dtype, device=betas.device), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.bc_coef = bc_coef

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_shifted_left', alphas_cumprod_shifted_left)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod_minus_one', torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_shifted_left) / (1. - alphas_cumprod)
        posterior_variance = torch.clamp(posterior_variance, min=1e-20) # Avoid log(0)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_shifted_left) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_shifted_left) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        self.loss_fn = Losses[loss_type]()

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise_pred_by_model: torch.Tensor) -> torch.Tensor:
        """Predicts x0 (original action) from diffused xt and predicted noise."""
        if self.bc_coef: # Model directly predicts x0
            return noise_pred_by_model
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recip_alphas_cumprod_minus_one, t, x_t.shape) * noise_pred_by_model
        )

    def q_posterior(self, x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the mean, variance, and log variance of the posterior q(x_{t-1} | x_t, x_0)."""
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor, s_cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the mean and variance for the reverse process p(x_{t-1} | x_t)."""
        model_output = self.model(x_t, t, s_cond) # model(x_diffused, time, state_condition)
        x_0_recon = self.predict_start_from_noise(x_t, t, model_output)

        if self.clip_denoised:
            x_0_recon.clamp_(-self.max_action_clamp, self.max_action_clamp)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_0=x_0_recon, x_t=x_t, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    # @torch.no_grad() # Typically for sampling
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor, s_cond: torch.Tensor) -> torch.Tensor:
        """Samples x_{t-1} from x_t using the reverse process."""
        batch_size = x_t.shape[0]
        model_mean, _, model_log_variance = self.p_mean_variance(x_t=x_t, t=t, s_cond=s_cond)
        noise = torch.randn_like(x_t)
        nonzero_mask = (1 - (t == 0).float()).reshape(batch_size, *((1,) * (len(x_t.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    # @torch.no_grad() # Typically for sampling
    def p_sample_loop(self, state_condition: torch.Tensor, shape: tuple, verbose: bool = False, return_diffusion: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Iteratively samples x0 by reversing the diffusion process from T to 0."""
        device = self.betas.device
        batch_size = shape[0]
        x_t = torch.randn(shape, device=device) # Start from pure noise x_T

        diffusion_history = [x_t.clone()] if return_diffusion else None
        progress = Progress(self.n_timesteps) if verbose else Silent()

        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x_t = self.p_sample(x_t, timesteps, state_condition)
            if return_diffusion:
                diffusion_history.append(x_t.clone())
            progress.update({'t': i})
        progress.close()

        if return_diffusion:
            return x_t, torch.stack(diffusion_history, dim=1)
        return x_t

    # @torch.no_grad() # Typically for sampling
    def sample(self, state_condition: torch.Tensor) -> torch.Tensor:
        """
        Generates action parameters given a state_condition by running the reverse diffusion process.

        Args:
            state_condition: The environment observation (flattened). Shape (batch_size, obs_dim).

        Returns:
            The generated flat action parameters. Shape (batch_size, action_dim).
        """
        batch_size = state_condition.shape[0] if state_condition.ndim > 1 else 1
        # Ensure state_condition is batched for p_sample_loop
        if state_condition.ndim == 1 :
            state_condition_batched = state_condition.unsqueeze(0)
        else:
            state_condition_batched = state_condition
            
        action_shape = torch.Size((batch_size, self.action_dim))
        action_params = self.p_sample_loop(state_condition_batched, action_shape)
        # Clamp final output, especially for continuous parts like priority_weights
        return action_params.clamp_(-self.max_action_clamp, self.max_action_clamp)

    def q_sample(self, x_0_target_action: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Diffuses the target action x0 to timestep t: q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_0_target_action)
        diffused_sample = (
            extract(self.sqrt_alphas_cumprod, t, x_0_target_action.shape) * x_0_target_action +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_0_target_action.shape) * noise
        )
        return diffused_sample, noise

    def p_losses(self, x_0_target_action: torch.Tensor, state_condition: torch.Tensor, t: torch.Tensor, weights: float | torch.Tensor = 1.0) -> torch.Tensor:
        """Computes the loss for training the diffusion model."""
        x_t_diffused, noise_added_to_x_0 = self.q_sample(x_0_target_action, t)
        model_output = self.model(x_t_diffused, t, state_condition)

        if self.bc_coef: # Model predicts x0 (action)
            loss = self.loss_fn(model_output, x_0_target_action, weights)
        else: # Model predicts noise
            loss = self.loss_fn(model_output, noise_added_to_x_0, weights)
        return loss

    def loss(self, x_0_target_action: torch.Tensor, state_condition: torch.Tensor, weights: float | torch.Tensor = 1.0) -> torch.Tensor:
        """
        Computes the total training loss by sampling random timesteps.

        Args:
            x_0_target_action: The "clean" target action parameters (e.g., from expert).
            state_condition: The conditioning environment observation.
            weights: Optional weights for the loss.

        Returns:
            The computed loss value.
        """
        batch_size = x_0_target_action.shape[0]
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x_0_target_action.device).long()
        return self.p_losses(x_0_target_action, state_condition, t, weights)

    def forward(self, state_condition: torch.Tensor) -> torch.Tensor:
        """Alias for sample() during inference."""
        return self.sample(state_condition)

if __name__ == '__main__':
    # Dummy model for testing Diffusion class structure
    class DummyCoreModel(nn.Module):
        def __init__(self, obs_dim, action_dim_out, time_emb_dim=16):
            super().__init__()
            # These attributes are for the Diffusion class to potentially infer action_dim
            self.output_dim = action_dim_out # Critical for Diffusion if action_dim not passed
            self.action_dim = action_dim_out

            self.time_mlp = nn.Sequential(nn.Linear(time_emb_dim,time_emb_dim), nn.Mish(), nn.Linear(time_emb_dim,time_emb_dim))
            # Simplified: model takes concatenation of diffused_action_input, time_emb, and state_condition
            # A real U-Net would be more complex. x_t_diffused is the diffused action here.
            self.fc = nn.Linear(action_dim_out + time_emb_dim + obs_dim, action_dim_out)

        def forward(self, x_t_diffused: torch.Tensor, t: torch.Tensor, s_cond: torch.Tensor) -> torch.Tensor:
            # Create a dummy time embedding
            if t.ndim == 1: # t is (B,)
                t_emb = torch.zeros(t.size(0), 16, device=t.device)
                for i in range(t.size(0)): t_emb[i, t[i].item() % 16] = 1.0
            else: # t is scalar
                t_emb = torch.zeros(1, 16, device=s_cond.device)
                t_emb[0, t.item() % 16] = 1.0
                if s_cond.ndim > 1 and s_cond.size(0) > 1: t_emb = t_emb.expand(s_cond.size(0), -1)
            
            # Ensure s_cond is batched if x_t_diffused is
            if x_t_diffused.ndim > 1 and s_cond.ndim == 1 and x_t_diffused.size(0) > 1:
                s_cond = s_cond.unsqueeze(0).expand(x_t_diffused.size(0), -1)
            if x_t_diffused.ndim == 1 and s_cond.ndim > 1 and s_cond.size(0) == 1: # x_t is (F), s_cond is (1,F_obs)
                s_cond = s_cond.squeeze(0)


            combined_input = torch.cat([x_t_diffused, t_emb, s_cond], dim=-1)
            return self.fc(combined_input)

    BATCH_S = 4
    core_model = DummyCoreModel(
        obs_dim=config.FLATTENED_OBS_DIM,
        action_dim_out=config.STRUCTURED_ACTION_FLAT_DIM
    )
    diffusion_process = Diffusion(
        core_model,
        n_timesteps=config.N_DIFFUSION_TIMESTEPS,
        action_dim=config.STRUCTURED_ACTION_FLAT_DIM, # Explicitly pass, or ensure core_model.output_dim is set
        bc_coef=True # Test with bc_coef=True (model predicts x0)
    )

    dummy_state_obs = torch.randn(BATCH_S, config.FLATTENED_OBS_DIM)
    
    print("Testing Diffusion sampling...")
    sampled_action_params = diffusion_process.sample(dummy_state_obs)
    print("Sampled action params shape:", sampled_action_params.shape)
    assert sampled_action_params.shape == (BATCH_S, config.STRUCTURED_ACTION_FLAT_DIM)

    print("\nTesting Diffusion loss calculation...")
    dummy_x0_target = torch.randn(BATCH_S, config.STRUCTURED_ACTION_FLAT_DIM)
    # Simulate clamping for continuous part if necessary (e.g. priority weights)
    pw_start_idx = config.K_DIM_BUCKETS + config.M_BATCH_SIZE_OPTIONS
    dummy_x0_target[:, pw_start_idx:] = torch.tanh(dummy_x0_target[:, pw_start_idx:]) * config.MAX_ACTION_CONTINUOUS_PART

    loss_val = diffusion_process.loss(dummy_x0_target, dummy_state_obs)
    print("Calculated loss:", loss_val.item())
    assert loss_val.ndim == 0

    print("\nDiffusion class basic tests with config dimensions seem OK.")

