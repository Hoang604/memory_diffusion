"""
Actor Network (MLPUNet) for the Diffusion-based Policy.

This module defines the U-Net like MLP architecture used as the core of the
diffusion model actor. It takes the environment observation and a timestep
as input and outputs parameters for the structured action.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import config # Import shared configurations

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal Positional Embedding for timesteps."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class MLPBlock(nn.Module):
    """A block of MLP layers with Mish activation, BatchNorm, and Dropout, incorporating time embedding."""
    def __init__(self, input_dim: int, output_dim: int, dropout_rate: float = 0.1, t_dim: int = 32):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim * 2), nn.Mish(),
            nn.Linear(t_dim * 2, output_dim * 2), nn.Mish()
        )
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(output_dim) # Changed from bn to bn1 for clarity
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # Handle potential 1D input for BatchNorm
        is_1d_input = x.ndim == 1
        if is_1d_input:
            x = x.unsqueeze(0)

        h = F.mish(self.bn1(self.fc1(x)))

        # Process time embedding and apply FiLM-like modulation
        t_emb_processed = self.time_mlp(t_emb)
        if is_1d_input and t_emb_processed.ndim > 1: # Ensure t_emb matches x's (un)squeezed dim
             t_emb_processed = t_emb_processed.squeeze(0)
        elif not is_1d_input and t_emb_processed.ndim == 1 and x.size(0) > 1:
             t_emb_processed = t_emb_processed.unsqueeze(0).expand(x.size(0), -1)


        scale, shift = t_emb_processed.chunk(2, dim=-1)
        h = h * (scale + 1) + shift
        h = self.drop(h)
        h = F.mish(self.bn2(self.fc2(h)))
        h = self.drop(h)

        if is_1d_input:
            h = h.squeeze(0)
        return h

class MLPUNet(nn.Module):
    """
    MLP-based U-Net architecture for the actor.

    This network takes a flattened observation and a timestep embedding as input,
    and outputs a flat vector representing the parameters for a structured action.
    The output dimension is determined by `config.STRUCTURED_ACTION_FLAT_DIM`.
    """
    def __init__(self,
                 input_dim: int = config.FLATTENED_OBS_DIM,
                 output_dim: int = config.STRUCTURED_ACTION_FLAT_DIM,
                 intermediate_dims: list[int] = [256, 128, 64], # Example, configure as needed
                 time_emb_dim: int = 32, # Dimension of the sinusoidal time embedding
                 dropout_rate: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim # Should be config.STRUCTURED_ACTION_FLAT_DIM

        hidden_dims = intermediate_dims
        self.time_pos_emb = SinusoidalPosEmb(time_emb_dim)
        self.mlp_time_emb = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4), nn.Mish(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]), nn.Mish(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[0], hidden_dims[0]), nn.Mish(), nn.Dropout(dropout_rate)
        )

        self.encoders = nn.ModuleList()
        for i in range(len(hidden_dims)):
            is_last = i == len(hidden_dims) - 1
            self.encoders.append(nn.ModuleList([
                MLPBlock(hidden_dims[i], hidden_dims[i], dropout_rate, t_dim=time_emb_dim),
                nn.Linear(hidden_dims[i], hidden_dims[i+1]) if not is_last else nn.Identity()
            ]))

        self.bottleneck1 = MLPBlock(hidden_dims[-1], hidden_dims[-1], dropout_rate, t_dim=time_emb_dim)
        self.bottleneck2 = MLPBlock(hidden_dims[-1], hidden_dims[-1], dropout_rate, t_dim=time_emb_dim)

        self.decoders = nn.ModuleList()
        for i in range(len(hidden_dims)):
            is_first = i == 0 # Corresponds to the largest feature map before output projection
            current_level_idx = len(hidden_dims) - 1 - i # Index from the end of hidden_dims
            
            # Input to decoder MLPBlock is current_level_dim (from upsample) + skip_connection_dim (same as current_level_dim)
            decoder_mlp_input_dim = hidden_dims[current_level_idx] * 2
            decoder_mlp_output_dim = hidden_dims[current_level_idx]
            
            # Upsample layer maps to the dimension of the next (larger) decoder stage, or is Identity if this is the last decoder stage before output_proj
            upsample_output_dim = hidden_dims[current_level_idx-1] if current_level_idx > 0 else hidden_dims[0]

            self.decoders.append(nn.ModuleList([
                MLPBlock(decoder_mlp_input_dim, decoder_mlp_output_dim, dropout_rate, t_dim=time_emb_dim),
                nn.Linear(decoder_mlp_output_dim, upsample_output_dim) if current_level_idx > 0 else nn.Identity()
            ]))
            
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[0]), nn.Mish(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[0], self.output_dim)
        )

    def forward(self, x_obs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLPUNet.

        Args:
            x_obs: The flattened environment observation. Shape (batch_size, input_dim).
            t: Timesteps for the diffusion process. Shape (batch_size,).

        Returns:
            A flat tensor of action parameters. Shape (batch_size, output_dim).
        """
        t_emb = self.time_pos_emb(t)
        t_emb = self.mlp_time_emb(t_emb)

        x = self.input_proj(x_obs)
        skip_connections = []
        for mlp_block, downsample_layer in self.encoders:
            x = mlp_block(x, t_emb)
            skip_connections.append(x)
            x = downsample_layer(x)

        x = self.bottleneck1(x, t_emb)
        x = self.bottleneck2(x, t_emb)

        skip_connections = skip_connections[::-1]
        for i in range(len(self.decoders)):
            mlp_block, upsample_layer = self.decoders[i]
            current_skip = skip_connections[i]
            
            # Ensure skip connection has batch dim if x has it and shapes are compatible for cat
            if x.ndim > 1 and current_skip.ndim == 1 and x.size(0) > 1 :
                 current_skip = current_skip.unsqueeze(0).expand(x.size(0), -1)
            
            if x.shape[0] != current_skip.shape[0] and x.shape[0] == 1 and current_skip.shape[0] > 1:
                # If x is single sample and skip is batched (should not happen if input is consistent)
                # This case is tricky, usually U-Net expects consistent batching.
                # For now, we assume this won't happen or needs specific handling.
                pass # Or raise error
            elif x.shape[0] != current_skip.shape[0] and current_skip.shape[0] == 1 and x.shape[0] > 1:
                 current_skip = current_skip.expand(x.shape[0], -1)


            x = torch.cat((x, current_skip), dim=-1)
            x = mlp_block(x, t_emb)
            x = upsample_layer(x)
            
        x = self.output_proj(x)
        return x

if __name__ == "__main__":
    from torchinfo import summary

    BATCH_SIZE = 16
    # Using dimensions from config.py
    actor_model = MLPUNet(
        input_dim=config.FLATTENED_OBS_DIM,
        output_dim=config.STRUCTURED_ACTION_FLAT_DIM,
        intermediate_dims=[128, 64, 32], # Example intermediate dims
        time_emb_dim=32, # Example time embedding dim
        dropout_rate=0.1
    )

    dummy_state_obs = torch.randn(BATCH_SIZE, config.FLATTENED_OBS_DIM)
    dummy_timesteps = torch.randint(0, config.N_DIFFUSION_TIMESTEPS, (BATCH_SIZE,)).float()

    output_action_params = actor_model(dummy_state_obs, dummy_timesteps)
    print(f"MLPUNet Output shape: {output_action_params.shape}")
    assert output_action_params.shape == (BATCH_SIZE, config.STRUCTURED_ACTION_FLAT_DIM)

    # Example of how the output might be sliced (this logic is in the policy)
    k_logits_ex = output_action_params[:, :config.K_DIM_BUCKETS]
    m_logits_ex = output_action_params[:, config.K_DIM_BUCKETS : config.K_DIM_BUCKETS + config.M_BATCH_SIZE_OPTIONS]
    priority_w_ex = output_action_params[:, config.K_DIM_BUCKETS + config.M_BATCH_SIZE_OPTIONS:]
    
    print(f"  Example k_dim_logits shape: {k_logits_ex.shape}")
    print(f"  Example m_batch_size_logits shape: {m_logits_ex.shape}")
    print(f"  Example priority_weights shape: {priority_w_ex.shape}")

    # summary(actor_model, input_data=[dummy_state_obs, dummy_timesteps], col_names=["input_size", "output_size", "num_params", "mult_adds"])
    print("\nMLPUNet basic test with config dimensions seems OK.")
