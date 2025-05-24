"""
Actor Network (MLPUNet) for the Diffusion-based Policy.

This module defines the U-Net like MLP architecture used as the core of the
diffusion model actor. It takes the diffused action (x_t_action), 
a timestep (t_timestep), and the conditioning environment observation (s_cond_observation)
as input, and outputs parameters for the structured action (or the predicted noise).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math # Required for SinusoidalPosEmb

import config # Import shared configurations

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal Positional Embedding for timesteps."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        # Corrected embedding calculation from original DDPM, ensuring half_dim >= 1
        if half_dim < 1: 
            # Fallback for very small dimensions, though typically dim is larger
            if self.dim == 1: return x.unsqueeze(-1).float() # Or some other simple mapping
            # This case should ideally not be hit with typical embedding dimensions.
            # For dim=0, it's undefined. For dim=1, it's tricky.
            # Let's assume dim >= 2 for standard sinusoidal embedding.
            # If dim is odd, the last element of emb might be handled differently or dim made even.
            # For simplicity, assuming dim is even and reasonably large.
            # If half_dim is 0 (dim=0 or 1), log(10000)/(half_dim-1) is problematic.
            # A common practice is to ensure dim is even.
            # If dim = 1, half_dim = 0.
            # If dim = 2, half_dim = 1, then half_dim - 1 = 0, division by zero.
            # The formula is often math.log(10000) / (D/2 - 1) where D is the total target dimension for embedding.
            # Or, simpler: emb = torch.exp(torch.arange(half_dim, device=device) * -(math.log(10000.0) / half_dim))
            # Let's use a robust version:
            if half_dim == 0: # if self.dim is 0 or 1
                 if self.dim == 1: return torch.zeros_like(x).unsqueeze(-1).float() # Or x.unsqueeze(-1)
                 return torch.empty(x.shape[0], 0, device=device) # if self.dim is 0

        div_term = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -(math.log(10000.0) / (half_dim -1e-6))) # Avoid div by zero if half_dim is small
        emb = x.float().unsqueeze(1) * div_term.unsqueeze(0) # Ensure x is float
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1: # If original dim is odd, pad or truncate
            emb = emb[:, :self.dim] # Truncate if needed, or handle padding
        return emb


class MLPBlock(nn.Module):
    """A block of MLP layers with Mish activation, BatchNorm, and Dropout, incorporating combined time & condition embedding."""
    def __init__(self, input_dim: int, output_dim: int, dropout_rate: float = 0.1, cond_emb_dim: int = 32):
        super().__init__()
        self.cond_modulation_mlp = nn.Sequential(
            nn.Linear(cond_emb_dim, cond_emb_dim * 2), nn.Mish(),
            nn.Linear(cond_emb_dim * 2, output_dim * 2), nn.Mish()
        )
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, combined_cond_emb: torch.Tensor) -> torch.Tensor:
        is_1d_input_x = x.ndim == 1
        if is_1d_input_x:
            x = x.unsqueeze(0)

        h = F.mish(self.bn1(self.fc1(x)))

        is_1d_input_cond = combined_cond_emb.ndim == 1
        if is_1d_input_cond and not is_1d_input_x : # x is (B,F), cond is (F_cond)
            combined_cond_emb_expanded = combined_cond_emb.unsqueeze(0).expand(x.size(0), -1)
        elif not is_1d_input_cond and is_1d_input_x and combined_cond_emb.size(0) == 1: # x is (F), cond is (1,F_cond)
            combined_cond_emb_expanded = combined_cond_emb.squeeze(0)
        else: # Both batched or both single, or x is (B,F) and cond is (B,F_cond)
            combined_cond_emb_expanded = combined_cond_emb


        modulation_params = self.cond_modulation_mlp(combined_cond_emb_expanded)
        
        # Ensure modulation_params match h's batch dimension if h was originally 1D
        if is_1d_input_x and modulation_params.ndim > 1 and modulation_params.size(0) == 1:
             modulation_params = modulation_params.squeeze(0)
        elif not is_1d_input_x and modulation_params.ndim == 1 and h.size(0) > 1: # h is (B,F_out), modulation_params (F_out*2)
             modulation_params = modulation_params.unsqueeze(0).expand(h.size(0), -1)


        scale, shift = modulation_params.chunk(2, dim=-1)
        h = h * (scale + 1) + shift
        
        h = self.drop(h)
        h = F.mish(self.bn2(self.fc2(h)))
        h = self.drop(h)

        if is_1d_input_x:
            h = h.squeeze(0)
        return h

class MLPUNet(nn.Module):
    """
    MLP-based U-Net architecture for the actor's core model.
    It predicts noise or x0 based on diffused action (x_t_action), timestep (t_timestep),
    and conditioning observation (s_cond_observation).
    """
    def __init__(self,
                 input_dim: int = config.STRUCTURED_ACTION_FLAT_DIM, # For x_t_action
                 output_dim: int = config.STRUCTURED_ACTION_FLAT_DIM,
                 cond_obs_dim: int = config.FLATTENED_OBS_DIM,
                 intermediate_dims: list[int] = [256, 128, 64],
                 time_emb_dim: int = 32,
                 cond_obs_emb_dim: int = 64,
                 dropout_rate: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        hidden_dims = intermediate_dims
        
        self.time_pos_emb = SinusoidalPosEmb(time_emb_dim)
        self.mlp_time_emb = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4), nn.Mish(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        self.cond_obs_mlp = nn.Sequential(
            nn.Linear(cond_obs_dim, cond_obs_emb_dim * 2), nn.Mish(), # Increased capacity
            nn.Linear(cond_obs_emb_dim * 2, cond_obs_emb_dim)
        )

        self.combined_cond_dim = time_emb_dim + cond_obs_emb_dim

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]), nn.Mish(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[0], hidden_dims[0]), nn.Mish(), nn.Dropout(dropout_rate)
        )

        self.encoders = nn.ModuleList()
        for i in range(len(hidden_dims)):
            is_last = i == len(hidden_dims) - 1
            self.encoders.append(nn.ModuleList([
                MLPBlock(hidden_dims[i], hidden_dims[i], dropout_rate, cond_emb_dim=self.combined_cond_dim),
                nn.Linear(hidden_dims[i], hidden_dims[i+1]) if not is_last else nn.Identity()
            ]))

        self.bottleneck1 = MLPBlock(hidden_dims[-1], hidden_dims[-1], dropout_rate, cond_emb_dim=self.combined_cond_dim)
        self.bottleneck2 = MLPBlock(hidden_dims[-1], hidden_dims[-1], dropout_rate, cond_emb_dim=self.combined_cond_dim)

        self.decoders = nn.ModuleList()
        for i in range(len(hidden_dims)):
            current_level_idx = len(hidden_dims) - 1 - i
            decoder_mlp_input_dim = hidden_dims[current_level_idx] * 2
            decoder_mlp_output_dim = hidden_dims[current_level_idx]
            upsample_output_dim = hidden_dims[current_level_idx-1] if current_level_idx > 0 else hidden_dims[0]

            self.decoders.append(nn.ModuleList([
                MLPBlock(decoder_mlp_input_dim, decoder_mlp_output_dim, dropout_rate, cond_emb_dim=self.combined_cond_dim),
                nn.Linear(decoder_mlp_output_dim, upsample_output_dim) if current_level_idx > 0 else nn.Identity()
            ]))
            
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[0]), nn.Mish(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[0], self.output_dim)
        )

    def forward(self, x_t_action: torch.Tensor, t_timestep: torch.Tensor, s_cond_observation: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLPUNet.
        Args:
            x_t_action: The diffused action parameters at current step t. Shape (batch_size, action_dim).
            t_timestep: Timesteps for the diffusion process. Shape (batch_size,).
            s_cond_observation: The conditioning environment observation (flattened). Shape (batch_size, obs_dim).
        Returns:
            Predicted noise or x0. Shape (batch_size, action_dim).
        """
        time_emb = self.time_pos_emb(t_timestep)
        time_emb = self.mlp_time_emb(time_emb)

        cond_obs_emb = self.cond_obs_mlp(s_cond_observation)
        
        # Ensure embeddings are broadcastable if batch sizes differ (e.g. t_timestep might be scalar for some reason)
        if time_emb.shape[0] != x_t_action.shape[0] and time_emb.shape[0] == 1:
            time_emb = time_emb.expand(x_t_action.shape[0], -1)
        if cond_obs_emb.shape[0] != x_t_action.shape[0] and cond_obs_emb.shape[0] == 1:
            cond_obs_emb = cond_obs_emb.expand(x_t_action.shape[0], -1)


        combined_cond_embedding = torch.cat([time_emb, cond_obs_emb], dim=-1)

        x = self.input_proj(x_t_action)
        skip_connections = []
        for mlp_block, downsample_layer in self.encoders:
            x = mlp_block(x, combined_cond_embedding)
            skip_connections.append(x)
            x = downsample_layer(x)

        x = self.bottleneck1(x, combined_cond_embedding)
        x = self.bottleneck2(x, combined_cond_embedding)

        skip_connections = skip_connections[::-1]
        for i in range(len(self.decoders)):
            mlp_block, upsample_layer = self.decoders[i]
            current_skip = skip_connections[i]
            
            # Ensure skip connection batch dim matches x
            if x.shape[0] != current_skip.shape[0]:
                if current_skip.shape[0] == 1: # If skip is (1, F) and x is (B, F)
                    current_skip = current_skip.expand(x.shape[0], -1)
                # Other mismatches might indicate an issue
                elif x.shape[0] ==1: # If x is (1,F) and skip is (B,F) - this should not happen in typical U-Net flow
                     pass # Or raise error

            x = torch.cat((x, current_skip), dim=-1)
            x = mlp_block(x, combined_cond_embedding)
            x = upsample_layer(x)
            
        x = self.output_proj(x)
        return x

if __name__ == "__main__":
    from torchinfo import summary

    BATCH_SIZE = 16
    actor_model = MLPUNet(
        input_dim=config.STRUCTURED_ACTION_FLAT_DIM, # Processes x_t_action
        output_dim=config.STRUCTURED_ACTION_FLAT_DIM,
        cond_obs_dim=config.FLATTENED_OBS_DIM, # For s_cond_observation
        intermediate_dims=[128, 64, 32], 
        time_emb_dim=32,
        cond_obs_emb_dim=64, # Embedding dim for s_cond_observation
        dropout_rate=0.1
    )

    # Dummy inputs matching the new forward signature
    dummy_x_t_action = torch.randn(BATCH_SIZE, config.STRUCTURED_ACTION_FLAT_DIM)
    dummy_s_cond_observation = torch.randn(BATCH_SIZE, config.FLATTENED_OBS_DIM)
    dummy_timesteps = torch.randint(0, config.N_DIFFUSION_TIMESTEPS, (BATCH_SIZE,)).float()

    output_prediction = actor_model(dummy_x_t_action, dummy_timesteps, dummy_s_cond_observation)
    print(f"MLPUNet Output shape (prediction for x0 or noise): {output_prediction.shape}")
    assert output_prediction.shape == (BATCH_SIZE, config.STRUCTURED_ACTION_FLAT_DIM)

    # For summary, provide a list of input tensors or their shapes
    # summary(actor_model, 
    #         input_data=[dummy_x_t_action, dummy_timesteps, dummy_s_cond_observation], 
    #         col_names=["input_size", "output_size", "num_params", "mult_adds"])
    print("\nMLPUNet basic test with 3-argument forward seems OK.")

