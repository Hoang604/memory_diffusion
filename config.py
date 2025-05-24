"""
Configuration file for the Reinforcement Learning Photo Editing Server.

This file contains shared parameters used across different modules of the project,
including environment settings, action space definitions, observation space
definitions, network architectures, and training hyperparameters.
"""

import numpy as np

# --- Action Space Configuration ---
# Defines the structure of the action output by the policy's actor network.

K_DIM_BUCKETS: int = 4
"""Number of discrete buckets for image dimensions (e.g., 256x256, 512x512)."""

M_BATCH_SIZE_OPTIONS: int = 4
"""Number of discrete options for batch sizes (e.g., 1, 2, 4, 8 images)."""

N_PRIORITY_WEIGHTS: int = 3
"""Number of continuous priority weights the actor learns (e.g., for speed, waiting time)."""

STRUCTURED_ACTION_FLAT_DIM: int = K_DIM_BUCKETS + M_BATCH_SIZE_OPTIONS + N_PRIORITY_WEIGHTS
"""Total dimension of the flat action vector produced by the diffusion model."""


# --- Environment Configuration ---

MAX_REQUESTS_IN_QUEUE_OBS: int = 20
"""Maximum number of requests the observation space's 'request_queue' can hold."""

REQUEST_FEATURE_SIZE: int = 4
"""Number of features per request: [token_count, internet_speed, img_size, waiting_time]."""

IMG_SIZE_BUCKETS: np.ndarray = np.array([256, 512, 768, 1024])
"""Actual image sizes corresponding to each dimension bucket index."""

BATCH_SIZE_OPTIONS: np.ndarray = np.array([1, 2, 4, 8])
"""Actual batch sizes corresponding to each batch size option index."""

MAX_STEPS_PER_EPISODE: int = 200
"""Maximum number of simulation steps per episode in the environment."""


# --- Observation Space Configuration ---

SERVER_STATE_DIM: int = 4
"""Dimension of the 'server_state': [M_avail, G_avail, B_avail, N_connections_avail]."""

FLATTENED_OBS_DIM: int = SERVER_STATE_DIM + (MAX_REQUESTS_IN_QUEUE_OBS * REQUEST_FEATURE_SIZE)
"""Total dimension of the flattened observation vector for actor/critic input."""


# --- Critic Network Configuration ---

PROCESSED_ACTION_DIM_FOR_CRITIC: int = K_DIM_BUCKETS + M_BATCH_SIZE_OPTIONS + N_PRIORITY_WEIGHTS
"""Dimension of the processed action vector fed into the critic's action_mlp."""


# --- Diffusion Model (Actor) Configuration ---

N_DIFFUSION_TIMESTEPS: int = 5
"""Number of timesteps for the diffusion process in the actor model."""

MAX_ACTION_CONTINUOUS_PART: float = 1.0
"""Max absolute value for continuous parts of action (e.g., priority weights in [-1,1])."""

# --- Training Hyperparameters (Examples) ---

LEARNING_RATE_ACTOR: float = 1e-4
"""Learning rate for the actor optimizer."""

LEARNING_RATE_CRITIC: float = 3e-4
"""Learning rate for the critic optimizer."""

GAMMA: float = 0.99
"""Discount factor for future rewards."""

TAU: float = 0.005
"""Soft update coefficient for target networks."""

BATCH_SIZE_TRAINING: int = 64
"""Batch size for training the agent."""

EXPLORATION_NOISE_STD: float = 0.1
"""Standard deviation for Gaussian exploration noise added to actions."""

# --- Network Architecture (Examples) ---
# These can be used to configure the hidden layers of MLPUNet and Critic

# Example: Hidden dimensions for the U-Net encoder/decoder paths in MLPUNet
# ACTOR_MLP_UNET_INTERMEDIATE_DIMS: list[int] = [256, 128, 64, 32]

# Example: Hidden dimension for the critic's MLPs
# CRITIC_HIDDEN_DIM: int = 256

