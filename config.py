"""
Configuration file for the Reinforcement Learning Photo Editing Server.

This file contains shared parameters used across different modules of the project,
including environment settings, action space definitions, observation space
definitions, network architectures, and training hyperparameters.

All configurations related to image sizes, batch sizes, processing times,
and peak memory usage are derived from benchmark data (e.g., 'benchmark_results_refine.csv').
"""

import numpy as np

# --- Dynamically generated from benchmark_results_refine.csv ---
# These values are derived from your CSV data.

IMG_SIZE_BUCKETS = np.array([256, 512, 768, 1024, 1536, 2048], dtype=np.float32)
"""
Defines the discrete image sizes (based on width, assuming square or width-dominant)
that the agent can choose or that appear in requests.
Populated from the 'Width' column of the benchmark CSV.
"""

BATCH_SIZE_OPTIONS = np.array([1, 2, 4, 8, 12, 16, 24, 32], dtype=np.int32)
"""
Defines the discrete batch size options the agent can choose.
Populated from the 'BatchSize' column of the benchmark CSV.
"""

K_DIM_BUCKETS: int = 6
"""
Number of discrete image size buckets. Corresponds to the length of `IMG_SIZE_BUCKETS`.
Derived from unique 'Width' values in the benchmark CSV.
"""

M_BATCH_SIZE_OPTIONS: int = 8
"""
Number of discrete batch size options. Corresponds to the length of `BATCH_SIZE_OPTIONS`.
Derived from unique 'BatchSize' values in the benchmark CSV.
"""

# PROCESSING_TIMES_BY_INDEX (Time_s from CSV, averaged over Height for same Width-BatchSize)
# Rows: IMG_SIZE_BUCKETS (based on Width), Columns: BATCH_SIZE_OPTIONS
PROCESSING_TIMES_BY_INDEX = np.array([
    [0.732884, 1.251707, 2.216377, 4.162984, 6.172665, 7.965257, 12.019527, 16.064785], # 256x256
    [3.013681, 4.860316, 8.879983, 17.105791, 25.34041, 33.985377, 29.076608, 38.72665], # 512 x256
    [6.232718, 12.593526, 25.867722, 52.81025, 66.613857, np.nan, np.nan, np.nan], # 768x512, Handle NaN: value for (768, 16), (768,24), (768,32) missing 
    [14.006143, 30.776509, 63.680154, 68.954202, np.nan, np.nan, np.nan, np.nan], # 1024x512, Handle NaN
    [54.535401, 121.255513, 147.341923, np.nan, np.nan, np.nan, np.nan, np.nan],    # 1536x768 Handle NaN
    [253.366551, 130.241431, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]     # 2048x1024 Handle NaN
])
"""
2D array storing processing times in seconds (Time_s from CSV).
Rows correspond to `IMG_SIZE_BUCKETS` indices (based on Width).
Columns correspond to `BATCH_SIZE_OPTIONS` indices.
Values are averaged over different 'Height' if multiple exist for the same Width-BatchSize pair.
IMPORTANT: NaN values must be replaced with sensible defaults (e.g., float('inf') or extrapolated values)
if those combinations are possible for the agent to choose or encounter.
"""

# PEAK_MEMORY_GB_BY_INDEX (PeakMemory_GB from CSV, averaged over Height for same Width-BatchSize)
# This represents the memory for the *entire batch* of this type.
# Rows: IMG_SIZE_BUCKETS (based on Width), Columns: BATCH_SIZE_OPTIONS
PEAK_MEMORY_GB_BY_INDEX = np.array([
    [0.252957, 0.47593, 0.946534, 1.887574, 2.829667, 3.77118, 5.654206, 7.537965], 
    [0.753201, 1.413918, 2.822511, 5.639527, 8.457597, 9.773028, 11.282136, 12.037789], 
    [1.87875, 3.524392, 7.043679, 12.392389, 13.520778, np.nan, np.nan, np.nan],  # Handle NaN
    [3.0043, 5.634866, 10.262943, 12.016253, np.nan, np.nan, np.nan, np.nan],   # Handle NaN
    [6.50609, 11.074534, 13.513879, np.nan, np.nan, np.nan, np.nan, np.nan],     # Handle NaN
    [10.675035, 12.010602, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]      # Handle NaN
])
"""
2D array storing peak GPU memory usage in Gigabytes (PeakMemory_GB from CSV) for an entire batch.
Rows correspond to `IMG_SIZE_BUCKETS` indices (based on Width).
Columns correspond to `BATCH_SIZE_OPTIONS` indices.
Values are averaged over different 'Height' if multiple exist for the same Width-BatchSize pair.
IMPORTANT: NaN values must be replaced with sensible defaults (e.g., a very large number if it means it won't fit,
or an extrapolated value).
"""

# --- Action Space Configuration ---
N_PRIORITY_WEIGHTS: int = 3
"""
Number of continuous priority weights the actor learns. These weights are used by the
environment to score and prioritize requests in the queue based on their features
(e.g., internet speed, waiting time, token count).
"""

STRUCTURED_ACTION_FLAT_DIM: int = K_DIM_BUCKETS + M_BATCH_SIZE_OPTIONS + N_PRIORITY_WEIGHTS
"""
Total dimension of the flat action vector produced by the diffusion model (actor).
This vector is then parsed into discrete choices for image size and batch size,
and continuous values for priority weights.
"""

# --- Environment Configuration ---
MAX_REQUESTS_IN_QUEUE_OBS: int = 20
"""
Maximum number of photo editing requests visible to the agent in the 'request_queue'
part of its observation. The actual internal queue in the environment might be larger.
"""

REQUEST_FEATURE_SIZE: int = 4
"""
Number of features describing each request in the queue.
Typically: [token_count, internet_speed, img_size, waiting_time].
"""

MAX_STEPS_PER_EPISODE: int = 2000
"""
Maximum number of simulation steps before an episode is truncated (ended early).
"""

STEP_DURATION_SECONDS: float = 0.5
"""
Defines the duration of one environment step in real-world seconds.
This tell how much "simulated real-world time" has elapsed during that one step
where the agent made its decision and the environment reacted.
This is used to convert `Time_s` from the benchmark CSV (which is in seconds)
into the number of environment steps a task will take.
IMPORTANT: Adjust based on your simulation's desired time scale relative to CSV Time_s.
"""

# --- GPU Configuration ---
GPU_TOTAL_CAPACITY_GB: float = 24
"""
Total GPU memory capacity of the simulated server in Gigabytes.
Used in conjunction with `PEAK_MEMORY_GB_BY_INDEX` to determine if a batch can be processed.
IMPORTANT: Set this to your actual or desired simulated server GPU memory.
"""

# --- Observation Space Configuration ---
SERVER_STATE_DIM: int = 4
"""
Dimension of the 'server_state' vector in the observation.
Typically: [M_avail, G_avail_baseline, B_avail, N_connections_avail].
- M_avail: Available general memory (0-1).
- G_avail_baseline: Baseline GPU availability trend (0-1), influenced by external factors.
- B_avail: Available bandwidth (0-1).
- N_connections_avail: Number of available concurrent processing connections.
"""

FLATTENED_OBS_DIM: int = SERVER_STATE_DIM + (MAX_REQUESTS_IN_QUEUE_OBS * REQUEST_FEATURE_SIZE)
"""
Total dimension of the flattened observation vector when server_state and request_queue
are combined. This is often the input dimension for actor and critic networks.
"""

# --- Critic Network Configuration ---
PROCESSED_ACTION_DIM_FOR_CRITIC: int = STRUCTURED_ACTION_FLAT_DIM
"""
Dimension of the action vector after any processing (e.g., applying softmax to logits)
that is fed into the critic network along with the state.
Often the same as the actor's flat output dimension if no complex processing occurs before critic input.
"""

# --- Diffusion Model (Actor) Configuration ---
N_DIFFUSION_TIMESTEPS: int = 5
"""
Number of timesteps used in the diffusion process by the actor model
for generating or refining actions.
"""

MAX_ACTION_CONTINUOUS_PART: float = 1.0
"""
Maximum absolute value for the continuous parts of the action vector output by the actor
(e.g., priority weights are often scaled to be within [-1, 1] or [0, 1]).
"""

# --- Training Hyperparameters ---
LEARNING_RATE_ACTOR: float = 1e-4
"""Learning rate for the actor model's optimizer."""

LEARNING_RATE_CRITIC: float = 3e-4
"""Learning rate for the critic model's optimizer."""

GAMMA: float = 0.99
"""Discount factor for future rewards in reinforcement learning."""

TAU: float = 0.005
"""Coefficient for soft updates of target networks in actor-critic algorithms."""

BATCH_SIZE_TRAINING: int = 64
"""Number of experiences sampled from the replay buffer for each training step."""

EXPLORATION_NOISE_STD: float = 0.1
"""Standard deviation for Gaussian noise added to actions during exploration phase."""

# --- Profit Calculation Enhancements ---
MAX_WAIT_TIME_PENALTY_THRESHOLD: int = 50
"""
Maximum number of environment steps a request can wait in the queue before
a penalty is applied to the profit calculation for processing that request.
"""

WAIT_TIME_PENALTY_AMOUNT: float = 10.0
"""
The amount of penalty subtracted from profit if a request's waiting time
exceeds `MAX_WAIT_TIME_PENALTY_THRESHOLD`.
"""

BASE_PROFIT_PER_TOKEN: float = 0.1
"""Base profit factor associated with the token count of a request."""

BASE_PROFIT_PER_IMG_SIZE_UNIT: float = 0.001
"""Base profit factor associated with the image size of a request."""

INTERNET_SPEED_PROFIT_FACTOR: float = 0.5
"""Factor influencing how a request's internet speed affects its calculated profit."""

# --- Server State Update Enhancements (External Factors) ---
EXTERNAL_STATE_CHANGE_PROBABILITY: float = 0.1
"""Probability that an external factor will change the server state in any given step."""

GPU_AVAIL_EXTERNAL_CHANGE_MIN: float = -0.02
"""
Minimum change factor for the baseline GPU availability (`G_avail_baseline`)
due to external events. Can be negative (decrease).
"""
GPU_AVAIL_EXTERNAL_CHANGE_MAX: float = 0.1
"""
Maximum change factor for the baseline GPU availability (`G_avail_baseline`)
due to external events. Can be positive (increase/regeneration).
"""

RESOURCE_EXTERNAL_CHANGE_MIN: float = -0.05
"""
Minimum change factor for other server resources (e.g., `M_avail`, `B_avail`)
due to external events.
"""
RESOURCE_EXTERNAL_CHANGE_MAX: float = 0.05
"""
Maximum change factor for other server resources (e.g., `M_avail`, `B_avail`)
due to external events.
"""

EXTERNAL_CONNECTIONS_CHANGE_MIN: int = -2
"""Minimum change in the number of available server connections due to external events."""
EXTERNAL_CONNECTIONS_CHANGE_MAX: int = 2
"""Maximum change in the number of available server connections due to external events."""

# --- Network Architecture (Examples - uncomment and adjust if needed) ---
# ACTOR_MLP_UNET_INTERMEDIATE_DIMS: list[int] = [256, 128, 64, 32]
# """Example: Hidden dimensions for the U-Net encoder/decoder paths in MLPUNet actor."""

# CRITIC_HIDDEN_DIM: int = 256
# """Example: Hidden dimension for the critic's MLPs."""


BASE_DOWNLOAD_TIME_SECONDS = 1.0  # Example: Base time in seconds for download component of a task
INTERNET_SPEED_EPSILON = 1e-6    # Example: Small epsilon to prevent division by zero for internet speed
