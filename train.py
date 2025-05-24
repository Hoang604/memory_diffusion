"""
Main Training Script for Photo Editing Server RL Agent.

This script sets up and runs the training process for the DiffusionOPT policy
using the custom photo editing server environment. It leverages Tianshou for
the training loop, data collection, and logging.
"""
import os
import torch
import numpy as np
import time

# Import configurations and custom modules
import config
from env import Env # From env/__init__.py
from net import MLPUNet, DoubleCritic # From net/__init__.py
from diffusion import Diffusion # From diffusion/__init__.py
from policy import DiffusionOPT, DiffusionOPTStats # From policy/__init__.py

# Tianshou imports
from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter


def get_args():
    """Parses command line arguments or returns default training parameters."""
    # For simplicity, we'll use defaults from config.py, but you can add argparse here.
    args = {
        "seed": 42,
        "buffer_size": 100000,
        "actor_lr": config.LEARNING_RATE_ACTOR,
        "critic_lr": config.LEARNING_RATE_CRITIC,
        "gamma": config.GAMMA,
        "tau": config.TAU,
        "exploration_noise_std": config.EXPLORATION_NOISE_STD,
        "n_step": 3, # N-step returns
        "batch_size": config.BATCH_SIZE_TRAINING,
        "epoch": 50, # Total training epochs
        "step_per_epoch": 5000, # Env steps per epoch
        "step_per_collect": 100, # Env steps before one training update batch
        "update_per_step": 1.0 / 10, # Train actor/critic every 10 env steps (0.1 updates per step)
        "num_train_envs": 4, # Number of parallel training environments
        "num_test_envs": 2,  # Number of parallel testing environments
        "episode_per_test": 10,
        "logdir": "logs",
        "experiment_name": "diffusion_opt_photo_edit",
        "device": 'cuda' if torch.cuda.is_available() else 'cpu',
        "resume_path": None, # Path to checkpoint for resuming training
        "watch": False, # Set to True to watch the agent play after training
        # Behavior Cloning related
        "bc_active": False, # Set to True if you want to use Behavior Cloning for diffusion actor
        "expert_data_path": None # Path to your expert data (e.g., a .pth file of (obs, expert_action_flat) tuples)
    }
    return type('Args', (object,), args)() # Convert dict to object for dot notation

def setup_training_components(args):
    """Initializes and returns the environment, policy, optimizers, etc."""
    # --- Environment ---
    def make_env():
        return Env(max_steps_per_episode=config.MAX_STEPS_PER_EPISODE)

    train_envs = DummyVectorEnv([make_env for _ in range(args.num_train_envs)])
    test_envs = DummyVectorEnv([make_env for _ in range(args.num_test_envs)])

    # Seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # --- Networks ---
    actor_core_net = MLPUNet(
        input_dim=config.FLATTENED_OBS_DIM,
        output_dim=config.STRUCTURED_ACTION_FLAT_DIM
        # intermediate_dims=config.ACTOR_MLP_UNET_INTERMEDIATE_DIMS (if defined in config)
    ).to(args.device)

    actor_diffusion_model = Diffusion(
        model=actor_core_net,
        beta_schedule='vp',
        n_timesteps=config.N_DIFFUSION_TIMESTEPS,
        action_dim=config.STRUCTURED_ACTION_FLAT_DIM,
        max_action_clamp=config.MAX_ACTION_CONTINUOUS_PART,
        bc_coef=args.bc_active # Critical: True if training actor with BC loss
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor_diffusion_model.parameters(), lr=args.actor_lr)

    critic_net = DoubleCritic(
        state_dim=config.FLATTENED_OBS_DIM
        # hidden_dim=config.CRITIC_HIDDEN_DIM (if defined in config)
    ).to(args.device)
    critic_optim = torch.optim.Adam(critic_net.parameters(), lr=args.critic_lr)

    # --- Policy ---
    policy = DiffusionOPT(
        actor=actor_diffusion_model,
        actor_optim=actor_optim,
        critic=critic_net,
        critic_optim=critic_optim,
        device=args.device,
        tau=args.tau,
        gamma=args.gamma,
        exploration_noise_std=args.exploration_noise_std if not args.bc_active else 0.0, # Less noise if primarily BC
        n_step=args.n_step,
        bc_coef_in_actor_diffusion_model=args.bc_active,
        # Pass dummy action_space for Tianshou BasePolicy compatibility if needed.
        # The actual action structure is handled internally by DiffusionOPT.
        action_space=train_envs.action_space[0] # Get action space from one of the envs
    )

    # --- Replay Buffer ---
    # If using prioritized replay or other buffer types, instantiate them here.
    # VectorReplayBuffer is for parallel environments.
    replay_buffer = VectorReplayBuffer(
        total_size=args.buffer_size,
        buffer_num=args.num_train_envs # One buffer per parallel training environment
    )

    # Load expert data if BC is active and path is provided
    if args.bc_active and args.expert_data_path:
        try:
            expert_data = torch.load(args.expert_data_path) # Expects a list of Batch or similar
            # You'll need to populate the buffer with this data.
            # Tianshou's ReplayBuffer expects Batch objects with obs, act, rew, done, obs_next, info.
            # For BC, 'info' should contain 'expert_action_flat'.
            # 'act' can be the expert_action_flat or a dummy action if not used by BC loss directly.
            print(f"Attempting to load expert data from: {args.expert_data_path}")
            # Example: Assuming expert_data is a list of Batch objects
            # for expert_batch in expert_data:
            #     replay_buffer.add(expert_batch)
            # This part requires a specific format for your expert data.
            # For now, this is a placeholder.
            print(f"Expert data loading placeholder. Ensure your data format is compatible with ReplayBuffer and BC loss.")
        except Exception as e:
            print(f"Warning: Could not load expert data from {args.expert_data_path}. Error: {e}")


    # --- Collectors ---
    train_collector = Collector(policy, train_envs, replay_buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs) # No exploration noise for evaluation

    return policy, train_collector, test_collector, replay_buffer # Return buffer for potential pre-filling

def train_agent(args, policy, train_collector, test_collector, logger):
    """Sets up and runs the Tianshou OffpolicyTrainer."""

    trainer = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        episode_per_test=args.episode_per_test,
        batch_size=args.batch_size,
        update_per_step=args.update_per_step,
        logger=logger,
        # test_in_train=False, # Set to False to run tests only after each epoch
        # save_best_fn=save_best_policy_fn, # Optional: function to save the best policy
        # save_checkpoint_fn=save_checkpoint_fn, # Optional: function to save checkpoints
    )
    print(f"Starting training on device: {args.device}")
    trainer.run()

def watch_agent(args, policy):
    """Visualizes the trained agent playing in the environment."""
    env = Env() # Single environment for watching
    policy.eval() # Set policy to evaluation mode
    collector = Collector(policy, env)
    collector.collect(n_episode=5, render=1/35) # Render with a delay


if __name__ == '__main__':
    args = get_args()

    # --- Logger Setup ---
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(args.logdir, args.experiment_name, timestamp)
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer, train_interval=args.step_per_collect, update_interval=args.step_per_collect)

    print(f"Logging to: {log_path}")

    policy, train_collector, test_collector, replay_buffer = setup_training_components(args)

    if args.resume_path:
        try:
            checkpoint = torch.load(args.resume_path, map_location=args.device)
            policy.load_state_dict(checkpoint['model'])
            if 'actor_optim' in checkpoint: policy._actor_optim.load_state_dict(checkpoint['actor_optim'])
            if 'critic_optim' in checkpoint: policy._critic_optim.load_state_dict(checkpoint['critic_optim'])
            print(f"Resumed training from checkpoint: {args.resume_path}")
        except Exception as e:
            print(f"Could not load checkpoint from {args.resume_path}. Starting from scratch. Error: {e}")


    # --- Pre-fill buffer with some initial random exploration or expert data ---
    if not args.resume_path: # Only collect initial samples if not resuming
        if args.bc_active and args.expert_data_path:
            # Logic for pre-filling with expert data should ensure buffer is populated correctly.
            # For now, we assume if expert_data_path is used, setup_training_components handles it.
            # If not, you might want to collect some initial data using the policy.
            print("BC active with expert data. Buffer pre-filling depends on expert data loading.")
            if len(replay_buffer) < args.step_per_collect : # Or some other threshold
                 print(f"Replay buffer has {len(replay_buffer)} samples. Collecting initial samples...")
                 train_collector.collect(n_step=args.step_per_collect * args.num_train_envs)

        else: # Collect initial random/exploratory data
            print("Collecting initial random/exploratory samples...")
            # Collect at least `batch_size` or `step_per_collect` data before starting training.
            # Tianshou trainer usually handles initial collection if buffer is too small.
            train_collector.collect(n_step=max(args.batch_size, args.step_per_collect) * args.num_train_envs)


    # --- Start Training ---
    try:
        train_agent(args, policy, train_collector, test_collector, logger)
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        print("Training finished or interrupted.")
        # You might want to save the final policy here.
        # torch.save({
        #     'model': policy.state_dict(),
        #     'actor_optim': policy._actor_optim.state_dict(),
        #     'critic_optim': policy._critic_optim.state_dict(),
        # }, os.path.join(log_path, 'final_policy.pth'))
        # print(f"Final policy saved to {os.path.join(log_path, 'final_policy.pth')}")

    # --- Watch Agent (Optional) ---
    if args.watch:
        print("\nWatching trained agent...")
        watch_agent(args, policy)

    print(f"\nExperiment complete. Logs saved in: {log_path}")
