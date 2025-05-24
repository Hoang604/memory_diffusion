"""
Main Training Script for Photo Editing Server RL Agent.

This script sets up and runs the training process for the DiffusionOPT policy
using the custom photo editing server environment. It leverages Tianshou for
the training loop, data collection, and logging.
"""
import os
import torch
import numpy as np
import time # For timestamping log paths

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
    args = {
        "seed": 42,
        "buffer_size": 100000,
        "actor_lr": config.LEARNING_RATE_ACTOR,
        "critic_lr": config.LEARNING_RATE_CRITIC,
        "gamma": config.GAMMA,
        "tau": config.TAU,
        "exploration_noise_std": config.EXPLORATION_NOISE_STD,
        "n_step": 3, 
        "batch_size": config.BATCH_SIZE_TRAINING,
        "epoch": 50, 
        "step_per_epoch": 5000, 
        "step_per_collect": 100, 
        "update_per_step": 1.0 / 10, 
        "num_train_envs": 4, 
        "num_test_envs": 2,  
        "episode_per_test": 10,
        "logdir": "logs",
        "experiment_name": "diffusion_opt_photo_edit",
        "device": 'cuda' if torch.cuda.is_available() else 'cpu',
        "resume_path": None, 
        "watch": False, 
        "bc_active": False, # Set to False to run without BC data initially
        "expert_data_path": None 
    }
    return type('Args', (object,), args)() 

def setup_training_components(args):
    """Initializes and returns the environment, policy, optimizers, etc."""
    def make_env():
        return Env(max_steps_per_episode=config.MAX_STEPS_PER_EPISODE)

    train_envs = DummyVectorEnv([make_env for _ in range(args.num_train_envs)])
    test_envs = DummyVectorEnv([make_env for _ in range(args.num_test_envs)])

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    actor_core_net = MLPUNet(
        input_dim=config.STRUCTURED_ACTION_FLAT_DIM,    # This should be 11 (for x_t_action)
        output_dim=config.STRUCTURED_ACTION_FLAT_DIM,   # Output dimension
        cond_obs_dim=config.FLATTENED_OBS_DIM,          # For the s_cond_observation (environment state)
        # You can also specify other MLPUNet parameters here:
        # intermediate_dims=[256, 128, 64], # Example
        # time_emb_dim=32,                  # Example
        # cond_obs_emb_dim=64,              # Example
        # dropout_rate=0.1                  # Example
    ).to(args.device)

    actor_diffusion_model = Diffusion(
        model=actor_core_net,
        beta_schedule='vp',
        n_timesteps=config.N_DIFFUSION_TIMESTEPS,
        action_dim=config.STRUCTURED_ACTION_FLAT_DIM,
        max_action_clamp=config.MAX_ACTION_CONTINUOUS_PART,
        bc_coef=args.bc_active 
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor_diffusion_model.parameters(), lr=args.actor_lr)

    critic_net = DoubleCritic(
        state_dim=config.FLATTENED_OBS_DIM
    ).to(args.device)
    critic_optim = torch.optim.Adam(critic_net.parameters(), lr=args.critic_lr)

    policy = DiffusionOPT(
        actor=actor_diffusion_model,
        actor_optim=actor_optim,
        critic=critic_net,
        critic_optim=critic_optim,
        device=args.device,
        tau=args.tau,
        gamma=args.gamma,
        exploration_noise_std=args.exploration_noise_std if not args.bc_active else 0.0, 
        n_step=args.n_step,
        bc_coef_in_actor_diffusion_model=args.bc_active,
        action_space=train_envs.action_space[0] 
    )

    replay_buffer = VectorReplayBuffer(
        total_size=args.buffer_size,
        buffer_num=args.num_train_envs 
    )

    if args.bc_active and args.expert_data_path:
        try:
            expert_data = torch.load(args.expert_data_path) 
            print(f"Attempting to load expert data from: {args.expert_data_path}")
            print(f"Expert data loading placeholder. Ensure your data format is compatible with ReplayBuffer and BC loss.")
        except Exception as e:
            print(f"Warning: Could not load expert data from {args.expert_data_path}. Error: {e}")

    train_collector = Collector(policy, train_envs, replay_buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs) 

    return policy, train_collector, test_collector, replay_buffer 

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
    )
    print(f"Starting training on device: {args.device}")
    trainer.run()

def watch_agent(args, policy):
    """Visualizes the trained agent playing in the environment."""
    env = Env() 
    policy.eval() 
    collector = Collector(policy, env)
    collector.collect(n_episode=5, render=1/35, reset_before_collect=True) # Added reset_before_collect here too for safety


if __name__ == '__main__':
    args = get_args()

    timestamp = time.strftime("%Y%m%d-%H%M%S") # Corrected 'время' to 'time'
    log_path = os.path.join(args.logdir, args.experiment_name, timestamp)
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer, train_interval=args.step_per_collect, update_interval=args.step_per_collect)

    print(f"Logging to: {log_path}")

    policy, train_collector, test_collector, replay_buffer = setup_training_components(args)

    if args.resume_path:
        try:
            checkpoint = torch.load(args.resume_path, map_location=args.device)
            policy.load_state_dict(checkpoint['model'])
            if 'actor_optim' in checkpoint and hasattr(policy, '_actor_optim') and policy._actor_optim is not None: 
                policy._actor_optim.load_state_dict(checkpoint['actor_optim'])
            if 'critic_optim' in checkpoint and hasattr(policy, '_critic_optim') and policy._critic_optim is not None: 
                policy._critic_optim.load_state_dict(checkpoint['critic_optim'])
            print(f"Resumed training from checkpoint: {args.resume_path}")
        except Exception as e:
            print(f"Could not load checkpoint from {args.resume_path}. Starting from scratch. Error: {e}")

    if not args.resume_path: 
        if args.bc_active and args.expert_data_path:
            print("BC active with expert data. Buffer pre-filling depends on expert data loading.")
            if len(replay_buffer) < args.step_per_collect : 
                 print(f"Replay buffer has {len(replay_buffer)} samples. Collecting initial samples...")
                 # Add reset_before_collect=True
                 train_collector.collect(n_step=args.step_per_collect * args.num_train_envs, reset_before_collect=True)
        else: 
            print("Collecting initial random/exploratory samples...")
            # Add reset_before_collect=True
            train_collector.collect(n_step=max(args.batch_size, args.step_per_collect) * args.num_train_envs, reset_before_collect=True)

    try:
        train_agent(args, policy, train_collector, test_collector, logger)
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        print("Training finished or interrupted.")
        final_policy_path = os.path.join(log_path, 'final_policy.pth')
        save_data = {'model': policy.state_dict()}
        if hasattr(policy, '_actor_optim') and policy._actor_optim is not None:
            save_data['actor_optim'] = policy._actor_optim.state_dict()
        if hasattr(policy, '_critic_optim') and policy._critic_optim is not None:
            save_data['critic_optim'] = policy._critic_optim.state_dict()
        
        torch.save(save_data, final_policy_path)
        print(f"Final policy saved to {final_policy_path}")

    if args.watch:
        print("\nWatching trained agent...")
        watch_agent(args, policy)

    print(f"\nExperiment complete. Logs saved in: {log_path}")
