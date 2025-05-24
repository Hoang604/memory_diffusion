"""
Custom Gymnasium Environment for simulating a photo editing server.

The environment simulates a server that receives photo editing requests and
schedules them for processing on a GPU. The agent's goal is to learn a
scheduling policy that maximizes profit.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import collections

import config # Import shared configurations

class Env(gym.Env):
    """
    Photo Editing Server Simulation Environment.

    This environment models a server that processes photo editing requests.
    The agent (a reinforcement learning policy) observes the server state and
    the queue of pending requests, and then decides which requests to process next
    by outputting parameters for batch formation.

    Attributes:
        action_space (gym.spaces.Dict): Defines the structure of the action
            (parameters for batch selection) the agent can take.
        observation_space (gym.spaces.Dict): Defines the structure of the
            observation (server state and request queue) the agent receives.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, max_steps_per_episode: int = config.MAX_STEPS_PER_EPISODE):
        """
        Initializes the environment.

        Args:
            max_steps_per_episode: The maximum number of steps before an episode is truncated.
        """
        super().__init__()

        self.max_steps_per_episode = max_steps_per_episode
        self.current_step = 0

        self._observation_space = spaces.Dict({
            'server_state': spaces.Box(
                low=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0, 20.0], dtype=np.float32), # Max N_connections_avail
                shape=(config.SERVER_STATE_DIM,),
                dtype=np.float32
            ),
            'request_queue': spaces.Box(
                low=np.zeros((config.MAX_REQUESTS_IN_QUEUE_OBS, config.REQUEST_FEATURE_SIZE), dtype=np.float32),
                high=np.array(
                    [[200.0, 1.0, 1024.0, 1000.0]] * config.MAX_REQUESTS_IN_QUEUE_OBS, # Max token, speed, img_size, waiting_time
                    dtype=np.float32
                ),
                dtype=np.float32
            )
        })

        self.action_space = spaces.Dict({
            "dim_bucket_logits": spaces.Box(
                low=-np.inf, high=np.inf, shape=(config.K_DIM_BUCKETS,), dtype=np.float32
            ),
            "batch_size_logits": spaces.Box(
                low=-np.inf, high=np.inf, shape=(config.M_BATCH_SIZE_OPTIONS,), dtype=np.float32
            ),
            "priority_weights": spaces.Box(
                low=-config.MAX_ACTION_CONTINUOUS_PART,
                high=config.MAX_ACTION_CONTINUOUS_PART,
                shape=(config.N_PRIORITY_WEIGHTS,),
                dtype=np.float32
            )
        })

        self.actual_request_queue = collections.deque(maxlen=config.MAX_REQUESTS_IN_QUEUE_OBS * 2)
        self.next_request_id = 0
        self._server_state = np.zeros(config.SERVER_STATE_DIM, dtype=np.float32)

    def _get_img_size_from_bucket_index(self, index: int) -> float:
        """Maps a dimension bucket index to an actual image size."""
        if 0 <= index < len(config.IMG_SIZE_BUCKETS):
            return config.IMG_SIZE_BUCKETS[index]
        # Fallback or error for invalid index
        print(f"Warning: Invalid dimension bucket index {index}, using smallest.")
        return config.IMG_SIZE_BUCKETS[0]


    def _get_batch_size_from_option_index(self, index: int) -> int:
        """Maps a batch size option index to an actual batch size."""
        if 0 <= index < len(config.BATCH_SIZE_OPTIONS):
            return int(config.BATCH_SIZE_OPTIONS[index])
        # Fallback or error for invalid index
        print(f"Warning: Invalid batch size option index {index}, using smallest.")
        return int(config.BATCH_SIZE_OPTIONS[0])

    def _generate_initial_server_state(self) -> None:
        """Generates an initial random state for server resources."""
        self._server_state = np.array([
            self.np_random.random(),
            self.np_random.random(),
            self.np_random.random(),
            float(self.np_random.integers(5, config.MAX_REQUESTS_IN_QUEUE_OBS + 1))
        ], dtype=np.float32)

    def _generate_new_requests(self, num_requests: int = 1) -> None:
        """Generates new incoming requests and adds them to the actual queue."""
        possible_token_count = np.arange(10, 201, 10)
        for _ in range(num_requests):
            if len(self.actual_request_queue) < self.actual_request_queue.maxlen:
                request = {
                    "id": self.next_request_id,
                    "token_count": float(self.np_random.choice(possible_token_count)),
                    "internet_speed": self.np_random.random(),
                    "img_size": float(self.np_random.choice(config.IMG_SIZE_BUCKETS)), # New requests can have any valid size
                    "arrival_time": self.current_step
                }
                self.actual_request_queue.append(request)
                self.next_request_id += 1

    def _get_observation(self) -> dict:
        """Constructs the observation dictionary from the internal state."""
        obs_request_queue = np.zeros((config.MAX_REQUESTS_IN_QUEUE_OBS, config.REQUEST_FEATURE_SIZE), dtype=np.float32)
        for i, req in enumerate(list(self.actual_request_queue)[:config.MAX_REQUESTS_IN_QUEUE_OBS]):
            waiting_time = float(self.current_step - req["arrival_time"])
            obs_request_queue[i, 0] = req["token_count"]
            obs_request_queue[i, 1] = req["internet_speed"]
            obs_request_queue[i, 2] = req["img_size"]
            obs_request_queue[i, 3] = waiting_time
        return {
            'server_state': self._server_state.copy(),
            'request_queue': obs_request_queue
        }

    def _calculate_profit(self, chosen_batch: list) -> float:
        """
        Calculates the profit for a batch of processed requests.
        This is a placeholder; the actual profit function f(x) needs to be defined.
        """
        if not chosen_batch:
            return 0.0
        profit = 0.0
        for req in chosen_batch:
            base_profit = req["token_count"] * 0.1 + req["img_size"] * 0.001 # Reduced img_size impact
            speed_factor = req["internet_speed"] + 1e-6
            profit += base_profit * speed_factor
        return profit

    def _update_server_state_after_processing(self, chosen_batch: list) -> None:
        """
        Updates server resources after processing a batch. Simplified model.
        """
        if not chosen_batch:
            return
        gpu_mem_consumed = len(chosen_batch) * 0.02 # Reduced consumption
        gpu_power_consumed = len(chosen_batch) * 0.05 # Reduced consumption

        self._server_state[0] = np.clip(self._server_state[0] - gpu_mem_consumed, 0.0, 1.0)
        self._server_state[1] = np.clip(self._server_state[1] - gpu_power_consumed, 0.0, 1.0)
        self._server_state[0] = np.clip(self._server_state[0] + 0.01, 0.0, 1.0) # Reduced recovery
        self._server_state[1] = np.clip(self._server_state[1] + 0.02, 0.0, 1.0) # Reduced recovery
        # self._server_state[3] (N_avail) could be reduced by len(chosen_batch)
        # and recover over time, but this adds complexity not modeled here.

    def step(self, action: dict) -> tuple[dict, float, bool, bool, dict]:
        """
        Executes one time step within the environment.

        The agent takes an action (parameters for batch selection), and the
        environment transitions to a new state, returning the observation,
        reward, and termination/truncation signals.

        Args:
            action: A dictionary matching `self.action_space` structure.

        Returns:
            A tuple (next_observation, reward, terminated, truncated, info).
        """
        self.current_step += 1

        dim_bucket_logits = action["dim_bucket_logits"]
        batch_size_logits = action["batch_size_logits"]
        priority_weights = action["priority_weights"]

        chosen_dim_bucket_index = int(np.argmax(dim_bucket_logits))
        chosen_batch_size_option_index = int(np.argmax(batch_size_logits))

        target_img_size = self._get_img_size_from_bucket_index(chosen_dim_bucket_index)
        target_batch_size = self._get_batch_size_from_option_index(chosen_batch_size_option_index)

        eligible_requests = [req for req in self.actual_request_queue if req["img_size"] == target_img_size]
        
        scored_requests = []
        if eligible_requests:
            # Normalize features for scoring over the current eligible set
            speeds = np.array([req["internet_speed"] for req in eligible_requests], dtype=np.float32)
            waiting_times = np.array([self.current_step - req["arrival_time"] for req in eligible_requests], dtype=np.float32)
            profit_metrics = np.array([req["token_count"] for req in eligible_requests], dtype=np.float32) # Example metric

            # Handle cases with single eligible request to avoid NaN from std=0
            norm_speeds = (speeds - np.mean(speeds)) / (np.std(speeds) + 1e-6) if len(speeds) > 1 else np.zeros_like(speeds)
            norm_waiting_times = (waiting_times - np.mean(waiting_times)) / (np.std(waiting_times) + 1e-6) if len(waiting_times) > 1 else np.zeros_like(waiting_times)
            norm_profit_metrics = (profit_metrics - np.mean(profit_metrics)) / (np.std(profit_metrics) + 1e-6) if len(profit_metrics) > 1 else np.zeros_like(profit_metrics)

            for i, req in enumerate(eligible_requests):
                score = (priority_weights[0] * norm_speeds[i] +
                         priority_weights[1] * norm_waiting_times[i] +
                         priority_weights[2] * norm_profit_metrics[i])
                scored_requests.append((score, req))
        
        scored_requests.sort(key=lambda x: x[0], reverse=True)
        
        chosen_batch = []
        # Max N_avail is server_state[3]
        max_concurrent_processing = min(target_batch_size, int(self._server_state[3]))
        for _, req in scored_requests:
            if len(chosen_batch) < max_concurrent_processing:
                chosen_batch.append(req)
            else:
                break
        
        reward = self._calculate_profit(chosen_batch)

        processed_request_ids = {req["id"] for req in chosen_batch}
        self.actual_request_queue = collections.deque(
            [req for req in self.actual_request_queue if req["id"] not in processed_request_ids],
            maxlen=self.actual_request_queue.maxlen
        )

        self._update_server_state_after_processing(chosen_batch)
        self._generate_new_requests(num_requests=self.np_random.integers(0, 3))

        next_observation = self._get_observation()
        terminated = False
        truncated = self.current_step >= self.max_steps_per_episode
        info = {"chosen_batch_size": len(chosen_batch), "processed_ids": [req["id"] for req in chosen_batch]}

        return next_observation, float(reward), terminated, truncated, info

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """
        Resets the environment to an initial state.

        Args:
            seed: Optional seed for the random number generator.
            options: Optional dictionary of environment-specific options.

        Returns:
            A tuple (initial_observation, info).
        """
        super().reset(seed=seed)
        self.current_step = 0
        self.next_request_id = 0

        self._generate_initial_server_state()
        self.actual_request_queue.clear()
        self._generate_new_requests(num_requests=self.np_random.integers(
            min(5, config.MAX_REQUESTS_IN_QUEUE_OBS // 2),
            config.MAX_REQUESTS_IN_QUEUE_OBS // 2 + 1
        ))

        initial_observation = self._get_observation()
        info = {"message": "Environment reset."}
        
        return initial_observation, info

    @property
    def observation_space(self) -> gym.spaces.Space:
        """Returns the observation space of the environment."""
        return self._observation_space
    
    @property
    def action_space(self) -> gym.spaces.Space:
        """Returns the action space of the environment."""
        # Overwrite to ensure it's returned correctly if modified after super().__init__
        action_s = spaces.Dict({
            "dim_bucket_logits": spaces.Box(
                low=-np.inf, high=np.inf, shape=(config.K_DIM_BUCKETS,), dtype=np.float32
            ),
            "batch_size_logits": spaces.Box(
                low=-np.inf, high=np.inf, shape=(config.M_BATCH_SIZE_OPTIONS,), dtype=np.float32
            ),
            "priority_weights": spaces.Box(
                low=-config.MAX_ACTION_CONTINUOUS_PART,
                high=config.MAX_ACTION_CONTINUOUS_PART,
                shape=(config.N_PRIORITY_WEIGHTS,),
                dtype=np.float32
            )
        })
        return action_s

if __name__ == '__main__':
    # from gymnasium.utils.env_checker import check_env # Optional: for deeper validation
    env = Env()
    # check_env(env) # Uncomment to run Gymnasium's environment checker

    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)

    obs, info = env.reset(seed=42)
    print("\nInitial Observation (Server State):", obs['server_state'])
    print("Initial Observation (Request Queue Sample - first 3):")
    print(obs['request_queue'][:3])

    total_reward_test = 0
    for test_step in range(config.MAX_STEPS_PER_EPISODE):
        random_action = env.action_space.sample()
        next_obs, reward_test, terminated_test, truncated_test, info_test = env.step(random_action)
        total_reward_test += reward_test
        if (test_step + 1) % 50 == 0:
            print(f"Step {test_step + 1}: Reward {reward_test:.2f}, Chosen Batch Size {info_test.get('chosen_batch_size', 0)}")
        if terminated_test or truncated_test:
            print(f"\nEpisode finished after {test_step + 1} steps. Total reward: {total_reward_test:.2f}")
            break
    if not (terminated_test or truncated_test):
         print(f"\nEpisode finished after {config.MAX_STEPS_PER_EPISODE} steps. Total reward: {total_reward_test:.2f}")
    env.close()
