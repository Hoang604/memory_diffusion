"""
Custom Gymnasium Environment for simulating a photo editing server.
- Resources (GPU memory) are consumed by a batch for its processing duration.
- Processing duration is based on CSV data (Time_s).
- Resources are freed after the task completes.
- External factors still influence baseline resource availability.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import collections
import math # For math.ceil

import config

class Env(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, max_steps_per_episode: int = config.MAX_STEPS_PER_EPISODE):
        super().__init__()
        self.max_steps_per_episode = max_steps_per_episode
        self.current_step = 0

        # server_state[1] is now G_avail_baseline (general availability trend)
        self._observation_space = spaces.Dict({
            'server_state': spaces.Box(
                low=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), # M, G_baseline, B, Connections
                high=np.array([1.0, 1.0, 1.0, config.MAX_REQUESTS_IN_QUEUE_OBS * 2.0], dtype=np.float32),
                shape=(config.SERVER_STATE_DIM,),
                dtype=np.float32
            ),
            'request_queue': spaces.Box(
                low=np.zeros((config.MAX_REQUESTS_IN_QUEUE_OBS, config.REQUEST_FEATURE_SIZE), dtype=np.float32),
                high=np.array(
                    [[200.0, 1.0, 1024.0, float(config.MAX_STEPS_PER_EPISODE)]] * config.MAX_REQUESTS_IN_QUEUE_OBS,
                    dtype=np.float32
                ),
                dtype=np.float32
            )
        })

        self.action_space = spaces.Dict({
            "dim_bucket_logits": spaces.Box(low=-np.inf, high=np.inf, shape=(config.K_DIM_BUCKETS,), dtype=np.float32),
            "batch_size_logits": spaces.Box(low=-np.inf, high=np.inf, shape=(config.M_BATCH_SIZE_OPTIONS,), dtype=np.float32),
            "priority_weights": spaces.Box(low=-config.MAX_ACTION_CONTINUOUS_PART, high=config.MAX_ACTION_CONTINUOUS_PART, shape=(config.N_PRIORITY_WEIGHTS,), dtype=np.float32)
        })

        self.actual_request_queue = collections.deque(maxlen=config.MAX_REQUESTS_IN_QUEUE_OBS * 2)
        self.next_request_id = 0
        self._server_state = np.zeros(config.SERVER_STATE_DIM, dtype=np.float32)
        
        # Tracks ongoing tasks: {'batch_id': int, 'gpu_gb_reserved': float, 'completion_step': int}
        self.active_processing_tasks = []
        self.next_batch_id_counter = 0


    def _get_img_size_from_bucket_index(self, index: int) -> float:
        # Ensure config.IMG_SIZE_BUCKETS is populated correctly from your CSV via the script
        if 0 <= index < len(config.IMG_SIZE_BUCKETS):
            return config.IMG_SIZE_BUCKETS[index]
        print(f"Warning: Invalid dimension bucket index {index} (max: {len(config.IMG_SIZE_BUCKETS)-1}), using smallest.")
        return config.IMG_SIZE_BUCKETS[0]

    def _get_batch_size_from_option_index(self, index: int) -> int:
        # Ensure config.BATCH_SIZE_OPTIONS is populated correctly
        if 0 <= index < len(config.BATCH_SIZE_OPTIONS):
            return int(config.BATCH_SIZE_OPTIONS[index])
        print(f"Warning: Invalid batch size option index {index} (max: {len(config.BATCH_SIZE_OPTIONS)-1}), using smallest.")
        return int(config.BATCH_SIZE_OPTIONS[0])

    def _get_batch_processing_duration_steps(self, img_size_bucket_idx: int, batch_size_option_idx: int) -> int:
        """Gets processing time in seconds from config and converts to steps."""
        valid_img_idx = np.clip(img_size_bucket_idx, 0, config.PROCESSING_TIMES_BY_INDEX.shape[0] - 1)
        valid_batch_idx = np.clip(batch_size_option_idx, 0, config.PROCESSING_TIMES_BY_INDEX.shape[1] - 1)
        
        time_seconds = config.PROCESSING_TIMES_BY_INDEX[valid_img_idx, valid_batch_idx]
        if np.isnan(time_seconds):
            print(f"Warning: NaN processing time for img_idx {valid_img_idx}, batch_idx {valid_batch_idx}. Using a large default.")
            time_seconds = float(self.max_steps_per_episode * config.STEP_DURATION_SECONDS) # Default to very long
            
        if config.STEP_DURATION_SECONDS <= 1e-6:
            return int(self.max_steps_per_episode) # Avoid division by zero, effectively task never finishes
            
        duration_steps = math.ceil(time_seconds / config.STEP_DURATION_SECONDS)
        return max(1, duration_steps) # Task takes at least 1 step

    def _get_batch_gpu_consumption_gb(self, img_size_bucket_idx: int, batch_size_option_idx: int) -> float:
        """Gets peak GPU memory for the batch configuration from config."""
        valid_img_idx = np.clip(img_size_bucket_idx, 0, config.PEAK_MEMORY_GB_BY_INDEX.shape[0] - 1)
        valid_batch_idx = np.clip(batch_size_option_idx, 0, config.PEAK_MEMORY_GB_BY_INDEX.shape[1] - 1)
        
        gpu_gb = config.PEAK_MEMORY_GB_BY_INDEX[valid_img_idx, valid_batch_idx]
        if np.isnan(gpu_gb):
            print(f"Warning: NaN peak memory for img_idx {valid_img_idx}, batch_idx {valid_batch_idx}. Using 0 or default.")
            return 0.0 # Or a default value if appropriate
        return gpu_gb

    def _generate_initial_server_state(self) -> None:
        self._server_state[0] = self.np_random.random()  # M_avail
        self._server_state[1] = 1.0  # G_avail_baseline - start with full general GPU availability
        self._server_state[2] = self.np_random.random()  # B_avail
        self._server_state[3] = float(self.np_random.integers(5, config.MAX_REQUESTS_IN_QUEUE_OBS + 1))

    def _generate_new_requests(self, num_requests: int = 1) -> None:
        possible_token_count = np.arange(10, 201, 10)
        for _ in range(num_requests):
            if len(self.actual_request_queue) < self.actual_request_queue.maxlen:
                request = {
                    "id": self.next_request_id,
                    "token_count": float(self.np_random.choice(possible_token_count)),
                    "internet_speed": self.np_random.random(),
                    "img_size": float(self.np_random.choice(config.IMG_SIZE_BUCKETS)), # Assumes IMG_SIZE_BUCKETS is populated
                    "arrival_time": self.current_step
                }
                self.actual_request_queue.append(request)
                self.next_request_id += 1

    def _get_observation(self) -> dict:
        obs_request_queue = np.zeros((config.MAX_REQUESTS_IN_QUEUE_OBS, config.REQUEST_FEATURE_SIZE), dtype=np.float32)
        current_queue_list = list(self.actual_request_queue)
        for i, req in enumerate(current_queue_list[:config.MAX_REQUESTS_IN_QUEUE_OBS]):
            waiting_time = float(self.current_step - req["arrival_time"])
            obs_request_queue[i, 0] = req["token_count"]
            obs_request_queue[i, 1] = req["internet_speed"]
            obs_request_queue[i, 2] = req["img_size"]
            obs_request_queue[i, 3] = waiting_time
        return {'server_state': self._server_state.copy(), 'request_queue': obs_request_queue}

    def _calculate_profit(self, chosen_batch: list, chosen_dim_bucket_index: int, chosen_batch_size_option_index: int) -> float:
        if not chosen_batch:
            return 0.0
        total_base_profit = 0.0
        total_penalty = 0.0
        for req in chosen_batch:
            base_profit_for_req = (req["token_count"] * config.BASE_PROFIT_PER_TOKEN +
                                   req["img_size"] * config.BASE_PROFIT_PER_IMG_SIZE_UNIT)
            speed_multiplier = 1.0 + (req["internet_speed"] * config.INTERNET_SPEED_PROFIT_FACTOR)
            total_base_profit += base_profit_for_req * speed_multiplier
            waiting_time = self.current_step - req["arrival_time"] # Waiting time until start of processing
            if waiting_time > config.MAX_WAIT_TIME_PENALTY_THRESHOLD:
                total_penalty += config.WAIT_TIME_PENALTY_AMOUNT
        net_profit = total_base_profit - total_penalty
        
        # Profit per unit time is based on the *scheduled* processing time of the batch type
        # not the actual time it might take if environment steps are coarse.
        # We use the same _get_batch_processing_duration_steps but convert back to seconds for consistency with Time_s
        processing_time_seconds = self._get_batch_processing_duration_steps(chosen_dim_bucket_index, chosen_batch_size_option_index) * config.STEP_DURATION_SECONDS
        
        return net_profit / processing_time_seconds if processing_time_seconds > 1e-6 else net_profit


    def _update_active_tasks_and_server_state(self) -> None:
        """Frees resources from completed tasks and applies external changes to baseline state."""
        # Free resources from completed tasks
        completed_task_indices = []
        for i, task in enumerate(self.active_processing_tasks):
            if task['completion_step'] <= self.current_step:
                completed_task_indices.append(i)
                # print(f"Debug: Task {task['batch_id']} completed at step {self.current_step}. Freed {task['gpu_gb_reserved']} GB GPU.")

        # Remove completed tasks by iterating in reverse to maintain indices
        for i in sorted(completed_task_indices, reverse=True):
            del self.active_processing_tasks[i]

        # Apply external changes to the baseline server state
        if self.np_random.random() < config.EXTERNAL_STATE_CHANGE_PROBABILITY:
            for i in [0, 2]: # M_avail, B_avail
                change = self.np_random.uniform(config.RESOURCE_EXTERNAL_CHANGE_MIN, config.RESOURCE_EXTERNAL_CHANGE_MAX)
                self._server_state[i] = np.clip(self._server_state[i] + change, 0.0, 1.0)
            
            gpu_change = self.np_random.uniform(config.GPU_AVAIL_EXTERNAL_CHANGE_MIN, config.GPU_AVAIL_EXTERNAL_CHANGE_MAX)
            self._server_state[1] = np.clip(self._server_state[1] + gpu_change, 0.0, 1.0) # G_avail_baseline
            
            conn_change = self.np_random.integers(config.EXTERNAL_CONNECTIONS_CHANGE_MIN, config.EXTERNAL_CONNECTIONS_CHANGE_MAX + 1)
            self._server_state[3] = np.clip(self._server_state[3] + conn_change, 1.0, config.MAX_REQUESTS_IN_QUEUE_OBS * 2.0)


    def step(self, action: dict) -> tuple[dict, float, bool, bool, dict]:
        self.current_step += 1

        # 1. Update active tasks (free resources from completed ones) & apply external changes to baseline state
        self._update_active_tasks_and_server_state()

        # 2. Determine agent's choice
        dim_bucket_logits = action["dim_bucket_logits"]
        batch_size_logits = action["batch_size_logits"]
        priority_weights = action["priority_weights"]

        chosen_dim_bucket_index = int(np.argmax(dim_bucket_logits))
        chosen_batch_size_option_index = int(np.argmax(batch_size_logits))

        target_img_size = self._get_img_size_from_bucket_index(chosen_dim_bucket_index)
        target_batch_size = self._get_batch_size_from_option_index(chosen_batch_size_option_index)

        # 3. Calculate currently available GPU resources for THIS STEP
        gpu_reserved_by_active_tasks_gb = sum(task['gpu_gb_reserved'] for task in self.active_processing_tasks)
        baseline_gpu_available_gb = self._server_state[1] * config.GPU_TOTAL_CAPACITY_GB
        currently_usable_gpu_gb = max(0, baseline_gpu_available_gb - gpu_reserved_by_active_tasks_gb)
        
        current_step_connections_budget = int(self._server_state[3]) # Connections are not time-locked in this model, just a limit per step.
        
        chosen_batch = []
        processed_request_ids_in_step = set()
        gpu_gb_consumed_by_new_batch = 0.0

        # 4. Batch Formation (Agent's Choice Only)
        # Check if the *intended batch type* can be processed
        intended_batch_gpu_cost_gb = self._get_batch_gpu_consumption_gb(chosen_dim_bucket_index, chosen_batch_size_option_index)
        
        if target_batch_size > 0 and \
           intended_batch_gpu_cost_gb <= currently_usable_gpu_gb and \
           target_batch_size <= current_step_connections_budget:
            
            # If the batch type is feasible, try to fill it with eligible requests
            eligible_requests = [req for req in self.actual_request_queue if req["img_size"] == target_img_size]
            scored_requests = []
            if eligible_requests:
                # (Your existing scoring logic here)
                speeds = np.array([req["internet_speed"] for req in eligible_requests], dtype=np.float32)
                current_waiting_times = np.array([self.current_step - req["arrival_time"] for req in eligible_requests], dtype=np.float32)
                token_counts = np.array([req["token_count"] for req in eligible_requests], dtype=np.float32)
                norm_speeds = (speeds - np.mean(speeds)) / (np.std(speeds) + 1e-6) if len(speeds) > 1 else np.zeros_like(speeds)
                norm_waiting_times = (current_waiting_times - np.mean(current_waiting_times)) / (np.std(current_waiting_times) + 1e-6) if len(current_waiting_times) > 1 else np.zeros_like(current_waiting_times)
                norm_token_counts = (token_counts - np.mean(token_counts)) / (np.std(token_counts) + 1e-6) if len(token_counts) > 1 else np.zeros_like(token_counts)
                for i, req in enumerate(eligible_requests):
                    score = (priority_weights[0] * norm_speeds[i] +
                             priority_weights[1] * norm_waiting_times[i] +
                             priority_weights[2] * norm_token_counts[i])
                    scored_requests.append((score, req))
                scored_requests.sort(key=lambda x: x[0], reverse=True)

                # Form the batch up to target_batch_size, ensuring we don't exceed connections
                # The GPU check was for the whole batch type, assuming it's uniform.
                for score, req in scored_requests:
                    if len(chosen_batch) < target_batch_size:
                        chosen_batch.append(req)
                    else:
                        break
            
            if chosen_batch: # If any requests were actually selected for the batch
                gpu_gb_consumed_by_new_batch = intended_batch_gpu_cost_gb # This is the cost for the *type* of batch
                processing_duration_steps = self._get_batch_processing_duration_steps(chosen_dim_bucket_index, chosen_batch_size_option_index)
                
                self.active_processing_tasks.append({
                    'batch_id': self.next_batch_id_counter,
                    'gpu_gb_reserved': gpu_gb_consumed_by_new_batch,
                    'completion_step': self.current_step + processing_duration_steps
                })
                self.next_batch_id_counter += 1
                processed_request_ids_in_step = {req["id"] for req in chosen_batch}
                # print(f"Debug: Step {self.current_step} - Scheduled batch {self.next_batch_id_counter-1}, "
                #       f"consuming {gpu_gb_consumed_by_new_batch:.2f} GB GPU, "
                #       f"duration {processing_duration_steps} steps, "
                #       f"completes at step {self.current_step + processing_duration_steps}.")

        # 5. Calculate reward for the chosen_batch
        reward = self._calculate_profit(chosen_batch, chosen_dim_bucket_index, chosen_batch_size_option_index)

        # 6. Update request queue
        self.actual_request_queue = collections.deque(
            [req for req in self.actual_request_queue if req["id"] not in processed_request_ids_in_step],
            maxlen=self.actual_request_queue.maxlen
        )
        
        # 7. Generate new requests for the next step
        self._generate_new_requests(num_requests=self.np_random.integers(0, 3))

        # 8. Get next observation (based on the baseline server state)
        next_observation = self._get_observation()
        terminated = False
        truncated = self.current_step >= self.max_steps_per_episode
        
        info = {
            "chosen_batch_size": np.int32(len(chosen_batch)),
            "processed_ids": list(processed_request_ids_in_step),
            "target_img_size_agent": target_img_size,
            "target_batch_size_agent": target_batch_size,
            "gpu_gb_consumed_by_new_batch": gpu_gb_consumed_by_new_batch,
            "active_tasks_count": len(self.active_processing_tasks),
            "total_gpu_reserved_active_gb": sum(t['gpu_gb_reserved'] for t in self.active_processing_tasks)
        }
        return next_observation, float(reward), terminated, truncated, info

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        super().reset(seed=seed)
        self.current_step = 0
        self.next_request_id = 0
        self.active_processing_tasks = []
        self.next_batch_id_counter = 0
        
        self._generate_initial_server_state()
        self.actual_request_queue.clear()
        self._generate_new_requests(num_requests=self.np_random.integers(
            min(5, config.MAX_REQUESTS_IN_QUEUE_OBS // 2),
            config.MAX_REQUESTS_IN_QUEUE_OBS // 2 + 1
        ))
        initial_observation = self._get_observation()
        info = {
            "message": "Environment reset.",
            "chosen_batch_size": np.int32(0),
            "processed_ids": np.array([], dtype=object),
            "active_tasks_count": 0
        }
        return initial_observation, info

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

if __name__ == '__main__':
    # --- Dummy Config for Standalone Testing ---
    class DummyConfigForEnv:
        MAX_STEPS_PER_EPISODE = 50
        STEP_DURATION_SECONDS = 0.5 # Each step is 0.5s
        SERVER_STATE_DIM = 4
        MAX_REQUESTS_IN_QUEUE_OBS = 10
        REQUEST_FEATURE_SIZE = 4
        # Ensure these match the dimensions of your test CSV-derived arrays
        IMG_SIZE_BUCKETS = np.array([256.0, 512.0], dtype=np.float32)
        BATCH_SIZE_OPTIONS = np.array([1, 2], dtype=np.int32)
        K_DIM_BUCKETS = len(IMG_SIZE_BUCKETS)
        M_BATCH_SIZE_OPTIONS = len(BATCH_SIZE_OPTIONS)

        N_PRIORITY_WEIGHTS = 3
        MAX_ACTION_CONTINUOUS_PART = 1.0
        MAX_WAIT_TIME_PENALTY_THRESHOLD = 10 # steps
        WAIT_TIME_PENALTY_AMOUNT = 5.0
        BASE_PROFIT_PER_TOKEN = 0.1
        BASE_PROFIT_PER_IMG_SIZE_UNIT = 0.0005
        INTERNET_SPEED_PROFIT_FACTOR = 0.2
        # Derived from CSV (Time_s)
        PROCESSING_TIMES_BY_INDEX = np.array([
            [1.345, 0.987], # For 256px, batch 1 & 2
            [2.5,   1.8]    # For 512px, batch 1 & 2
        ], dtype=np.float32)
        # Derived from CSV (PeakMemory_GB for whole batch)
        PEAK_MEMORY_GB_BY_INDEX = np.array([
            [0.26, 0.47],   # For 256px, batch 1 & 2
            [0.5,  0.9]     # For 512px, batch 1 & 2
        ], dtype=np.float32)

        EXTERNAL_STATE_CHANGE_PROBABILITY = 0.2
        GPU_AVAIL_EXTERNAL_CHANGE_MIN = -0.05
        GPU_AVAIL_EXTERNAL_CHANGE_MAX = 0.15 # Bias towards regeneration
        RESOURCE_EXTERNAL_CHANGE_MIN = -0.05
        RESOURCE_EXTERNAL_CHANGE_MAX = 0.05
        EXTERNAL_CONNECTIONS_CHANGE_MIN = -1
        EXTERNAL_CONNECTIONS_CHANGE_MAX = 1
        GPU_TOTAL_CAPACITY_GB = 1.0 # Smaller total GPU for testing to see impact

    config_backup = config
    config = DummyConfigForEnv()

    env = Env()
    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)

    obs, info_reset = env.reset(seed=42)
    print(f"\nInitial State: Server Obs: {obs['server_state']}")

    for i in range(20): # Run more steps to see tasks complete
        random_action = env.action_space.sample()
        print(f"\n--- Step {env.current_step + 1} (Actual Env Step: {env.current_step}) ---")
        
        server_state_before_step = obs['server_state'].copy()
        gpu_baseline_frac_before = server_state_before_step[1]
        active_gpu_reserved_before = sum(t['gpu_gb_reserved'] for t in env.active_processing_tasks)
        usable_gpu_before_step = max(0, gpu_baseline_frac_before * config.GPU_TOTAL_CAPACITY_GB - active_gpu_reserved_before)

        print(f"Server State Before: {server_state_before_step} (GPU Baseline Frac: {gpu_baseline_frac_before:.2f})")
        print(f"Active GPU Reserved Before: {active_gpu_reserved_before:.2f} GB. Usable GPU for this step: {usable_gpu_before_step:.2f} GB")
        # print(f"Active tasks before: {env.active_processing_tasks}")

        next_obs, reward, terminated, truncated, info = env.step(random_action)
        
        print(f"Agent Choice: ImgSize {info['target_img_size_agent']}, BatchSize {info['target_batch_size_agent']}")
        print(f"Actual Batch Processed Size: {info['chosen_batch_size']}")
        if info['chosen_batch_size'] > 0:
            print(f"  New Batch GPU Reserved: {info['gpu_gb_consumed_by_new_batch']:.2f} GB")
        print(f"Reward: {reward:.3f}")
        print(f"Server State After (Baseline): {next_obs['server_state']} (GPU Baseline Frac: {next_obs['server_state'][1]:.2f})")
        print(f"Active Tasks After: {info['active_tasks_count']}, Total GPU Reserved: {info['total_gpu_reserved_active_gb']:.2f} GB")
        # print(f"Active tasks after: {env.active_processing_tasks}")
        
        obs = next_obs
        if truncated:
            print("Episode truncated.")
            break
    
    config = config_backup
