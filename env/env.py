"""
Custom Gymnasium Environment for simulating a photo editing server.

This environment models a server that receives photo editing requests and processes
them in batches. The primary goal for a Reinforcement Learning (RL) agent
interacting with this environment is to learn an optimal scheduling policy that
maximizes profit. Profit is derived from processing requests, considering factors
like request complexity (token count, image size), user internet speed, and
penalties for excessive waiting times.

Core Simulation Mechanics:
-   Resource Consumption: GPU memory is a critical resource. When a batch of
    requests is scheduled for processing, it consumes a specific amount of GPU
    memory (defined in `config.PEAK_MEMORY_GB_BY_INDEX`) for the duration of its
    processing.
-   Processing Duration: The time it takes to process a batch depends on the
    image size and batch size (defined in `config.PROCESSING_TIMES_BY_INDEX`,
    derived from benchmark CSV data 'Time_s'). This duration is converted into
    simulation steps.
-   Resource Release: GPU memory and processing connections are freed once a
    batch completes its processing.
-   Dynamic Server State: The server's baseline resource availability (general
    memory, GPU, bandwidth, connections) can be influenced by simulated external
    factors, introducing stochasticity.
-   Request Queue: New photo editing requests arrive dynamically and are added
    to a queue. Each request has features like token count, internet speed,
    target image size, and arrival time.
-   Batch Formation: The RL agent decides which type of batch to form (image size,
    batch size) and provides priority weights. The environment then attempts to
    form this batch from eligible requests in the queue, prioritizing them based
    on the agent's weights and request features.

Dependencies:
-   This environment heavily relies on `config.py` for all its core parameters,
    including action/observation space dimensions, resource capacities,
    processing times, memory costs, and reward calculation parameters.
    Understanding `config.py` is crucial for understanding this environment.

Observation Space (`spaces.Dict`):
    -   `server_state` (`spaces.Box`): A vector representing the current state of
        the server. Its components are typically:
        * `M_avail`: Normalized available general system memory (0.0 to 1.0).
        * `G_avail_baseline`: Normalized baseline GPU availability trend (0.0 to 1.0).
            This is the general availability before accounting for currently active tasks.
        * `B_avail`: Normalized available bandwidth (0.0 to 1.0).
        * `Connections`: Number of available concurrent processing connections/slots.
    -   `request_queue` (`spaces.Box`): A 2D array representing a snapshot of the
        photo editing requests currently waiting. Each row corresponds to a request
        and its features are typically:
        * `token_count`: A measure of processing complexity/value.
        * `internet_speed`: Normalized user internet speed, affecting profit.
        * `img_size`: The target image size for the editing task.
        * `waiting_time`: Number of simulation steps the request has been in the queue.

Action Space (`spaces.Dict`):
    The agent's action is a dictionary containing:
    -   `dim_bucket_logits` (`spaces.Box`): Logits for selecting a discrete image
        size bucket (from `config.IMG_SIZE_BUCKETS`). The environment uses
        `argmax` to determine the chosen bucket.
    -   `batch_size_logits` (`spaces.Box`): Logits for selecting a discrete batch
        size option (from `config.BATCH_SIZE_OPTIONS`). The environment uses
        `argmax` to determine the chosen option.
    -   `priority_weights` (`spaces.Box`): A vector of continuous weights used by
        the environment to score and prioritize requests from the queue when
        forming a batch. These weights are typically applied to normalized request
        features (e.g., internet speed, waiting time, token count).

Key Internal Attributes:
    -   `actual_request_queue` (`collections.deque`): The internal queue holding
        all current photo editing requests.
    -   `active_processing_tasks` (list of dicts): Tracks tasks currently being
        processed by the server. Each entry contains:
        * `'batch_id'`: A unique identifier for the batch.
        * `'gpu_gb_reserved'`: The amount of GPU memory (in GB) reserved by this batch.
        * `'completion_step'`: The simulation step at which this batch will finish
            processing and release its resources.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import collections
import math

import config

class Env(gym.Env):
    """
    ## Custom Gymnasium Environment for simulating a photo editing server.

    This environment models a server that processes photo editing requests.
    An RL agent interacts with this environment to learn an optimal scheduling
    policy to maximize profit, considering resource constraints (GPU, connections),
    processing times, and request characteristics.

    Attributes:
        max_steps_per_episode (int): Maximum number of steps before an episode is truncated.
        current_step (int): The current step within the ongoing episode.
        action_space (gym.spaces.Dict): Defines the structure of actions the agent can take.
            It includes logits for choosing image size and batch size, and continuous
            weights for prioritizing requests.
        observation_space (gym.spaces.Dict): Defines the structure of observations
            provided to the agent. It includes the server's current state (resources)
            and a view of the request queue.
        actual_request_queue (collections.deque): Holds all incoming photo editing requests.
        next_request_id (int): Counter for assigning unique IDs to new requests.
        _server_state (np.ndarray): Internal array representing the server's resource state.
        active_processing_tasks (list): A list of dictionaries, where each dictionary
            represents a batch currently being processed. It stores the batch ID,
            GPU memory reserved, and the simulation step at which it will complete.
        next_batch_id_counter (int): Counter for assigning unique IDs to new processing batches.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, max_steps_per_episode: int = config.MAX_STEPS_PER_EPISODE):
        super().__init__()
        self.max_steps_per_episode = max_steps_per_episode
        self.current_step = 0

        # Define the observation space
        # server_state[0]: M_avail (general memory availability, 0-1)
        # server_state[1]: G_avail_baseline (baseline GPU availability trend, 0-1)
        # server_state[2]: B_avail (bandwidth availability, 0-1)
        # server_state[3]: N_connections_avail (number of available processing connections)
        self._observation_space = spaces.Dict({
            'server_state': spaces.Box(
                low=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), # M, G_baseline, B, Connections
                high=np.array([1.0, 1.0, 1.0, config.MAX_REQUESTS_IN_QUEUE_OBS * 2.0], dtype=np.float32), # Max connections can be higher than queue view
                shape=(config.SERVER_STATE_DIM,),
                dtype=np.float32
            ),
            # request_queue features: [token_count, internet_speed, img_size, waiting_time]
            'request_queue': spaces.Box(
                low=np.zeros((config.MAX_REQUESTS_IN_QUEUE_OBS, config.REQUEST_FEATURE_SIZE), dtype=np.float32),
                high=np.array( # Define reasonable upper bounds for request features
                    [[80.0, # Max token_count
                      1.0,   # Max internet_speed (normalized)
                      config.IMG_SIZE_BUCKETS[-1] if config.IMG_SIZE_BUCKETS.size > 0 else 2048.0, # Max img_size from config or a default
                      float(config.MAX_STEPS_PER_EPISODE)] # Max waiting_time (can wait up to full episode length)
                     ] * config.MAX_REQUESTS_IN_QUEUE_OBS,
                    dtype=np.float32
                ),
                dtype=np.float32
            )
        })

        # Define the action space
        # "dim_bucket_logits": Logits for choosing an image size processing bucket.
        # "batch_size_logits": Logits for choosing a batch size for processing.
        # "priority_weights": Continuous weights for prioritizing requests in the queue.
        self.action_space = spaces.Dict({
            "dim_bucket_logits": spaces.Box(low=-np.inf, high=np.inf, shape=(config.K_DIM_BUCKETS,), dtype=np.float32),
            "batch_size_logits": spaces.Box(low=-np.inf, high=np.inf, shape=(config.M_BATCH_SIZE_OPTIONS,), dtype=np.float32),
            "priority_weights": spaces.Box(
                low=-config.MAX_ACTION_CONTINUOUS_PART,
                high=config.MAX_ACTION_CONTINUOUS_PART,
                shape=(config.N_PRIORITY_WEIGHTS,),
                dtype=np.float32
            )
        })

        # Initialize internal state variables
        # Internal queue can be larger than observed part, like some request is coming but not arrived yet
        self.actual_request_queue = collections.deque(maxlen=config.MAX_REQUESTS_IN_QUEUE_OBS * 2) 
        self.next_request_id = 0
        self._server_state = np.zeros(config.SERVER_STATE_DIM, dtype=np.float32)
        
        self.active_processing_tasks = [] # Tracks ongoing tasks: {'batch_id': int, 'gpu_gb_reserved': float, 'completion_step': int}
        self.next_batch_id_counter = 0


    def _get_img_size_from_bucket_index(self, index: int) -> float:
        """
        Retrieves the image size corresponding to a given bucket index.

        Args:
            index: The index of the image size bucket.

        Returns:
            The image size (e.g., width in pixels) or the smallest size if index is invalid.
        """
        if 0 <= index < len(config.IMG_SIZE_BUCKETS):
            return config.IMG_SIZE_BUCKETS[index]
        print(f"Warning: Invalid dimension bucket index {index} (max: {len(config.IMG_SIZE_BUCKETS)-1}), using smallest.")
        return config.IMG_SIZE_BUCKETS[0] if len(config.IMG_SIZE_BUCKETS) > 0 else 256.0 # Default if empty

    def _get_batch_size_from_option_index(self, index: int) -> int:
        """
        Retrieves the batch size corresponding to a given option index.

        Args:
            index: The index of the batch size option.

        Returns:
            The batch size (number of items) or the smallest option if index is invalid.
        """
        if 0 <= index < len(config.BATCH_SIZE_OPTIONS):
            return int(config.BATCH_SIZE_OPTIONS[index])
        print(f"Warning: Invalid batch size option index {index} (max: {len(config.BATCH_SIZE_OPTIONS)-1}), using smallest.")
        return int(config.BATCH_SIZE_OPTIONS[0]) if len(config.BATCH_SIZE_OPTIONS) > 0 else 1 # Default if empty

    def _get_batch_processing_duration_steps(self, img_size_bucket_idx: int, batch_size_option_idx: int) -> int:
        """
        Calculates the processing duration for a batch in simulation steps.
        It retrieves the base processing time in seconds from `config.PROCESSING_TIMES_BY_INDEX`
        and converts it to steps using `config.STEP_DURATION_SECONDS`. Handles NaN values
        by defaulting to a very long duration.

        Args:
            img_size_bucket_idx: Index for the image size bucket.
            batch_size_option_idx: Index for the batch size option.

        Returns:
            The processing duration in simulation steps (at least 1).
        """
        # Clip indices to be within the bounds of the config array to prevent errors
        valid_img_idx = np.clip(img_size_bucket_idx, 0, config.PROCESSING_TIMES_BY_INDEX.shape[0] - 1)
        valid_batch_idx = np.clip(batch_size_option_idx, 0, config.PROCESSING_TIMES_BY_INDEX.shape[1] - 1)
        
        time_seconds = config.PROCESSING_TIMES_BY_INDEX[valid_img_idx, valid_batch_idx]
        
        # Handle NaN processing times (e.g., for infeasible combinations)
        if np.isnan(time_seconds):
            print(f"Warning: NaN processing time for img_idx {valid_img_idx}, batch_idx {valid_batch_idx}. Using a large default duration.")
            # Default to a duration that effectively makes this choice highly unprofitable
            time_seconds = float(self.max_steps_per_episode * config.STEP_DURATION_SECONDS * 2) # Significantly longer than episode
                        
        duration_steps = math.ceil(time_seconds / config.STEP_DURATION_SECONDS)
        return max(1, duration_steps) # Ensure task takes at least 1 simulation step

    def _get_batch_gpu_consumption_gb(self, img_size_bucket_idx: int, batch_size_option_idx: int) -> float:
        """
        Retrieves the peak GPU memory consumption for a given batch configuration.
        It uses `config.PEAK_MEMORY_GB_BY_INDEX`. If a NaN value is encountered (indicating
        an infeasible or unbenchmarked configuration), it returns a value higher than
        the total server GPU capacity, effectively making it unschedulable.

        Args:
            img_size_bucket_idx: Index for the image size bucket.
            batch_size_option_idx: Index for the batch size option.

        Returns:
            The peak GPU memory in GB for the batch. Returns a very high value for NaN entries.
        """
        # Clip indices to be within the bounds of the config array
        valid_img_idx = np.clip(img_size_bucket_idx, 0, config.PEAK_MEMORY_GB_BY_INDEX.shape[0] - 1)
        valid_batch_idx = np.clip(batch_size_option_idx, 0, config.PEAK_MEMORY_GB_BY_INDEX.shape[1] - 1)
        
        gpu_gb = config.PEAK_MEMORY_GB_BY_INDEX[valid_img_idx, valid_batch_idx]
        
        # Handle NaN peak memory values (CRITICAL FIX)
        if np.isnan(gpu_gb):
            print(f"Warning: NaN peak memory for img_idx {valid_img_idx}, batch_idx {valid_batch_idx}. Marking as infeasible (using very high memory).")
            # Return a value greater than total capacity to make this batch unschedulable
            return config.GPU_TOTAL_CAPACITY_GB + 1.0
        return gpu_gb

    def _generate_initial_server_state(self) -> None:
        """Initializes the server's resource state at the beginning of an episode."""
        self._server_state[0] = self.np_random.random()  # M_avail (general memory)
        self._server_state[1] = 1.0  # G_avail_baseline (GPU baseline availability) - start with full
        self._server_state[2] = self.np_random.random()  # B_avail (bandwidth)
        # Initial number of connections: random number between 5 and MAX_REQUESTS_IN_QUEUE_OBS
        self._server_state[3] = float(self.np_random.integers(5, config.MAX_REQUESTS_IN_QUEUE_OBS + 1))

    def _generate_new_requests(self, num_requests: int = 1) -> None:
        """
        Generates new photo editing requests and adds them to the `actual_request_queue`.
        Request features (token count, internet speed, image size) are randomized.

        Args:
            num_requests: The number of new requests to generate.
        """
        possible_token_count = np.arange(10, 81, 10) # Example: tokens from 10 to 200 in steps of 10
        for _ in range(num_requests):
            if len(self.actual_request_queue) < self.actual_request_queue.maxlen:
                _img_size_to_assign = 512.0  # Default if IMG_SIZE_BUCKETS is empty

                if config.IMG_SIZE_BUCKETS.size > 0:
                    if config.IMG_SIZE_BUCKETS.size == 6:
                        # Define weights for a distribution peaked at indices 2 and 3 (0-indexed)
                        # e.g., [~8%, ~17%, 25%, 25%, ~17%, ~8%]
                        _probabilities = np.array([1.0, 2.0, 3.0, 3.0, 2.0, 1.0], dtype=np.float32)
                        _probabilities /= np.sum(_probabilities) # Normalize to sum to 1
                        _img_size_to_assign = float(self.np_random.choice(config.IMG_SIZE_BUCKETS, p=_probabilities))
                    else:
                        # Fallback to uniform distribution if the number of sizes is not 6
                        _img_size_to_assign = float(self.np_random.choice(config.IMG_SIZE_BUCKETS))

                request = {
                    "id": self.next_request_id,
                    "token_count": float(self.np_random.choice(possible_token_count)),
                    "internet_speed": self.np_random.random(), # Normalized 0-1
                    "img_size": _img_size_to_assign,
                    "arrival_time": self.current_step # Mark arrival time with current step
                }
                self.actual_request_queue.append(request)
                self.next_request_id += 1

    def _get_observation(self) -> dict:
        """
        Constructs the observation dictionary to be returned to the agent.
        It includes the current server state and a snapshot of the request queue.

        Returns:
            A dictionary containing 'server_state' and 'request_queue'.
        """
        obs_request_queue = np.zeros((config.MAX_REQUESTS_IN_QUEUE_OBS, config.REQUEST_FEATURE_SIZE), dtype=np.float32)
        current_queue_list = list(self.actual_request_queue)
        for i, req in enumerate(current_queue_list[:config.MAX_REQUESTS_IN_QUEUE_OBS]): # Fill up to MAX_REQUESTS_IN_QUEUE_OBS
            waiting_time = float(self.current_step - req["arrival_time"])  # Exactly is waiting stepa
            obs_request_queue[i, 0] = req["token_count"]
            obs_request_queue[i, 1] = req["internet_speed"]
            obs_request_queue[i, 2] = req["img_size"]
            obs_request_queue[i, 3] = waiting_time
        return {'server_state': self._server_state.copy(), 'request_queue': obs_request_queue}

    def _calculate_profit(self, chosen_batch: list, chosen_dim_bucket_index: int, chosen_batch_size_option_index: int) -> float:
        """
        Calculates the profit for processing a chosen batch of requests.
        Profit considers base value from tokens and image size, internet speed multipliers,
        and penalties for excessive waiting times. The net profit is then normalized
        by the total operational time (processing time + estimated download time for the
        response) to get a profit rate.
        Formula:
            profit_rate = (total_base_profit - total_penalty) / total_operational_time_seconds
        where:
            - total_base_profit is the sum of profits from all requests in the batch.
            - total_penalty is the sum of penalties for requests that waited too long.
            - total_operational_time_seconds = processing_time_seconds + download_time_for_batch.
            - processing_time_seconds is the scheduled server-side processing duration.
            - download_time_for_batch is an estimate of the time to send the response,
              inversely proportional to the average internet speed of requests in the batch.
        This function returns 0.0 if the batch is empty.

        Args:
            chosen_batch: A list of request dictionaries that were processed.
            chosen_dim_bucket_index: The image size bucket index for the processed batch.
            chosen_batch_size_option_index: The batch size option index for the processed batch.

        Returns:
            The calculated profit rate for the batch. Returns 0.0 if the batch is empty.
        """
        if not chosen_batch:
            return 0.0
        
        total_base_profit = 0.0
        total_penalty = 0.0
        sum_internet_speed = 0.0
        for req in chosen_batch:
            # Calculate base profit for this request
            base_profit_for_req = (req["token_count"] * config.BASE_PROFIT_PER_TOKEN +
                                   req["img_size"] * config.BASE_PROFIT_PER_IMG_SIZE_UNIT)
            # Apply internet speed multiplier to profit
            speed_multiplier = 1.0 + (req["internet_speed"] * config.INTERNET_SPEED_PROFIT_FACTOR)
            total_base_profit += base_profit_for_req * speed_multiplier
            sum_internet_speed += req["internet_speed"]
            
            # Calculate waiting time penalty
            waiting_time = self.current_step - req["arrival_time"] # Waiting time until start of processing
            if waiting_time > config.MAX_WAIT_TIME_PENALTY_THRESHOLD:
                total_penalty += config.WAIT_TIME_PENALTY_AMOUNT
                
        net_profit = total_base_profit - total_penalty
        
        # Server-side processing time
        processing_time_seconds = self._get_batch_processing_duration_steps(
            chosen_dim_bucket_index, chosen_batch_size_option_index
        ) * config.STEP_DURATION_SECONDS
        
        # Estimate download time for the batch response
        # Assumes config.BASE_DOWNLOAD_TIME_SECONDS and config.INTERNET_SPEED_EPSILON are defined
        average_internet_speed = sum_internet_speed / len(chosen_batch)
        download_time_for_batch = config.BASE_DOWNLOAD_TIME_SECONDS / \
                                  (average_internet_speed + config.INTERNET_SPEED_EPSILON)
                                  
        total_operational_time_seconds = processing_time_seconds + download_time_for_batch
        
        return net_profit / total_operational_time_seconds if total_operational_time_seconds > 1e-6 else net_profit


    def _update_active_tasks_and_server_state(self) -> None:
        """
        Manages active processing tasks and updates the server's baseline state.
        1.  Frees resources (GPU) from tasks that have completed by the current step.
        2.  Applies random external changes to the server's baseline resource availability
            (general memory, GPU, bandwidth, connections) based on probabilities and
            magnitudes defined in `config.py`.
        """
        # --- 1. Free resources from completed tasks ---
        completed_task_indices = []
        for i, task in enumerate(self.active_processing_tasks):
            if task['completion_step'] <= self.current_step:
                completed_task_indices.append(i)
                # Optional: print(f"Debug: Task {task['batch_id']} completed at step {self.current_step}. Freed {task['gpu_gb_reserved']} GB GPU.")

        # Remove completed tasks by iterating in reverse to maintain correct indices during deletion
        for i in sorted(completed_task_indices, reverse=True):
            del self.active_processing_tasks[i]

        # --- 2. Apply external changes to the baseline server state ---
        if self.np_random.random() < config.EXTERNAL_STATE_CHANGE_PROBABILITY:
            # Update general memory (M_avail) and bandwidth (B_avail)
            for i in [0, 2]: 
                change = self.np_random.uniform(config.RESOURCE_EXTERNAL_CHANGE_MIN, config.RESOURCE_EXTERNAL_CHANGE_MAX)
                self._server_state[i] = np.clip(self._server_state[i] + change, 0.0, 1.0)
            
            # Update baseline GPU availability (G_avail_baseline)
            gpu_change = self.np_random.uniform(config.GPU_AVAIL_EXTERNAL_CHANGE_MIN, config.GPU_AVAIL_EXTERNAL_CHANGE_MAX)
            self._server_state[1] = np.clip(self._server_state[1] + gpu_change, 0.0, 1.0)
            
            # Update number of available connections
            conn_change = self.np_random.integers(config.EXTERNAL_CONNECTIONS_CHANGE_MIN, config.EXTERNAL_CONNECTIONS_CHANGE_MAX + 1)
            self._server_state[3] = np.clip(self._server_state[3] + conn_change, 1.0, config.MAX_REQUESTS_IN_QUEUE_OBS * 2.0) # Ensure at least 1 connection


    def step(self, action: dict) -> tuple[dict, float, bool, bool, dict]:
        """
        Executes one time step in the environment based on the agent's action.

        The process involves:
        1.  Incrementing the current step.
        2.  Updating active tasks (releasing resources for completed ones) and applying
            external random changes to the server's baseline state.
        3.  Interpreting the agent's action (chosen image size, batch size, priority weights).
        4.  Calculating currently available GPU resources and connection budget for this step.
        5.  Attempting to form a batch based on the agent's choice and resource availability:
            - Checks if the *intended batch type* (image size + batch size combination)
              is feasible given its GPU cost and the number of items fitting the connection budget.
            - If feasible, filters eligible requests from the queue (matching image size).
            - Scores eligible requests using the agent's priority weights.
            - Selects the highest-scoring requests to form the batch, up to the chosen batch size.
        6.  If a batch is successfully formed:
            - Reserves GPU memory for the batch.
            - Schedules the batch for processing by adding it to `active_processing_tasks`
              with a calculated completion step.
        7.  Calculating the reward based on the profit from the processed batch.
        8.  Updating the request queue by removing processed requests.
        9.  Generating new incoming requests for the next step.
        10. Determining if the episode has terminated or truncated.
        11. Returning the next observation, reward, termination/truncation flags, and info.

        Args:
            action: A dictionary representing the agent's action, containing logits for
                    image size and batch size, and continuous priority weights.

        Returns:
            A tuple containing
            -   `next_observation` (dict): The observation for the next state.
            -   `reward` (float): The reward received for the action taken.
            -   `terminated` (bool): Whether the episode has ended naturally (not used here, always False).
            -   `truncated` (bool): Whether the episode was cut short (e.g., max steps reached).
            -   `info` (dict): Auxiliary information about the step.
        """
        self.current_step += 1

        # 1. Update active tasks & apply external changes to baseline server state
        self._update_active_tasks_and_server_state()

        # 2. Determine agent's choice from action dictionary
        dim_bucket_logits = action["dim_bucket_logits"]
        batch_size_logits = action["batch_size_logits"]
        priority_weights = action["priority_weights"]

        chosen_dim_bucket_index = int(np.argmax(dim_bucket_logits))
        chosen_batch_size_option_index = int(np.argmax(batch_size_logits))

        target_img_size = self._get_img_size_from_bucket_index(chosen_dim_bucket_index)
        target_batch_size = self._get_batch_size_from_option_index(chosen_batch_size_option_index)

        # 3. Calculate currently available GPU resources and connection budget for THIS STEP
        gpu_reserved_by_active_tasks_gb = sum(task['gpu_gb_reserved'] for task in self.active_processing_tasks)
        baseline_gpu_available_gb = self._server_state[1] * config.GPU_TOTAL_CAPACITY_GB # G_avail_baseline * Total GPU
        currently_usable_gpu_gb = max(0, baseline_gpu_available_gb - gpu_reserved_by_active_tasks_gb)
        
        current_step_connections_budget = int(self._server_state[3]) # Connections are a per-step budget
        
        chosen_batch = [] # Stores requests selected for processing in this step
        processed_request_ids_in_step = set() # IDs of requests processed
        gpu_gb_consumed_by_new_batch = 0.0 # GPU consumed by the batch formed in this step

        # 4. Batch Formation Logic
        # First, check if the *intended batch type* (img_size + batch_size combo) is even possible
        intended_batch_gpu_cost_gb = self._get_batch_gpu_consumption_gb(chosen_dim_bucket_index, chosen_batch_size_option_index)
        
        # Conditions for attempting to form a batch:
        # - Agent chose a positive target batch size.
        # - The GPU cost of the chosen batch *type* is within currently usable GPU.
        # - The target number of items in the batch is within the connection budget.
        if target_batch_size > 0 and \
           intended_batch_gpu_cost_gb <= currently_usable_gpu_gb and \
           target_batch_size <= current_step_connections_budget:
            
            # If the batch type is feasible, try to fill it with eligible requests
            eligible_requests = [req for req in self.actual_request_queue if req["img_size"] == target_img_size]
            scored_requests = [] # To store (score, request) tuples

            if eligible_requests:
                # Normalize features of eligible requests for scoring
                speeds = np.array([req["internet_speed"] for req in eligible_requests], dtype=np.float32)
                current_waiting_times = np.array([self.current_step - req["arrival_time"] for req in eligible_requests], dtype=np.float32)
                token_counts = np.array([req["token_count"] for req in eligible_requests], dtype=np.float32)
                
                # Normalize (handle single element case for std dev)
                norm_speeds = (speeds - np.mean(speeds)) / (np.std(speeds) + 1e-6) if len(speeds) > 1 else np.zeros_like(speeds)
                norm_waiting_times = (current_waiting_times - np.mean(current_waiting_times)) / (np.std(current_waiting_times) + 1e-6) if len(current_waiting_times) > 1 else np.zeros_like(current_waiting_times)
                norm_token_counts = (token_counts - np.mean(token_counts)) / (np.std(token_counts) + 1e-6) if len(token_counts) > 1 else np.zeros_like(token_counts)

                # Score each eligible request using agent's priority weights
                for i, req in enumerate(eligible_requests):
                    score = (priority_weights[0] * norm_speeds[i] +
                             priority_weights[1] * norm_waiting_times[i] +
                             priority_weights[2] * norm_token_counts[i])
                    scored_requests.append((score, req))
                
                scored_requests.sort(key=lambda x: x[0], reverse=True) # Sort by score, highest first

                # Form the batch up to target_batch_size with highest scored requests
                for score, req in scored_requests:
                    if len(chosen_batch) < target_batch_size:
                        chosen_batch.append(req)
                    else:
                        break # Batch is full
            
            # If any requests were actually selected for the batch
            if chosen_batch:
                gpu_gb_consumed_by_new_batch = intended_batch_gpu_cost_gb # GPU cost is for the batch *type*
                processing_duration_steps = self._get_batch_processing_duration_steps(chosen_dim_bucket_index, chosen_batch_size_option_index)
                
                # Add this new batch to active processing tasks
                self.active_processing_tasks.append({
                    'batch_id': self.next_batch_id_counter,
                    'gpu_gb_reserved': gpu_gb_consumed_by_new_batch, # Reserve the GPU
                    'completion_step': self.current_step + processing_duration_steps # Mark when it will finish
                })
                self.next_batch_id_counter += 1
                processed_request_ids_in_step = {req["id"] for req in chosen_batch}
                # Optional: print(f"Debug: Step {self.current_step} - Scheduled batch {self.next_batch_id_counter-1}, "
                #       f"consuming {gpu_gb_consumed_by_new_batch:.2f} GB GPU, "
                #       f"duration {processing_duration_steps} steps, "
                #       f"completes at step {self.current_step + processing_duration_steps}.")

        # 5. Calculate reward for the chosen_batch (will be 0 if chosen_batch is empty)
        reward = self._calculate_profit(chosen_batch, chosen_dim_bucket_index, chosen_batch_size_option_index)

        # 6. Update actual_request_queue: remove processed requests
        self.actual_request_queue = collections.deque(
            [req for req in self.actual_request_queue if req["id"] not in processed_request_ids_in_step],
            maxlen=self.actual_request_queue.maxlen
        )
        
        # 7. Generate new requests for the next step (simulating arrivals)
        self._generate_new_requests(num_requests=self.np_random.integers(0, 3)) # Randomly 0, 1, or 2 new requests

        # 8. Get next observation
        next_observation = self._get_observation()
        
        # Determine termination and truncation
        terminated = False # This environment typically doesn't have a natural termination condition based on state
        truncated = self.current_step >= self.max_steps_per_episode
        
        # Gather information for debugging or analysis
        info = {
            "chosen_batch_size": np.int32(len(chosen_batch)), # Actual number of items processed
            "processed_ids": list(processed_request_ids_in_step),
            "target_img_size_agent": target_img_size, # Agent's intended image size
            "target_batch_size_agent": target_batch_size, # Agent's intended batch size
            "gpu_gb_consumed_by_new_batch": gpu_gb_consumed_by_new_batch,
            "active_tasks_count": len(self.active_processing_tasks),
            "total_gpu_reserved_active_gb": sum(t['gpu_gb_reserved'] for t in self.active_processing_tasks)
        }
        return next_observation, float(reward), terminated, truncated, info

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """
        Resets the environment to an initial state for the beginning of a new episode.

        Args:
            seed: Optional seed for the random number generator.
            options: Optional dictionary of environment-specific options.

        Returns:
            A tuple containing the initial observation and an info dictionary.
        """
        super().reset(seed=seed) # Important to call super for proper seeding
        self.current_step = 0
        self.next_request_id = 0
        self.active_processing_tasks = [] # Clear any ongoing tasks
        self.next_batch_id_counter = 0
        
        self._generate_initial_server_state() # Set initial server resources
        self.actual_request_queue.clear() # Clear existing requests
        # Generate an initial set of requests
        self._generate_new_requests(num_requests=self.np_random.integers(
            min(5, config.MAX_REQUESTS_IN_QUEUE_OBS // 2), # At least 5 or half the observable queue
            config.MAX_REQUESTS_IN_QUEUE_OBS // 2 + 1
        ))
        
        initial_observation = self._get_observation()
        info = { # Basic info for reset
            "message": "Environment reset.",
            "initial_requests_generated": len(self.actual_request_queue),
            "chosen_batch_size": np.int32(0), # For consistency with step info
            "processed_ids": [],
            "active_tasks_count": 0
        }
        return initial_observation, info

    @property
    def observation_space(self) -> gym.spaces.Space:
        """Returns the observation space of the environment."""
        return self._observation_space
    
    # render method is not implemented as it's a simulation, but required by Gym
    def render(self):
        """
        Render the environment. Currently, no visual rendering is implemented.
        This method is a placeholder to satisfy the `gym.Env` interface.
        """
        if self.metadata['render_modes'] == ['human']:
            pass # No specific human-readable rendering for now.
                 # Could print server state or queue if needed for debugging.
        else:
            return super().render() # For other modes if supported by parent

    # close method is good practice for cleanup, though not strictly needed here
    def close(self):
        """
        Perform any necessary cleanup.
        """
        pass


if __name__ == '__main__':
    # --- Dummy Config for Standalone Testing ---
    # This allows testing env.py independently of the main project's config.py
    # by temporarily overriding it.
    class DummyConfigForEnv:
        MAX_STEPS_PER_EPISODE = 50
        STEP_DURATION_SECONDS = 0.5 # Each step is 0.5s
        SERVER_STATE_DIM = 4
        MAX_REQUESTS_IN_QUEUE_OBS = 10
        REQUEST_FEATURE_SIZE = 4
        
        IMG_SIZE_BUCKETS = np.array([256.0, 512.0, 1024.0], dtype=np.float32) # Added a third option
        BATCH_SIZE_OPTIONS = np.array([1, 2, 4], dtype=np.int32) # Added a third option
        K_DIM_BUCKETS = len(IMG_SIZE_BUCKETS)
        M_BATCH_SIZE_OPTIONS = len(BATCH_SIZE_OPTIONS)

        N_PRIORITY_WEIGHTS = 3
        MAX_ACTION_CONTINUOUS_PART = 1.0
        
        MAX_WAIT_TIME_PENALTY_THRESHOLD = 10 # steps
        WAIT_TIME_PENALTY_AMOUNT = 5.0
        BASE_PROFIT_PER_TOKEN = 0.1
        BASE_PROFIT_PER_IMG_SIZE_UNIT = 0.0005
        INTERNET_SPEED_PROFIT_FACTOR = 0.2
        
        # Derived from CSV (Time_s) - expanded for 3x3
        PROCESSING_TIMES_BY_INDEX = np.array([
            [1.345, 0.987, 0.700], # For 256px, batch 1, 2, 4
            [2.5,   1.8,   1.200], # For 512px, batch 1, 2, 4
            [5.0,   3.5,   2.000]  # For 1024px, batch 1, 2, 4
        ], dtype=np.float32)
        
        # Derived from CSV (PeakMemory_GB for whole batch) - expanded for 3x3
        # Introduce a NaN to test handling
        PEAK_MEMORY_GB_BY_INDEX = np.array([
            [0.26, 0.47, 0.80],   # For 256px, batch 1, 2, 4
            [0.5,  0.9,  1.50],   # For 512px, batch 1, 2, 4
            [1.0,  np.nan, 3.00]  # For 1024px, batch 1, (NaN for 2), 4
        ], dtype=np.float32)

        EXTERNAL_STATE_CHANGE_PROBABILITY = 0.2
        GPU_AVAIL_EXTERNAL_CHANGE_MIN = -0.05
        GPU_AVAIL_EXTERNAL_CHANGE_MAX = 0.15 
        RESOURCE_EXTERNAL_CHANGE_MIN = -0.05
        RESOURCE_EXTERNAL_CHANGE_MAX = 0.05
        EXTERNAL_CONNECTIONS_CHANGE_MIN = -1
        EXTERNAL_CONNECTIONS_CHANGE_MAX = 1
        GPU_TOTAL_CAPACITY_GB = 2.0 # Adjusted total GPU for testing NaN case (1024px, batch 4 needs 3GB)

    config_backup = config # Backup the original config
    config = DummyConfigForEnv() # Override with dummy for this test block

    env = Env()
    print("--- Environment Initialized with Dummy Config ---")
    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)

    obs, info_reset = env.reset(seed=42)
    print(f"\nInitial Observation: Server State: {obs['server_state']}, Requests in Obs: {np.count_nonzero(obs['request_queue'][:,0])}")
    print(f"Initial Info: {info_reset}")

    total_reward = 0
    for i in range(config.MAX_STEPS_PER_EPISODE + 5): # Run a few more steps than episode length
        random_action = env.action_space.sample() # Agent takes a random action
        
        # Manually set a specific action to test NaN GPU cost handling
        if i == 5: # On step 5, try to select the NaN configuration
            print("\n--- INTENTIONALLY TESTING NaN GPU COST CONFIGURATION ---")
            # This corresponds to: img_size_bucket_idx=2 (1024px), batch_size_option_idx=1 (batch size 2)
            # which has NaN in PEAK_MEMORY_GB_BY_INDEX
            random_action["dim_bucket_logits"] = np.array([-10.0, -10.0, 10.0]) # Force argmax to choose index 2
            random_action["batch_size_logits"] = np.array([-10.0, 10.0, -10.0])# Force argmax to choose index 1

        print(f"\n--- Step {env.current_step + 1} (Env Internal Step: {env.current_step}) ---")
        
        server_state_before_step = obs['server_state'].copy()
        gpu_baseline_frac_before = server_state_before_step[1]
        active_gpu_reserved_before = sum(t['gpu_gb_reserved'] for t in env.active_processing_tasks)
        usable_gpu_before_step = max(0, gpu_baseline_frac_before * config.GPU_TOTAL_CAPACITY_GB - active_gpu_reserved_before)

        print(f"Server State Before: {server_state_before_step} (GPU Baseline Frac: {gpu_baseline_frac_before:.2f})")
        print(f"Active GPU Reserved Before: {active_gpu_reserved_before:.2f} GB. Usable GPU for this step: {usable_gpu_before_step:.2f} GB")
        print(f"Connections available this step: {int(server_state_before_step[3])}")
        print(f"Requests in actual queue: {len(env.actual_request_queue)}")

        next_obs, reward, terminated, truncated, info = env.step(random_action)
        total_reward += reward
        
        print(f"Agent Choice: ImgSize {info['target_img_size_agent']}, BatchSize {info['target_batch_size_agent']}")
        print(f"  Intended Batch GPU Cost (from config): {env._get_batch_gpu_consumption_gb(np.argmax(random_action['dim_bucket_logits']), np.argmax(random_action['batch_size_logits'])):.2f} GB")
        print(f"Actual Batch Processed Size: {info['chosen_batch_size']}")
        if info['chosen_batch_size'] > 0:
            print(f"  New Batch GPU Reserved: {info['gpu_gb_consumed_by_new_batch']:.2f} GB")
        print(f"Reward for this step: {reward:.3f}")
        print(f"Server State After (Baseline): {next_obs['server_state']} (GPU Baseline Frac: {next_obs['server_state'][1]:.2f})")
        print(f"Active Tasks After: {info['active_tasks_count']}, Total GPU Reserved: {info['total_gpu_reserved_active_gb']:.2f} GB")
        
        obs = next_obs
        if terminated or truncated:
            print(f"\nEpisode ended at step {env.current_step}. Terminated: {terminated}, Truncated: {truncated}")
            print(f"Total reward for episode: {total_reward:.3f}")
            break
    
    config = config_backup # Restore original config
    print("\n--- Dummy Config Restored ---")
