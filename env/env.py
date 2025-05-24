import numpy as np
import gymnasium as gym # Import gymnasium
from gymnasium import spaces

class Env(gym.Env): # Inherit from gym.Env
    metadata = {'render_modes': ['human'], 'render_fps': 30} # Optional metadata

    def __init__(self):
        """
        Initializes the environment, defining observation and action spaces.
        _current_observation stores the current state of the environment.
        """
        super().__init__() # Call the constructor of the parent class (gym.Env)

        # Define the observation space
        self._observation_space = spaces.Dict({
            'server_state': spaces.Box(
                low=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), # gpu, vram, bandwidth, connections
                high=np.array([1.0, 1.0, 1.0, 20.0], dtype=np.float32),# gpu, vram, bandwidth, connections
                shape=(4,), # Four features in a single array
                dtype=np.float32
            ),
            'request_list': spaces.Box(
                low=np.zeros((20, 3), dtype=np.float32), # Min values for features
                high=np.array(
                    [[200.0, 1.0, 1024.0]] * 20, # Max token (200), max speed, max img_size
                    dtype=np.float32
                ), # Shape: (number_of_requests, features_per_request)
                dtype=np.float32
            )
        })

        # Define the action space
        # The actor needs to output 20 priority scores (one for each potential request slot),
        # each score between 0.0 and 1.0.
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(20,), # Corresponds to the 20 requests in request_list
            dtype=np.float32
        )

        # Initialize the current observation by generating an initial state.
        # Note: self._current_observation is for internal tracking.
        # The official way to get an observation is via reset() or step().
        self._current_observation = self._generate_new_random_state()

    def _generate_new_random_state(self):
        """
        Generates a new random state for the environment.

        The state is composed of server resource information and a list of pending requests.
        This method is called to produce a new state, typically during reset or stepping.

        Returns:
            dict: A dictionary containing:
                'server_state' (np.ndarray): An array representing server resources.
                'request_list' (np.ndarray): A 2D array representing pending requests.
        """
        # Server state: A NumPy array representing server resources.
        server_resources = np.array([
            self.np_random.random(),             # gpu_available, use self.np_random for reproducibility
            self.np_random.random(),             # vram_available
            self.np_random.random(),             # bandwidth_available
            float(self.np_random.integers(1, 21)) # number_of_connection_available (1 to 20)
        ], dtype=np.float32)

        # Request list (20 requests): A 2D NumPy array representing pending requests.
        request_data = np.full((20, 3), [200.0, 0.0, 1024.0], dtype=np.float32)
        
        number_of_actual_requests = self.np_random.integers(0, 21) 
        possible_token_count  = np.arange(10, 201, 10) 
        possible_img_size = np.array([128, 256, 512, 1024])

        for i in range(number_of_actual_requests):
            request_data[i, 0] = self.np_random.choice(possible_token_count)  # token_count
            request_data[i, 1] = self.np_random.random()                       # internet_speed 
            request_data[i, 2] = self.np_random.choice(possible_img_size)    # img_size
        
        new_state_dict = {
            'server_state': server_resources,
            'request_list': request_data
        }
        return new_state_dict

    def calculate_benefit(self, request_list_state, priorities):
        """
        Calculates the simulated benefit based on request properties and assigned priorities.
        This is a helper function and n xot part of the standard gym.Env API.
        """
        total_benefit = 0.0
        w_token = 0.5
        w_img_size = 0.3
        w_speed = 0.2

        max_token = 200.0 
        max_img_size = 1024.0
        max_speed = 1.0

        for i in range(request_list_state.shape[0]):
            token_count = request_list_state[i, 0]
            internet_speed = request_list_state[i, 1]
            img_size = request_list_state[i, 2]
            priority = priorities[i]

            if token_count > 0.001: 
                norm_token = token_count / max_token if max_token > 0 else 0
                norm_img_size = img_size / max_img_size if max_img_size > 0 else 0
                norm_speed = internet_speed / max_speed if max_speed > 0 else 0
                
                request_benefit = priority * (w_token * norm_token + \
                                              w_img_size * norm_img_size + \
                                              w_speed * norm_speed)
                total_benefit += request_benefit
        return total_benefit

    def step(self, action):
        """
        Executes one time step within the environment.
        The agent takes an action, and the environment transitions to a new state.
        """
        priorities = action

        reward = self.calculate_benefit(self._current_observation['request_list'], priorities)

        self._current_observation = self._generate_new_random_state()
        next_observation = self._current_observation

        terminated = False 
        truncated = False  
        info = {"message": "Step executed successfully."}

        return next_observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state and returns the initial observation.
        """
        super().reset(seed=seed) # Important for reproducibility and setting self.np_random

        self._current_observation = self._generate_new_random_state()
        initial_observation = self._current_observation
        info = {"message": "Environment reset."}
        return initial_observation, info

    @property
    def observation_space(self):
        """Returns the observation space of the environment."""
        return self._observation_space
    
    @property
    def current_observation(self):
        """Returns the current internal observation of the environment."""
        return self._current_observation


# Example usage (optional, for testing):
if __name__ == '__main__':
    # You can use Gymnasium's environment checker for more thorough validation
    from gymnasium.utils.env_checker import check_env
    env = Env()
    check_env(env) # This will raise an error if the environment is not compliant

    print("Observation Space:")
    print(env.observation_space)
    print("\nAction Space:")
    print(env.action_space)

    # Get an initial observation 
    obs, info = env.reset(seed=42) 
    print(f"\nInfo from reset: {info}")
    print("Initial Observation (Request List Sample from env.reset):")
    print(obs['request_list'][:5]) 

    random_action = env.action_space.sample()
    print("\nSample Random Action (Priorities):")
    print(random_action)

    next_obs, reward, terminated, truncated, info = env.step(random_action)
    print("\nAfter taking a step with the random action:")
    print(f"Reward received: {reward:.4f}")
    print("Next Observation (Request List Sample):")
    print(next_obs['request_list'][:5])
    print(f"Terminated: {terminated}, Truncated: {truncated}")
    print(f"Info from step: {info}")

    env.close() # Clean up the environment

    print("\n--- Direct Benefit Calculation Test (still valid for testing logic) ---")
    sample_requests = np.zeros((20, 3), dtype=np.float32)
    sample_requests[0] = [100.0, 0.8, 512.0]
    sample_requests[1] = [50.0, 0.5, 256.0]
    sample_requests[2] = [200.0, 0.0, 1024.0] 

    sample_priorities = np.zeros(20, dtype=np.float32)
    sample_priorities[0] = 0.9 
    sample_priorities[1] = 0.4
    sample_priorities[2] = 0.5 

    calculated_benefit = env.calculate_benefit(sample_requests, sample_priorities)
    print(f"Calculated benefit for sample requests and priorities: {calculated_benefit:.4f}")
