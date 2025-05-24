import torch
import torch.nn as nn

class DoubleCritic(nn.Module):
    """
    Core network for double critic evaluation.
    It takes the state and action as input and outputs two Q-value.
    """
    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_dim=256
    ):
        super(DoubleCritic, self).__init__()
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish()
        )
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish()
        )
        self.q1_net = nn.Sequential(
                    nn.Linear(2 * hidden_dim, 2 * hidden_dim),
                    nn.Mish(),
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    nn.Mish(),
                    nn.Linear(hidden_dim, 1))
        
        self.q2_net = nn.Sequential(
                    nn.Linear(2 * hidden_dim, 2 * hidden_dim),
                    nn.Mish(),
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    nn.Mish(),
                    nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        processed_state = self.state_mlp(state)
        processed_action = self.action_mlp(action)
        # Concatenate state and action
        x = torch.cat([processed_state, processed_action], dim=-1)
        return self.q1_net(x), self.q2_net(x)

    def q_min(self, obs, action):
        return torch.min(*self.forward(obs, action))