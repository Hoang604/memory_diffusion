import torch
import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class MLPBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.1, t_dim=32):
        super(MLPBlock, self).__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, output_dim * 2),
            nn.Mish()   
        )

        self.fc1 = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x, t_emb):
        x = F.mish(self.bn(self.fc1(x)))
        t_emb = self.time_mlp(t_emb)
        scale, shift = t_emb.chunk(2, dim=-1)
        x = x * (scale + 1) + shift
        x = self.drop(x)
        x = F.mish(self.bn2(self.fc2(x)))
        x = self.drop(x)
        return x

class MLPUNet(nn.Module):
    def __init__(self, input_dim, output_dim, intermediate_dim=[256, 128, 64, 32], dropout_rate=0.1):
        super(MLPUNet, self).__init__()

        hidden_dims = intermediate_dim

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.Mish(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[0], hidden_dims[0]),
            nn.Mish(),
            nn.Dropout(dropout_rate)
        )

        time_dim = intermediate_dim[2]
        self.time_pos_emb = SinusoidalPosEmb(time_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim)
        )

        self.number_of_blocks = len(hidden_dims)

        # --- Encoder Path ---
        encoders = nn.ModuleList()
        for i in range(self.number_of_blocks):
            is_not_last = i < self.number_of_blocks - 1
            encoders.append(nn.ModuleList([MLPBlock(hidden_dims[i], hidden_dims[i], dropout_rate, t_dim=time_dim),
                            nn.Linear(hidden_dims[i], hidden_dims[i+1]) if is_not_last else nn.Identity()]))
        self.encoders = encoders

        # --- Bottleneck ---
        self.bottleneck1 = MLPBlock(hidden_dims[-1], hidden_dims[-1], dropout_rate, t_dim=time_dim)
        self.bottleneck2 = MLPBlock(hidden_dims[-1], hidden_dims[-1], dropout_rate, t_dim=time_dim)

        # --- Decoder Path ---
        decoders = nn.ModuleList()
        for i in range(self.number_of_blocks):
            is_not_last = i < self.number_of_blocks - 1
            decoders.append(nn.ModuleList([MLPBlock(hidden_dims[-(i+1)] * 2, hidden_dims[-(i+1)], dropout_rate, t_dim=time_dim),
                            nn.Linear(hidden_dims[-(i+1)], hidden_dims[-(i+2)]) if is_not_last else nn.Identity()]))
        self.decoders = decoders

        # --- Output Layer ---
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[0]),
            nn.Mish(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[0], output_dim)
        )

    def forward(self, x, t):
        # --- Positional Encoding ---
        t_emb = self.time_pos_emb(t)
        t_emb = self.mlp(t_emb)
        # --- Encoder Path ---
        skip_connections = []
        x = self.input_proj(x)
        for mlp, down in self.encoders:
            x = mlp(x, t_emb)
            skip_connections.append(x)
            if down is not None:
                x = down(x)

        x = self.bottleneck1(x, t_emb)
        x = self.bottleneck2(x, t_emb)

        # --- Decoder Path ---
        for mlp, up in self.decoders:
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = mlp(x, t_emb)
            if up is not None:
                x = up(x)
        x = self.output_proj(x)
        return x

# --- Ví dụ sử dụng ---
if __name__ == "__main__":
    INPUT_DIM = 100  # Ví dụ: ảnh MNIST được làm phẳng (28*28)
    BATCH_SIZE = 16

    # Tạo mô hình
    model = MLPUNet(input_dim=INPUT_DIM, output_dim=INPUT_DIM, dropout_rate=0.25)
    x = model(torch.randn(BATCH_SIZE, INPUT_DIM), torch.randn(BATCH_SIZE,))
    print(f"Output shape: {x.shape}")
    summary(model, input_size=[(BATCH_SIZE, INPUT_DIM), (BATCH_SIZE,)], col_names=["input_size", "output_size", "num_params", "mult_adds"])
