from multihead import SelfAttention
from torch import nn

class TBlock(nn.Module):
    def __init__(self, k, heads):
        super().__init__()

        self.attention = SelfAttention(k, heads)

        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        self.ff = nn.Sequential(
            nn.Linear(k, 4 * k),  # 4 can be reduced to 2 or 1, if we want to save memory
            nn.ReLU(),
            nn.Linear(4 * k, k))
    
    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)
        fedforward = self.ff(x)

        return self.norm2(fedforward + x)