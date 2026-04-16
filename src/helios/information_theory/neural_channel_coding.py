import torch
from torch import nn
import numpy as np

class ChannelAutoencoder(nn.Module):
    def __init__(self, k=4, n=7, hidden=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(k, hidden), nn.ReLU(), nn.Linear(hidden, n)
        )
        self.decoder = nn.Sequential(
            nn.Linear(n, hidden), nn.ReLU(), nn.Linear(hidden, k), nn.Sigmoid()
        )

    def forward(self, x):
        code = self.encoder(x)
        noisy = code + torch.randn_like(code) * 0.2  # AWGN channel
        return self.decoder(noisy)

def train_autoencoder(snr_db=4.0, epochs=100, device=None):
    k, n = 4, 7
    model = ChannelAutoencoder(k, n).to(device or 'cpu')
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.BCELoss()
    for _ in range(epochs):
        bits = torch.randint(0, 2, (256, k), dtype=torch.float32, device=device)
        out = model(bits)
        loss = loss_fn(out, bits)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model

def eval_ber(model, snr_db=4.0, device=None):
    k = 4
    with torch.no_grad():
        bits = torch.randint(0, 2, (1024, k), dtype=torch.float32, device=device)
        out = model(bits)
        decoded = (out > 0.5).float()
        ber = (decoded != bits).float().mean().item()
    return ber

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = train_autoencoder(device=device)
    ber = eval_ber(model, device=device)
    print(f"Neural channel code BER: {ber:.4f}")
