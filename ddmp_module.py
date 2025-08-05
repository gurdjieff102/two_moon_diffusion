import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

def sample_two_moons(n_samples=2000):
    data, _ = make_moons(n_samples=n_samples, noise=0.05)
    return torch.tensor(data, dtype=torch.float32)

class MLPDenoiser(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
    def forward(self, x, t):
        t_emb = t.unsqueeze(1)  # [batch, 1]
        x_in = torch.cat([x, t_emb], dim=1)  # [batch, 3]
        return self.net(x_in)

def ddpm_schedule(start, end, timesteps=100):
    betas = torch.linspace(start, end, timesteps)
    alphas = 1. - betas
    alpha_hat = torch.cumprod(alphas, dim=0)
    sqrt_alpha_hat = torch.sqrt(alpha_hat)
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat)

    return {
        'sqrt_alpha_hat': sqrt_alpha_hat,
        'sqrt_one_minus_alpha_hat': sqrt_one_minus_alpha_hat,
        'betas': betas,
        'alphas': alphas,
        'alpha_hat': alpha_hat,
    } 

class DiffusionModel(pl.LightningModule):
    def __init__(self, timesteps=100, lr=1e-3, batch_size=1024):
        super().__init__()
        self.model = MLPDenoiser()
        self.timesteps = timesteps
        self.lr = lr
        self.batch_size = batch_size
        for k, v in ddpm_schedule(1e-4, 0.02, timesteps).items():
            self.register_buffer(k, v)

    def forward(self, x_0, t):
        noise = torch.randn_like(x_0)
        sqrt_alpha_hat_t = self.sqrt_alpha_hat[t].unsqueeze(1)
        sqrt_one_minus_alpha_hat_t = self.sqrt_one_minus_alpha_hat[t].unsqueeze(1)
        x_t = sqrt_alpha_hat_t * x_0 + sqrt_one_minus_alpha_hat_t * noise
        return x_t, noise

    def training_step(self, batch, batch_idx):
        x_0, = batch
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device)
        x_t, noise = self(x_0, t)
        noise_pred = self.model(x_t, t.float() / self.timesteps)
        loss = F.mse_loss(noise_pred, noise)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def train_dataloader(self):
        data = sample_two_moons(2000)
        dataset = torch.utils.data.TensorDataset(data)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @torch.no_grad()
    def sample(self, n_samples):
        x_t = torch.randn(n_samples, 2).to(self.device)
        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((n_samples,), t, device=self.device, dtype=torch.float32)
            pred_noise = self.model(x_t, t_tensor / self.timesteps)
            beta_t = self.betas[t].to(self.device)
            alpha_t = self.alphas[t].to(self.device)
            sqrt_one_minus_alpha_hat_t = self.sqrt_one_minus_alpha_hat[t].to(self.device)
            x_t = (1. / torch.sqrt(alpha_t)) * (x_t - beta_t / sqrt_one_minus_alpha_hat_t * pred_noise)
            if t > 0:
                x_t += torch.randn_like(x_t) * beta_t.sqrt()
        return x_t.cpu()

def compute_mmd(x, y, sigma=1.0):
    """Compute Maximum Mean Discrepancy (MMD) with RBF kernel"""
    xx = torch.mm(x, x.t())
    yy = torch.mm(y, y.t())
    xy = torch.mm(x, y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx
    dyy = ry.t() + ry - 2. * yy
    dxy = rx.t() + ry - 2. * xy

    kxx = torch.exp(-dxx / (2. * sigma**2))
    kyy = torch.exp(-dyy / (2. * sigma**2))
    kxy = torch.exp(-dxy / (2. * sigma**2))

    return kxx.mean() + kyy.mean() - 2. * kxy.mean()

def plot_results(real, fake, filename="two_moons_result.png"):
    plt.figure(figsize=(8, 4))
    
    plt.subplot(1, 2, 1)
    plt.title("Real Two Moons")
    plt.scatter(real[:, 0], real[:, 1], s=5)
    plt.axis("equal")

    plt.subplot(1, 2, 2)
    plt.title("Generated Samples")
    plt.scatter(fake[:, 0], fake[:, 1], s=5, color="red")
    plt.axis("equal")

    plt.tight_layout()
    plt.savefig(filename)  
    plt.show()

if __name__ == "__main__":
    pl.seed_everything(0)
    model = DiffusionModel(timesteps=100, lr=1e-3)

    trainer = pl.Trainer(max_epochs=2000,
                        accelerator='gpu' if torch.cuda.is_available else 'cpu',
                        log_every_n_steps=10)
    trainer.fit(model)

    real_data = sample_two_moons(1000)
    fake_data = model.sample(1000)

    # Compute and print MMD
    mmd_score = compute_mmd(real_data, fake_data)
    print(f"\n✅ MMD between real and generated samples: {mmd_score:.6f}\n")
    # Plot results
    plot_results(real_data, fake_data)
