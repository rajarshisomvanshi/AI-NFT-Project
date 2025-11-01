import os
import shutil
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
import torchvision
from torchvision import transforms

# ======================================================================================
# PART 1: THE VAE ARCHITECTURE (THE COMPRESSOR) - (No changes needed here)
# ======================================================================================

class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        modules = []
        hidden_dims = [32, 64, 128, 256]
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(hidden_dims[-1]*16, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*16, latent_dim)

        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 16)
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3, stride=2,
                                       padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3, stride=2,
                                               padding=1, output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels=3,
                                      kernel_size=3, padding=1),
                            nn.Tanh())

    def encode(self, x):
        spatial_latent = self.encoder(x)
        result = torch.flatten(spatial_latent, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var, spatial_latent

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 256, 4, 4)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var, _ = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return reconstruction, mu, log_var

# ======================================================================================
# PART 2: THE U-NET ARCHITECTURE (THE DENOISER) - (No changes needed here)
# ======================================================================================

class TimeEmbedding(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
    def forward(self, t: torch.Tensor):
        half_dim = self.n_channels // 2
        exponents = torch.arange(half_dim, device=t.device).float() / (half_dim - 1)
        embeddings = torch.exp(-math.log(10000) * exponents)
        embeddings = t[:, None] * embeddings[None, :]
        return torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.time_emb = nn.Linear(time_channels, out_channels)
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        h = self.conv1(self.act1(self.norm1(x)))
        time_emb_proj = self.time_emb(self.act2(t))
        h = h + time_emb_proj[:, :, None, None]
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.shortcut(x)

class AttentionBlock(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, n_channels)
        self.qkv = nn.Conv2d(n_channels, n_channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(n_channels, n_channels, kernel_size=1)
    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        h_ = self.norm(x)
        qkv = h_.view(b, c, h * w)
        q, k, v = qkv.chunk(3, dim=1)
        attn = torch.einsum('bci,bcj->bij', q, k) * (c ** -0.5)
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bij,bcj->bci', attn, v)
        out = out.view(b, c, h, w)
        return x + self.proj_out(out)

class UNet(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, n_channels=320):
        super().__init__()
        time_emb_dim = n_channels * 4
        self.time_embedding = TimeEmbedding(time_emb_dim)
        self.inc = nn.Conv2d(in_channels, n_channels, kernel_size=3, padding=1)
        self.down1_res = ResidualBlock(n_channels, n_channels, time_emb_dim)
        self.down1_attn = AttentionBlock(n_channels)
        self.down2_conv = nn.Conv2d(n_channels, n_channels * 2, kernel_size=3, stride=2, padding=1)
        self.down2_res = ResidualBlock(n_channels * 2, n_channels * 2, time_emb_dim)
        self.down2_attn = AttentionBlock(n_channels * 2)
        self.bot_res1 = ResidualBlock(n_channels * 2, n_channels * 2, time_emb_dim)
        self.bot_attn = AttentionBlock(n_channels * 2)
        self.bot_res2 = ResidualBlock(n_channels * 2, n_channels * 2, time_emb_dim)
        self.up1_res = ResidualBlock(n_channels * 4, n_channels * 2, time_emb_dim)
        self.up1_attn = AttentionBlock(n_channels * 2)
        self.up1_conv_transpose = nn.ConvTranspose2d(n_channels * 2, n_channels, kernel_size=2, stride=2)
        self.up2_res = ResidualBlock(n_channels * 2, n_channels, time_emb_dim)
        self.up2_attn = AttentionBlock(n_channels)
        self.outc = nn.Sequential(nn.GroupNorm(32, n_channels), nn.SiLU(), nn.Conv2d(n_channels, out_channels, kernel_size=3, padding=1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        t_emb = self.time_embedding(t)
        x1 = self.inc(x)
        x2 = self.down1_res(x1, t_emb); x2 = self.down1_attn(x2)
        x3_conv = self.down2_conv(x2)
        x3 = self.down2_res(x3_conv, t_emb); x3 = self.down2_attn(x3)
        x_bot = self.bot_res1(x3, t_emb); x_bot = self.bot_attn(x_bot); x_bot = self.bot_res2(x_bot, t_emb)
        x = torch.cat([x_bot, x3], dim=1)
        x = self.up1_res(x, t_emb); x = self.up1_attn(x); x = self.up1_conv_transpose(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2_res(x, t_emb); x = self.up2_attn(x)
        return self.outc(x)

# ======================================================================================
# PART 3: THE DRIVER / CONTROLLER (THE LOGIC) - (Major changes here)
# ======================================================================================

class DiffusionTrainer:
    def __init__(self, unet_model, vae_model, timesteps=1000, device='cpu', lr=1e-4, epochs=50):
        self.device = device
        self.unet_model = unet_model.to(device)
        self.vae_model = vae_model.to(device)
        self.timesteps = timesteps

        for param in self.vae_model.parameters():
            param.requires_grad = False

        self.betas = self._linear_beta_schedule(timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)

        self.optimizer = torch.optim.AdamW(self.unet_model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))

    def _linear_beta_schedule(self, timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps)

    def _get_index_from_list(self, vals, t, x_shape):
        batch_size = t.shape[0]
        out = vals.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def train_step(self, real_images):
        self.optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(self.device == 'cuda')):
            _, _, x_0 = self.vae_model.encode(real_images)
            t = torch.randint(0, self.timesteps, (x_0.shape[0],), device=self.device).long()
            noise = torch.randn_like(x_0)
            alphas_cumprod_t = self._get_index_from_list(self.alphas_cumprod, t, x_0.shape)
            noisy_latent = torch.sqrt(alphas_cumprod_t) * x_0 + torch.sqrt(1. - alphas_cumprod_t) * noise
            predicted_noise = self.unet_model(noisy_latent, t)
            loss = self.criterion(noise, predicted_noise)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item()

    @torch.no_grad()
    def sample(self, num_images=4):
        latent_x = torch.randn((num_images, 256, 4, 4), device=self.device)

        for i in tqdm(reversed(range(0, self.timesteps)), desc="Sampling", total=self.timesteps):
            t = torch.full((num_images,), i, device=self.device, dtype=torch.long)
            predicted_noise = self.unet_model(latent_x, t)
            alpha_t = self._get_index_from_list(self.alphas, t, latent_x.shape)
            alphas_cumprod_t = self._get_index_from_list(self.alphas_cumprod, t, latent_x.shape)
            beta_t = self._get_index_from_list(self.betas, t, latent_x.shape)
            noise_term = ((1 - alpha_t) / torch.sqrt(1 - alphas_cumprod_t)) * predicted_noise
            latent_x = (1 / torch.sqrt(alpha_t)) * (latent_x - noise_term)
            if i > 0:
                z = torch.randn_like(latent_x)
                latent_x += torch.sqrt(beta_t) * z

        sampled_images = self.vae_model.decoder(latent_x)
        sampled_images = self.vae_model.final_layer(sampled_images)
        return sampled_images

# ======================================================================================
# PART 4: MAIN EXECUTION BLOCK (PUTTING IT ALL TOGETHER)
# ======================================================================================

if __name__ == '__main__':
    # --- Configuration ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 8
    IMG_SIZE = 64
    EPOCHS = 15
    LEARNING_RATE = 2e-5
    DATASET_PATH = "./datasets/"

    print(f"Using device: {DEVICE}")

    # --- Data Loading and Transformations ---
    transforms_ = torchvision.transforms.Compose([
        torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = torchvision.datasets.ImageFolder(root=DATASET_PATH, transform=transforms_)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    # --- Initialize Models ---
    vae = VAE(in_channels=3, latent_dim=128)
    VAE_ENCODER_OUTPUT_CHANNELS = 256
    unet = UNet(in_channels=VAE_ENCODER_OUTPUT_CHANNELS, out_channels=VAE_ENCODER_OUTPUT_CHANNELS)

    # --- Initialize the Driver/Controller ---
    trainer = DiffusionTrainer(unet, vae, timesteps=1000, device=DEVICE, lr=LEARNING_RATE, epochs=EPOCHS)

    # --- Training Loop ---
    for epoch in range(EPOCHS):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch_idx, (real_images, _) in enumerate(progress_bar):
            real_images = real_images.to(DEVICE)
            loss = trainer.train_step(real_images)
            total_loss += loss
            progress_bar.set_postfix({"Loss": f"{loss:.4f}", "LR": f"{trainer.scheduler.get_last_lr()[0]:.1e}"})

        trainer.scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

        checkpoint = {
            'epoch': epoch + 1,
            'unet_state_dict': trainer.unet_model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': trainer.scheduler.state_dict(),
            'loss': avg_loss,
        }
        torch.save(checkpoint, f"checkpoint_epoch_{epoch+1}.pth")
        print(f"Saved checkpoint to checkpoint_epoch_{epoch+1}.pth")

    # --- Sampling ---
    print("Training finished. Starting sampling...")
    sampled_imgs = trainer.sample(num_images=4)
    grid = torchvision.utils.make_grid(sampled_imgs, nrow=4, normalize=True)
    torchvision.utils.save_image(grid, "final_generated_sample.png")
    print("Saved final generated images to 'final_generated_sample.png'")
