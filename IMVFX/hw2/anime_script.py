import math
import random
import numpy as np
# import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
import torch.nn.functional as F
from torch.utils.data import DataLoader
# import torchvision
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Lambda, Grayscale, Resize
import os
import requests, time
from utils import generate_new_images

SEND_INTERVAL = 300
webhook_url = "https://discord.com/api/webhooks/1351429559639212123/91nmrQ6FGn-K-_FXWPwrU4HtZhOUt90f3VmxhnREqXy3QQUfUEjZFrtrWJkaNUNZo_Fb"

workspace_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(workspace_dir)
print(f"Current working directory: {workspace_dir}")
# Root directory for the MNIST dataset
dataset_path = f"{workspace_dir}/animefacedataset"
# The path to save the model
model_store_path = f"{workspace_dir}/anime.pt"

def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def cosine_beta_schedule(timesteps, s=0.01):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-5, 0.1)

# Create the time embedding
def time_embedding(n, d):
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:,::2] = torch.sin(t * wk[:,::2])
    embedding[:,1::2] = torch.cos(t * wk[:,::2])
    return embedding

def time_embedding_v2(n, d):
    embedding = torch.zeros(n, d)
    position = torch.arange(n).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d, 2).float() * -(math.log(10000.0) / d))
    embedding[:, 0::2] = torch.sin(position * div_term)
    embedding[:, 1::2] = torch.cos(position * div_term)
    return embedding

class AttentionBlock(nn.Module):
    """Standard Self-Attention block for diffusion models."""
    def __init__(self, channels):
        super().__init__()
        g = 8 if channels % 8 == 0 else 1
        self.norm = nn.GroupNorm(g, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        norm_x = self.norm(x)
        # Generate Queries, Keys, and Values
        qkv = self.qkv(norm_x).view(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # Compute Attention scores
        attn = torch.einsum("bcn,bcm->bnm", q, k) * (C ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        # Apply Attention to Values
        out = torch.einsum("bnm,bcm->bcn", attn, v)
        out = out.view(B, C, H, W)
        
        # Residual connection
        return x + self.proj(out)

class TimeAwareBlock(nn.Module):
    def __init__(self, cin: int, cout: int, time_emb_dim: int):
        super().__init__()
        
        # Helpers
        g_in = 8 if cin % 8 == 0 else 1
        g_out = 8 if cout % 8 == 0 else 1
        
        self.norm1 = nn.GroupNorm(g_in, cin)
        self.conv1 = nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1)
        
        self.temb_proj = nn.Linear(time_emb_dim, cout)
        
        self.norm2 = nn.GroupNorm(g_out, cout)
        self.conv2 = nn.Conv2d(cout, cout, kernel_size=3, stride=1, padding=1)

    def forward(self, x, te):
        # Step 1: Norm -> SiLU -> Conv
        h = self.conv1(F.silu(self.norm1(x)))
        
        # Step 2: Add time embedding AFTER normalization
        h = h + self.temb_proj(te).view(-1, h.shape[1], 1, 1)
        
        # Step 3: Norm -> SiLU -> Conv
        h = self.conv2(F.silu(self.norm2(h)))
        return h
    
class UNet_anime(nn.Module):
    """
    Resolution-agnostic diffusion noise predictor U-Net.
    """

    def __init__(self, n_steps=1000, time_embedding_dim=256, base_channels=64):
        super().__init__()

        self.n_steps = n_steps
        self.time_embedding_dim = time_embedding_dim
        ch = base_channels

        self.time_step_embedding = nn.Embedding(n_steps, time_embedding_dim)
        self.time_step_embedding.weight.data = time_embedding(n_steps, time_embedding_dim)
        self.time_step_embedding.requires_grad_(False)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim)
        )

        # Initial Conv (No time embedding here!)
        self.init_conv = nn.Conv2d(3, ch, kernel_size=3, padding=1)

        # --- Encoder ---
        self.block1 = TimeAwareBlock(ch, ch, time_embedding_dim)
        self.down1 = nn.Conv2d(ch, ch, 4, 2, 1)

        self.block2 = TimeAwareBlock(ch, 2 * ch, time_embedding_dim)
        self.down2 = nn.Conv2d(2 * ch, 2 * ch, 4, 2, 1)

        self.block3 = TimeAwareBlock(2 * ch, 4 * ch, time_embedding_dim)
        self.down3 = nn.Sequential(
            nn.Conv2d(4 * ch, 4 * ch, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )

        # --- Bottleneck ---
        self.block_mid1 = TimeAwareBlock(4 * ch, 4 * ch, time_embedding_dim)
        self.attn_mid = AttentionBlock(4 * ch)
        self.block_mid2 = TimeAwareBlock(4 * ch, 4 * ch, time_embedding_dim)

        # --- Decoder ---
        self.up1 = TimeAwareBlock(8 * ch, 4 * ch, time_embedding_dim)
        self.block4 = TimeAwareBlock(4 * ch, 2 * ch, time_embedding_dim)
        self.up_conv1 = nn.ConvTranspose2d(2 * ch, 2 * ch, 4, 2, 1)

        self.up2 = TimeAwareBlock(4 * ch, 2 * ch, time_embedding_dim)
        self.block5 = TimeAwareBlock(2 * ch, ch, time_embedding_dim)
        self.up_conv2 = nn.ConvTranspose2d(ch, ch, 4, 2, 1)

        self.up3 = TimeAwareBlock(2 * ch, ch, time_embedding_dim)
        self.block6 = TimeAwareBlock(ch, ch, time_embedding_dim)

        self.final_layer = nn.Conv2d(ch, 3, 3, 1, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = self.time_mlp(self.time_step_embedding(t))

        # Initial Conv
        x0 = self.init_conv(x)

        # stage 1
        o1 = self.block1(x0, te)
        d1 = self.down1(o1)

        # stage 2
        o2 = self.block2(d1, te)
        d2 = self.down2(o2)

        # stage 3
        o3 = self.block3(d2, te)
        d3 = self.down3(o3)

        # bottleneck
        mid = self.block_mid1(d3, te)
        mid = self.attn_mid(mid)
        mid = self.block_mid2(mid, te)

        # up path + skips
        u1 = F.interpolate(mid, size=o3.shape[2:], mode='nearest')
        x4 = torch.cat([o3, u1], dim=1) 
        o4 = self.block4(self.up1(x4, te), te)
        u2 = self.up_conv1(o4)

        x5 = torch.cat([o2, u2], dim=1) 
        o5 = self.block5(self.up2(x5, te), te)
        u3 = self.up_conv2(o5)

        x6 = torch.cat([o1, u3], dim=1) 
        o6 = self.block6(self.up3(x6, te), te)

        return self.final_layer(o6)
    
# Define the class of DDPM
class DDPM_anime(nn.Module):
    def __init__(self, image_shape=(3, 48, 48), n_steps=1000, start_beta=1e-4, end_beta=0.02, device=None):
        super(DDPM_anime, self).__init__()
        self.device = device
        self.image_shape = image_shape
        self.n_steps = n_steps
        self.noise_predictor = UNet_anime(n_steps).to(device)
        self.betas = cosine_beta_schedule(n_steps).to(device)
        # self.betas = torch.linspace(start_beta, end_beta, n_steps).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    # Forward process
    # Add the noise to the images
    def forward(self, x0, t, eta=None):
        n, channel, height, width = x0.shape
        alpha_bar = self.alpha_bars[t]

        if eta is None:
            eta = torch.randn(n, channel, height, width).to(self.device)

        noise = alpha_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - alpha_bar).sqrt().reshape(n, 1, 1, 1) * eta
        return noise

    # Backward process
    # Predict the noise that was added to the images during the forward process
    def backward(self, x, t):
        return self.noise_predictor(x, t)

def send_notification(webhook_url, message="test321"):
    payload = {
        "content": message,
    }
    requests.post(webhook_url, json=payload)

def send_training_update(epoch, loss, eta_str):
    payload = {
        "content": (
            f"# **Training Update**\n"
            f"Epoch: {epoch+1}/{n_epochs}\n"
            f"Loss: {loss:.4f}\n"
            f"ETA: {eta_str}"
        )
    }
    requests.post(webhook_url, json=payload)

def trainer(ddpm, dataloader, n_epochs, optim, loss_function, device, model_store_path):

    best_loss = float("inf")
    n_steps = ddpm.n_steps
    outer_tqdm = tqdm(range(n_epochs), desc=f"Training progress", colour="green", ncols=80)
    color_ch = dataloader.dataset[0][0].shape[0]
    send_notification(webhook_url, f"Training started!\nModel: UNet_anime\nEpochs: {n_epochs}\nLearning Rate: {lr}\nSteps: {n_steps}")
    
    for epoch in outer_tqdm:
        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(dataloader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour="blue", ncols=80)):
            # Load data
            x0 = batch[0].to(device)
            n = len(x0)

            # Pick random noise for each of the images in the batch
            eta = torch.randn_like(x0).to(device)
            t = torch.randint(0, n_steps, (n,)).to(device)

            # Compute the noisy image based on x0 and the time step
            noises = ddpm(x0, t, eta)

            # Get model estimation of noise based on the images and the time step
            eta_theta = ddpm.backward(noises, t)

            # Optimize the Mean Squared Error (MSE) between the injected noise and the predicted noise
            loss = loss_function(eta_theta, eta)

            # First, initialize the optimizer's gradient and then update the network's weights
            optim.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), max_norm=1.0)  # Gradient clipping
            
            optim.step()

            # Aggregate the loss values from each iteration to compute the loss value for an epoch
            epoch_loss += loss.item() * len(x0) / len(dataloader.dataset)

            # Save Losses for plotting later
            loss_list.append(loss.item())

        log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}"
        outer_tqdm.set_postfix_str(f"Loss: {epoch_loss:.4f}")
        # Show images generated at the epoch
        if epoch % 3 == 0 or epoch == n_epochs - 1:
            generate_new_images(ddpm, device=device, channel=color_ch, height=x0.shape[2], width=x0.shape[3]), f"Images generated at epoch {epoch + 1}"

        # If the current loss is better than the previous one, then store the model
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(ddpm.state_dict(), model_store_path)
            log_string += " <Store the best model.>"

        print(log_string)

    send_notification(webhook_url, f"Training finished!\n"+log_string)

if __name__ == "__main__":
    same_seeds(42)
    batch_size = 128
    lr = 1e-4
    n_epochs = 20

    transform = Compose([
        Resize((48, 48)),
        ToTensor(),
        Lambda(lambda x: x * 2 - 1)  # Scale to [-1, 1]
    ])

    dataset = ImageFolder(root=dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    ddpm_anime = DDPM_anime(n_steps=1000, start_beta=1e-4, end_beta=0.02, device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    trainer(ddpm_anime, dataloader, n_epochs=n_epochs, optim=AdamW(ddpm_anime.parameters(), lr, weight_decay=1e-4), loss_function=nn.MSELoss(), device=ddpm_anime.device, model_store_path=model_store_path)