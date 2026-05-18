from utils import show_forward
from anime_script import DDPM_anime,  same_seeds
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Lambda, Grayscale, Resize
import matplotlib.pyplot as plt

if __name__ == "__main__":
    dataset_path = "/ssd/jasonzzz/NYCU_2026_spring/IMVFX/hw2/animefacedataset/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    torch.manual_seed(42); 
    ddpm_cosine = DDPM_anime(n_steps=1000, start_beta=1e-4, end_beta=0.02, device=device); 
    show_forward(ddpm_cosine, dataloader, device); plt.savefig('cosine_beta.png'); 
    ddpm_linear = DDPM_anime(n_steps=1000, start_beta=1e-4, end_beta=0.02, device=device); 
    ddpm_linear.betas = torch.linspace(1e-4, 0.02, 1000).to(device); 
    ddpm_linear.alphas = 1 - ddpm_linear.betas; 
    ddpm_linear.alpha_bars = torch.cumprod(ddpm_linear.alphas, dim=0); 
    show_forward(ddpm_linear, dataloader, device); plt.savefig('linear_beta.png')