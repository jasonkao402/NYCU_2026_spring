import numpy as np
import torch
# import torch.nn as nn
import cv2
import einops
import imageio
"""
Provided with a DDPM model, a specified number of samples to generate, and a chosen device,
this function returns a set of freshly generated samples while also saving the .gif of the reverse process
"""
def generate_new_images(ddpm, n_samples=16, device=None, frames_per_gif=25, gif_name="sampling.gif", channel=1, height=28, width=28):
    frame_idxs = np.linspace(0, ddpm.n_steps, frames_per_gif).astype(np.uint)
    frames = []

    with torch.no_grad():
        if device is None:
            device = ddpm.device

        # Starting from random noise
        x = torch.randn(n_samples, channel, height, width).to(device)

        for idx, t in enumerate(list(range(ddpm.n_steps))[::-1]):
            # Estimating noise to be removed
            # time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
            time_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
            eta_theta = ddpm.backward(x, time_tensor)

            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]

            # Get alpha_t_bar for the previous timestep (handle t=0 carefully)
            alpha_t_bar_prev = ddpm.alpha_bars[t-1] if t > 0 else torch.tensor(1.0, device=device)
            beta_t = ddpm.betas[t]

            # 1. Predict the fully denoised image (x_0) from the predicted noise
            pred_x0 = (x - (1 - alpha_t_bar).sqrt() * eta_theta) / alpha_t_bar.sqrt()

            # 2. THE FIX: Clamp the predicted x_0 to prevent explosions!
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

            # 3. Calculate the posterior mean to step back to x_{t-1}
            # Using Eq 7 from the original DDPM paper based on x_0
            coef1 = (beta_t * alpha_t_bar_prev.sqrt()) / (1 - alpha_t_bar)
            coef2 = ((1 - alpha_t_bar_prev) * alpha_t.sqrt()) / (1 - alpha_t_bar)
            mean = coef1 * pred_x0 + coef2 * x

            # 4. Add variance (noise)
            if t > 0:
                z = torch.randn(n_samples, channel, height, width).to(device)
                x = mean + beta_t.sqrt() * z
            else:
                x = mean

            # Adding frames to the GIF
            if idx in frame_idxs or t == 0:
                # Putting digits in range [0, 255]
                normalized = x.clone()
                # for i in range(len(normalized)):
                #     normalized[i] -= torch.min(normalized[i])
                #     normalized[i] *= 255 / torch.max(normalized[i])
                
                # 1. Clamp to [-1, 1] to catch any slight deviations
                normalized = normalized.clamp(-1.0, 1.0) 
                
                # 2. Shift to [0, 1]
                normalized = (normalized + 1.0) / 2.0 
                
                # 3. Scale to [0, 255]
                normalized = normalized * 255.0
                
                # Reshaping batch (n, c, h, w) to be a square frame
                frame = einops.rearrange(normalized, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=int(n_samples ** 0.5))
                frame = frame.cpu().numpy().astype(np.uint8)

                # Rendering frame
                frames.append(frame)

    if channel == 1:
        for i in range(len(frames)):
            frames[i] = cv2.cvtColor(frames[i], cv2.COLOR_GRAY2RGB)
    
    # Storing the gif
    with imageio.get_writer(gif_name, mode="I") as writer:
        for idx, frame in enumerate(frames):
            writer.append_data(frame)
            if idx == len(frames) - 1:
                for _ in range(frames_per_gif // 3):
                    writer.append_data(frames[-1])
    return x

