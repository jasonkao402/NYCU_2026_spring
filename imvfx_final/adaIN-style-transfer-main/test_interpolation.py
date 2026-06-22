import torch
import torchvision.utils as utils
from StyleTransfer import StyleTransfer
from PIL import Image
from ImageDataset import ImageDataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from ImageDataset import DeNormalize
import os
import time
if __name__ == '__main__':
    _dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(_dir_path)

    model = StyleTransfer()
    model.decoder.load_state_dict(torch.load("models/best_model.pth", map_location='cuda:0'))

    model.training = False
    model.eval()

    num_images = 4
    content = ImageDataset(flag='content', root_dir='./test_set/content', data_range=(0,num_images))
    
    # Load TWO distinct style datasets for interpolation
    style_A = ImageDataset(flag='style', root_dir='./test_set/style_A', data_range=(0,1))
    style_B = ImageDataset(flag='style', root_dir='./test_set/style_B', data_range=(0,1))
    
    content_img = DataLoader(dataset=content, batch_size=1, shuffle=False)
    style_img_A = DataLoader(dataset=style_A, batch_size=1, shuffle=False)
    style_img_B = DataLoader(dataset=style_B, batch_size=1, shuffle=False)
    print(len(content_img), len(style_img_A), len(style_img_B))
    denormalizer = DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Slider weight: 0.5 means a 50/50 mix of Style A and Style B
    num_steps = 7
    alpha_weight = np.linspace(0, 1, num=num_steps) 

    i = 0
    
    for content_batch, style_batch_A, style_batch_B in zip(content_img, style_img_A, style_img_B):
        _start = time.time()
        with torch.no_grad():
            # 1. Run the heavy extraction ONCE per image set
            model.cache_adain_features(content_batch, style_batch_A, style_batch_B)

            grid_images = []

            # 2. Quickly loop through the alpha slider values
            for j, alpha in enumerate(alpha_weight):
                # Fast forward pass using cached tensors
                decoded = model.decode_from_cache(alpha.item())
                
                saved = decoded.clone().detach()
                saved = denormalizer(saved)
                
                # Clamp values to [0, 1] to ensure make_grid handles colors properly
                saved = torch.clamp(saved, 0, 1)
                # filename = f"test_set/results/interp_img_{j:02d}.png"
                # utils.save_image(saved, filename)
                # Append to our list of images
                grid_images.append(saved)

        # 3. Concatenate all step images along the batch dimension (dim=0)
        # Resulting shape will be (num_steps, 3, H, W)
        grid_tensor = torch.cat(grid_images, dim=0)
        _end = time.time()
        print(f"Processed interpolation for image {i} in {_end - _start:.2f} seconds.")
        # 4. Stitch them into a single image grid side-by-side
        # nrow=num_steps means all steps will be in a single horizontal row
        grid = utils.make_grid(grid_tensor, nrow=num_steps, padding=2, normalize=False)

        filename = f"test_set/results/grid_interp_{i:02d}.png"
        utils.save_image(grid, filename)
        print(f"Saved {filename} with {num_steps} interpolation steps.")
        i += 1