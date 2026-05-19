# Import packages
import numpy as np
import cv2
import einops
import imageio
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Lambda, Grayscale, Resize
import os
import time
import requests, time
SEND_INTERVAL = 300
webhook_url = "https://discord.com/api/webhooks/1351429559639212123/91nmrQ6FGn-K-_FXWPwrU4HtZhOUt90f3VmxhnREqXy3QQUfUEjZFrtrWJkaNUNZo_Fb"

def send_notification(webhook_url, message="test321"):
    payload = {
        "content": message,
    }
    requests.post(webhook_url, json=payload)

abs_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(abs_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


img = cv2.imread("animefacedataset/images/10000_2004.jpg")
# cv2.imwrite("copy.jpg", img)
print(f"Image shape: {img.shape}")

total_n = 100
t_test = tqdm(total=total_n, desc="Testing")
start = int(time.time())
print(f"Start time: {time.ctime(start)} ({start})")
prepare_random = np.random.uniform(0.01, 0.3, size=(total_n,))

# Define a moving average window size
MOVING_AVERAGE_WINDOW = 10
recent_times = []

for i in range(total_n):
    info = t_test.format_dict
    current_time = time.time()

    # Calculate the time taken for the current iteration
    if i > 0:
        iteration_time = current_time - last_time
        recent_times.append(iteration_time)
        if len(recent_times) > MOVING_AVERAGE_WINDOW:
            recent_times.pop(0)

    # Calculate the moving average rate
    rate = sum(recent_times) / len(recent_times) if recent_times else 0.001

    eta = current_time + (info["total"] - info["n"]) * rate
    t_test.set_description(f"Testing (ETA: {time.ctime(eta)}) Rate: {rate:.2f}s/iter")

    t_test.update()
    last_time = current_time
    time.sleep(prepare_random[i])

end = int(time.time())
print(f"End time: {time.ctime(end)} ({end})")