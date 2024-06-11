from torchvision import transforms
import torch
import numpy as np
from tqdm import tqdm
from dataset_class import Custom_Traffic_Dataset

size = 512              # Height and width of frame
tube_hw = 16            # height and width of tubelet, frame size must be divisible by this number
tube_d = 16              # depth of tubelet, i.e. number of frames back 
n_channels = 3  

def calculate_mean_std(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=6, shuffle=False, num_workers=6)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for inputs, _ in tqdm(loader):
        for i in range(3):  # Assuming 3 channels (RGB)
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean /= len(loader)
    std /= len(loader)
    return mean, std

# Create a dataset instance without normalization
untransformed_dataset = Custom_Traffic_Dataset(tube_d, size, n_channels, "/data/ryan_thesis_ds/thesis_videos", '/data/ryan_thesis_ds/crash_frame_numbers.txt', mean=None, std=None)

mean, std = calculate_mean_std(untransformed_dataset)
mean = mean.numpy()
std = std.numpy()
print(f'Mean: {mean}, Std: {std}')