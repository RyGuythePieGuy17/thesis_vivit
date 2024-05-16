import torch, einops, torchvision, cv2, glob, os

import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from dataset_class import Custom_Traffic_Dataset
from vivit import ViVit
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomCrop

def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Model Parameters
    size = 224              # Height and width of frame
    tube_hw = 16            # height and width of tubelet, frame size must be divisible by this number
    tube_d = 4              # depth of tubelet, i.e. number of frames back 
    n_channels = 3          # R,G,B
    latent_size = 768       # Size of embedding, 
    num_class = 2           # Num of classes
    num_heads = 12          # Number of attention heads in a single encoder
    num_encoders = 12       # Number of encoders in model

    #Training Parameters
    dropout = 0.1           # Dropout
    epochs = 10             # Number of iterations through entirety of Data
    base_lr = 10e-3         # learning rate
    weight_decay = 0.03     # Weight Decay
    batch_size = 8          # batch size
    
    print('Loading Model...')
    model = ViVit(num_encoders, latent_size, device, num_heads, num_class, dropout, tube_hw, tube_d, n_channels, batch_size)
    print('Loading Dataset...')
    train_data = Custom_Traffic_Dataset(tube_d, size, n_channels, "/data/thesis_videos", '/data/crash_frame_numbers.txt')
    print('Initializing Dataloader...')
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                            shuffle=True, num_workers=8)

    print('Setting up training parameters...')
    # Betas used for Adam in paper are 0.9 and 0.999, which are the default in PyTorch
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.LinearLR(optimizer)


    # Training Loop
    model.train().to(device)
    print(f'Starting Training on Device: {device}...')
    for epoch in tqdm(range(epochs), total=epochs):
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs=model(inputs)
            loss = criterion(outputs,targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx%200 == 0:
                print('Batch {} epoch {} has loss = {}'.format(batch_idx, epoch, running_loss/200))
                running_loss = 0
    scheduler.step()


if __name__=='__main__':
    main()