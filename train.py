# Standard Imports
import torch
import numpy as np

from torch import nn
from transformers import ViTModel
from torch.utils.data import random_split, Subset

# Custom Imports
from train_class import Trainer
from factenc_vivit import ViVit, SemiCon_ViVit
from train_utils import load_vit_weights
from dataset_class import Custom_Traffic_Dataset

def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    SEED = 4000
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator = torch.Generator().manual_seed(SEED)
    
    # Dataset Parameters
    image_size = 384              # Height and width of frame
    tube_hw = 32            # height and width of tubelet, frame size must be divisible by this number
    latent_size = 1024       # Size of embedding,
    batch_size = 10        # batch size
    
    subst = False
    subst_ratio = 0.001
    #subst_ratio = 0.00011
    
    # Model Parameters 
    num_class = 2           # Num of classes
    num_heads = 16          # Number of attention heads in a single encoder
    num_spatial_encoders = 24       # Number of encoders in model
    
    #Training Parameters     
    epochs = 2000             # Number of iterations through entirety of Data
    max_lr = 2e-4          # learning rate
    min_lr = 1e-5
    accumulated_steps = 40 # number of forward passes before updating weights (Effective batch size = batch_size * accumulated_steps)
    #TODO: Reapply regularization
    weight_decay = 0        # Weight Decay
    dropout = 0           # Dropout
    val_steps = 100

    #File Management Parameters
    model_num = 1
    load_from_checkpoint = False
    load_checkpoint_path = f'./model28/checkpoint.pth'
    out_checkpoint_path = f'./results/models/model{model_num}/checkpoint.pth'
    best_checkpoint_path = f'./results/models/model{model_num}/best_checkpoint.pth'
    
    ###Testing Params
    interval = 32
    tube_d = 8             # depth of tubelet, i.e. number of frames back 
    num_temporal_encoders = 4       # Number of encoders in model
    n_channels = 1          # R,G,B -> Gray
    
    if n_channels == 3:
        mean= [0.41433686, 0.41618344, 0.4186155 ]  # Calculated from dataset
        std= [0.1953987,  0.19625916, 0.19745596]   # Calculated from dataset
    else:
        mean = [0.41735464]
        std = [0.1976403]
    
    # Models
    print('Loading Model...')
    #--depth = 2
    model = ViVit(num_spatial_encoders, num_temporal_encoders, latent_size, device, num_heads, num_class, dropout, tube_hw, tube_d, n_channels, batch_size, image_size)
    vit = ViTModel.from_pretrained('google/vit-large-patch32-384')

    # Load ViT weights into spatial encoder
    model_dict = load_vit_weights(model.state_dict(), vit.state_dict(), latent_size, SEED)
    model.load_state_dict(model_dict, strict=True)
    del vit, model_dict
    print('Loaded weights successfully')

    print('Loading Dataset...')
    train_data = Custom_Traffic_Dataset(tube_d, image_size, n_channels, "/data/ryan_thesis_ds/thesis_videos", '/data/ryan_thesis_ds/crash_frame_numbers.txt', mean, std, interval)
    train_data, test_data, val_data = random_split(train_data, [0.7, 0.2, 0.1], generator=generator)
    datasets = {'train': train_data, 'test': test_data, 'val': val_data}
    
    if subst:
        # Calculate the number of samples for the smaller training set (e.g., 10% of train_data)
        small_train_size = int(subst_ratio * len(train_data))

        # Create a subset of the train_data
        small_train_data = Subset(train_data, torch.arange(small_train_size))

        # Replace the original train_data in the datasets dictionary
        datasets['train'] = small_train_data
    
    print('Initializing Dataloader...')
    dataloaders = {x: torch.utils.data.DataLoader(
        datasets[x],
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=4,
        pin_memory=True,
        persistent_workers = False,
        drop_last=True,
        worker_init_fn=lambda worker_id: np.random.seed(SEED + worker_id)
        ) for x in ['train','test', 'val']}

    print('Setting up training parameters...')
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    #optimizer = ADOPT(model.parameters(), lr = max_lr, decoupled = True)
    
    weights = torch.tensor([1/0.47, 1/0.53])
    class_weights = weights / weights.sum()
    class_weights = class_weights.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0= 2,
        eta_min = min_lr
    )
    
    iters = len(dataloaders['train']) // accumulated_steps + (1 if len(dataloaders['train']) % accumulated_steps != 0 else 0)
    
    # Create Trainer instance
    trainer = Trainer(model, model_num, dataloaders, criterion, optimizer, scheduler, device, out_checkpoint_path, load_checkpoint_path, best_checkpoint_path, load_from_checkpoint, accumulated_steps, iters)
    
    # Train the model
    metrics = trainer.train(epochs=epochs, batch_size=batch_size, val_steps=val_steps)
    

if __name__ == '__main__':
    main()