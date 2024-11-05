# Standard Imports
import torch

from torch import nn
from torch.utils.data import random_split, Subset

# Custom Imports
from train_class import Trainer
from factenc_vivit import ViVit_2
from dataset_class import Custom_Traffic_Dataset

def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Dataset Parameters
    image_size = 384              # Height and width of frame
    tube_hw = 32            # height and width of tubelet, frame size must be divisible by this number
    tube_d = 12           # depth of tubelet, i.e. number of frames back 
    n_channels = 1          # R,G,B -> Gray
    latent_size = 1024       # Size of embedding,
    batch_size = 8        # batch size
    subst = False
    subst_ratio = 0.00011
    
    if n_channels == 3:
        mean= [0.41433686, 0.41618344, 0.4186155 ]  # Calculated from dataset
        std= [0.1953987,  0.19625916, 0.19745596]   # Calculated from dataset
    else:
        mean = [0.41735464]
        std = [0.1976403]
    
    
    # Model Parameters 
    num_class = 2           # Num of classes
    num_heads = 16          # Number of attention heads in a single encoder
    num_spatial_encoders = 24       # Number of encoders in model
    num_temporal_encoders = 4       # Number of encoders in model
    interval = 32

    #Training Parameters     
    epochs = 2000             # Number of iterations through entirety of Data
    base_lr = 1e-3          # learning rate
    restart = 5000          # Number of steps before resetting lr to base_lr
    weight_decay = 0        # Weight Decay
    dropout = 0             # Dropout
    loss_steps = 25
    val_steps = 250

    #File Management Parameters
    model_num = 25
    load_from_checkpoint = False
    load_checkpoint_path = f'./model25/checkpoint_step2.pth'
    load_pt_checkpoint_path = f'./model25/checkpoint.pth'
    out_checkpoint_path = f'./model{model_num}/checkpoint_step2.pth'
    best_checkpoint_path = f'./model{model_num}/best_step2_checkpoint.pth'
    
    # Models
    print('Loading Model...')
    model = ViVit_2(num_spatial_encoders, num_temporal_encoders, latent_size, device, num_heads, num_class, dropout, tube_hw, tube_d, n_channels, batch_size, image_size)
    # Filter out the weights that don't match ViVit_2 model
    print("Loading Pretrained Weights...")
    model_dict = model.state_dict()
    checkpoint = torch.load(load_pt_checkpoint_path, map_location=device)

    # Only keep the keys that match between both models
    pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict and 'MLP_adapter' not in k}

    # Update the state_dict of ViVit_2
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # Freeze the spatial_encStack layers in ViVit_2
    for param in model.spatial_encStack.parameters():
        param.requires_grad = False

    print('Loading Dataset...')
    train_data = Custom_Traffic_Dataset(tube_d, image_size, n_channels, "/data/ryan_thesis_ds/thesis_videos", '/data/ryan_thesis_ds/crash_frame_numbers.txt', mean, std, interval)
    generator1 = torch.Generator().manual_seed(42)
    train_data, test_data, val_data = random_split(train_data, [0.64, 0.2, 0.16], generator=generator1)
    datasets = {'train': train_data, 'test': test_data, 'val': val_data}
    
    if subst:
        # Calculate the number of samples for the smaller training set (e.g., 10% of train_data)
        small_train_size = int(subst_ratio * len(train_data))

        # Create a subset of the train_data
        small_train_data = Subset(train_data, torch.arange(small_train_size))

        # Replace the original train_data in the datasets dictionary
        datasets['train'] = small_train_data
    
    print('Initializing Dataloader...')
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size,
                                            shuffle=False, num_workers=4, drop_last=True) for x in ['train','test', 'val']}

    print('Setting up training parameters...')
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    class_weights = torch.tensor([(1/0.47)/((1/0.47) + (1/0.53)), (1/0.53)/((1/0.47) + (1/0.53))]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor = 1, total_iters = 1)
    
    # Create Trainer instance
    trainer = Trainer(model, model_num, dataloaders, datasets, criterion, optimizer, scheduler, device, out_checkpoint_path, load_checkpoint_path, load_from_checkpoint)
    
    # Train the model
    metrics = trainer.train(epochs=epochs, loss_steps=loss_steps, val_steps=val_steps, batch_size=batch_size)

if __name__ == '__main__':
    main()