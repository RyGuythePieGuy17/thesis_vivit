import matplotlib.pyplot as plt
import torch, os
from dataset_class import Custom_Traffic_Dataset

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Model Parameters
image_size = 384              # Height and width of frame
tube_hw = 32            # height and width of tubelet, frame size must be divisible by this number
tube_d = 105             # depth of tubelet, i.e. number of frames back 
n_channels = 1          # R,G,B -> Gray
latent_size = 1024       # Size of embedding, 
num_class = 2           # Num of classes
num_heads = 16          # Number of attention heads in a single encoder
num_spatial_encoders = 24       # Number of encoders in model
num_temporal_encoders = 4       # Number of encoders in model
interval = 34

#Training Parameters
dropout = 0.3           # Dropout
epochs = 20             # Number of iterations through entirety of Data
base_lr = 5e-5          # learning rate
weight_decay = 0.05     # Weight Decay
batch_size = 6        # batch size
loss_steps = 25
val_steps = 250

model_num = 17
load_from_checkpoint = False
if n_channels == 3:
    mean= [0.41433686, 0.41618344, 0.4186155 ]  # Calculated from dataset
    std= [0.1953987,  0.19625916, 0.19745596]   # Calculated from dataset
else:
    mean = [0.41735464]
    std = [0.1976403]

print('loading dataset....')
dataset = Custom_Traffic_Dataset(tube_d, image_size, n_channels, "/data/ryan_thesis_ds/thesis_videos", '/data/ryan_thesis_ds/crash_frame_numbers.txt', mean, std, interval)

# Function to display frames
def display_frames(frames, save_path='frames2.png', figsize_per_frame=(5, 5)):
    num_frames = frames.shape[0]
    fig, axes = plt.subplots(1, num_frames, figsize=(figsize_per_frame[0] * num_frames, figsize_per_frame[1]))
    if num_frames == 1:
        axes = [axes]  # Ensure axes is iterable even for a single frame
    
    for i, ax in enumerate(axes):
        if n_channels == 3:
            ax.imshow(frames[i].permute(1, 2, 0).numpy())  # Convert CHW to HWC for visualization
        else:
            ax.imshow(frames[i].numpy(), cmap='gray')
        ax.axis('off')
    
    plt.savefig(save_path)
    plt.close()

# Get an item from the dataset and display frames
idx = 10000  # Change this index to visualize different samples
frames, label = dataset[idx]

# Display the frames
print('pulling frames....')
display_frames(frames)
print(f'Label: {label.item()}')

