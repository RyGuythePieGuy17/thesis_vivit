from collections import Counter
from dataset_class import Custom_Traffic_Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch

tube_d = 4              # depth of tubelet, i.e. number of frames back 
n_channels = 3     
size = 224   

dataset = Custom_Traffic_Dataset(tube_d, size, n_channels, "/data/ryan_thesis_ds/thesis_videos", '/data/ryan_thesis_ds/crash_frame_numbers.txt')

batch_size = 26
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16)


# Initialize a counter to count the occurrences of each class
class_counts = Counter()
wrong_sizes = []
# Iterate through the DataLoader and count each class
for data in tqdm(dataloader):
    if data.shape != torch.Size([batch_size, tube_d, size, size, n_channels]):
        wrong_sizes.append(data.shape)
        print(data.shape)
    #class_counts.update(targets.numpy())
print(wrong_sizes)
    
# Print the class counts
#print("Class distribution:", class_counts)