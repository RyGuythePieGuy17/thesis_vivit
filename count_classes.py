from collections import Counter
from dataset_class import Custom_Traffic_Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader

tube_d = 4              # depth of tubelet, i.e. number of frames back 
n_channels = 3     
size = 224   

dataset = Custom_Traffic_Dataset(tube_d, size, n_channels, "/data/thesis_videos", '/data/crash_frame_numbers.txt')

batch_size = 16
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16)


# Initialize a counter to count the occurrences of each class
class_counts = Counter()

# Iterate through the DataLoader and count each class
for _, targets in tqdm(dataloader):
    class_counts.update(targets.numpy())
    
# Print the class counts
print("Class distribution:", class_counts)