import torch, einops, cv2, glob, os
import numpy as np
from torch.utils.data import Dataset

def count_frames(video_path):
    """Count the number of frames in a video."""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def calculate_segments(num_frames, segment_length):
    """Calculate the number of segments for a given number of frames and segment length."""
    return num_frames

def total_segments(vids_path, num_videos, segment_length):
    total_segments = 0
    for i in range(num_videos):
        video_path = os.path.join(vids_path, f'{i}.mp4')  # Assuming video names are like 0.mp4, 1.mp4, etc.
        frames = count_frames(video_path)
        segments = calculate_segments(frames, segment_length)
        total_segments += segments
    return total_segments

def make_seg_dic(vids_path, num_videos, segment_length):
    return {str(i): calculate_segments(count_frames(os.path.join(vids_path, f'{i}.mp4')), segment_length) 
            for i in range(num_videos)}

def make_labels_dic(txt_path):
    with open(txt_path, 'r') as f:
        labels = [line.split(':') for line in f.read().strip().split('\n')]

    for idx, label in enumerate(labels):
        labels[idx] = [int(frame_num) if frame_num.isdigit() else frame_num for frame_num in label]

    return {str(label[-1]): label[:-1] for label in labels}

def map_global_idx_to_local_idx(global_index, num_videos, seg_dic):
    cumulative_segments = 0
    for video_index in range(num_videos):
        num_segments = seg_dic[str(video_index)]
        if cumulative_segments + num_segments > global_index:
            segment_index = global_index - cumulative_segments
            return str(video_index), int(segment_index)
        cumulative_segments += num_segments 
    raise IndexError("Global index out of range")

def classify_frame(vid_num, frame_num, labels_dic):
    label = labels_dic[vid_num]
    default_return = 0  # Set the default return value
    if int(frame_num) < int(label[0]):
        return default_return

    # Iterate over the label in pairs (start, stop)
    for i in range(0, len(label), 2):
        start = int(label[i])
        try:
            stop = int(label[i+2])
        except IndexError:
            stop = float('inf')  # Handle case where no end label is present

        if start <= frame_num <= stop:
            return 1 if i % 4 == 0 else 0

    return default_return

def remove_z_pad(frame):
    x = frame.shape[0]
    y = frame.shape[1]
    dpadim = frame
    idx = []
    #get rid of x padding
    for i in range(x):
        row = frame[i,:,:]
        if(np.average(row)==0):
            idx.append(i)
    dpadim = np.delete(dpadim,idx,0)
    
    idx = []
    #get rid of y padding
    for j in range(y):
        col = frame[:,j,:]
        if(np.average(col)==0):
            idx.append(j)
    
    dpadim = np.delete(dpadim,idx,1)

    #delete row ignore by range function
    x2 = dpadim.shape[0]
    if((x-x2)>1):
        dpadim = np.delete(dpadim,0,0)
    
    return dpadim

def get_frame_sequence(video_path, frame_number, depth, size, n_channels):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Calculate the start frame
    start_frame = max(frame_number - depth + 1, 0)

    # Set the video position to the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Read frames from start_frame to frame_number
    frames = []
    if frame_number < depth:
        for _ in range(depth-frame_number):
            frames.append(np.zeros((size, size, n_channels)))
    for _ in range(min(depth, frame_number)):
        ret, frame = cap.read()
        if not ret:
            break  # Break if the video ends before reaching the target frame
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = remove_z_pad(frame)
        frame = cv2.resize(frame, (size, size))
        frames.append(frame)
        
    # Release the video capture object
    cap.release()
    
    frames = einops.rearrange(frames, 'f h w c -> f h w c')     #Turns list into numpy array
    frames_tensor = torch.from_numpy(frames)

    return frames_tensor

class Custom_Traffic_Dataset(Dataset):
    def __init__(self, depth, size, n_channels, vids_path, labels_path):
        self.depth = depth
        self.size = size
        self.n_channels = n_channels
        self.vids_path = vids_path
        self.num_vids =  len(glob.glob(os.path.join(self.vids_path, '*.mp4')))
        self.labels_dic = make_labels_dic(labels_path)
        self.seg_dic = make_seg_dic(self.vids_path, self.num_vids, self.depth)
        
    def __len__(self):
        return total_segments(self.vids_path, self.num_vids, self.depth)
    
    def __getitem__(self, idx):
        vid_num, frame_num = map_global_idx_to_local_idx(idx, self.num_vids, self.seg_dic)
        frame_class = classify_frame(vid_num, frame_num, self.labels_dic)
        frame_class = torch.tensor(frame_class)
        vid_path = os.path.join(self.vids_path, f'{vid_num}.mp4')
        vid_tensor = get_frame_sequence(vid_path, frame_num, self.depth, self.size, self.n_channels)
        
        return vid_tensor.float(), frame_class