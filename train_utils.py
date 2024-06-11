import torch, os
from tqdm import tqdm

#Eval function for readability 
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0
    running_corrects = 0
    TP=TN=FP=FN=0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc='Evaluation Loop'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            running_corrects += torch.sum(preds == targets.data)
            # Calculate TP, TN, FP, FN
            TP += ((preds == 1) & (targets == 1)).sum().item()
            TN += ((preds == 0) & (targets == 0)).sum().item()
            FP += ((preds == 1) & (targets == 0)).sum().item()
            FN += ((preds == 0) & (targets == 1)).sum().item()
    
    avg_loss = running_loss / len(dataloader)
    accuracy = running_corrects.double() / len(dataloader.dataset)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    return avg_loss, accuracy, precision, recall


#Print Metrics function for readability
def print_metrics(epoch, batch_idx, loss_avg, acc, precision, recall, phase='Training'):
    if phase == 'Validation':
        print(f'{phase} - Epoch: {epoch + 1} Batch: {batch_idx +1} Loss: {loss_avg:.4f} Accuracy: {acc:.4f} Precision: {precision:.4f} Recall: {recall:.4f}')
    else:
        print(f'{phase} - Epoch: {epoch + 1} Batch: {batch_idx +1} Loss: {loss_avg:.4f} Accuracy: {acc:.4f}')


def save_checkpoint(state, filename):
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    torch.save(state, filename)
    
def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)  # Move optimizer state to the device
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        start_batch_idx = checkpoint['batch_idx'] if 'batch_idx' in checkpoint else 0  # Load the last batch index if available
        print(f"Loaded checkpoint '{checkpoint_path}' (epoch: {checkpoint['epoch']}, batch: {start_batch_idx})")
        return start_epoch, checkpoint['loss'], start_batch_idx
    else:
        print(f"No checkpoint found at '{checkpoint_path}'")
        return 0, None