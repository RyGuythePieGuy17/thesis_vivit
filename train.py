import torch, matplotlib, os
#matplotlib.use('TkAgg')

import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score
from torch import nn
from tqdm import tqdm
from torch.utils.data import random_split
from dataset_class import Custom_Traffic_Dataset
from vivit import ViVit

def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Model Parameters
    size = 512              # Height and width of frame
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
    base_lr = 3e-3         # learning rate
    weight_decay = 0.03     # Weight Decay
    batch_size = 26         # batch size
    loss_steps = 10
    val_steps = 200
    
    print('Loading Model...')
    model = ViVit(num_encoders, latent_size, device, num_heads, num_class, dropout, tube_hw, tube_d, n_channels, batch_size)
    
    print('Loading Dataset...')
    train_data = Custom_Traffic_Dataset(tube_d, size, n_channels, "/data/ryan_thesis_ds/thesis_videos", '/data/ryan_thesis_ds/crash_frame_numbers.txt')
    generator1 = torch.Generator().manual_seed(42)      # Set seed so train/test/validate splits remain the same throughout all experiments
    train_data, test_data, val_data = random_split(train_data, [0.64, 0.2, 0.16], generator=generator1)
    datasets = {'train': train_data, 'test': test_data, 'val': val_data}
    
    print('Initializing Dataloader...')
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size,
                                            shuffle=True, num_workers=8, drop_last = True) for x in ['train','test', 'val']}

    print('Setting up training parameters...')
    # Betas used for Adam in paper are 0.9 and 0.999, which are the default in PyTorch
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.LinearLR(optimizer)
    
    # This is to ensure plt.show() is called only once
    fig, ax = plt.subplots(2, 2, figsize=(10, 16))
    
    def update_plots(metrics):
        nonlocal fig, ax

        # Clear the axes to avoid overlapping plots
        for row in ax:
            for a in row:
                a.clear()
        
        # Training Loss
        if metrics['train_losses']:
            steps, losses = zip(*metrics['train_losses'])
            ax[0,0].plot(steps, losses, label='Training Loss')
            ax[0,0].set_title('Training Loss')
            ax[0,0].set_xlabel('Step')
            ax[0,0].set_ylabel('Loss')
            ax[0,0].legend()
        
        # Training Accuracy
        if metrics['train_accs']:
            steps, accs = zip(*[(step, acc.cpu().numpy()) for step, acc in metrics['train_accs']])
            ax[0,1].plot(steps, accs, label='Training Accuracy')
            ax[0,1].set_title('Training Accuracy')
            ax[0,1].set_xlabel('Step')
            ax[0,1].set_ylabel('Accuracy')
            ax[0,1].legend()
        
        # Validation Loss
        if metrics['val_losses']:
            steps, losses = zip(*metrics['val_losses'])
            ax[1,0].plot(steps, losses, label='Validation Loss')
            ax[1,0].set_title('Validation Loss')
            ax[1,0].set_xlabel('Step')
            ax[1,0].set_ylabel('Loss')
            ax[1,0].legend()
        
        # Validation Accuracy
        if metrics['val_accs']:
            steps, accs = zip(*[(step, acc.cpu().numpy()) for step, acc in metrics['val_accs']])
            ax[1,1].plot(steps, accs, label='Validation Accuracy')
            ax[1,1].set_title('Validation Accuracy')
            ax[1,1].set_xlabel('Step')
            ax[1,1].set_ylabel('Accuracy')
            ax[1,1].legend()

        # Save the figure to a file
        plt.draw()
        plt.savefig('./plot_test2.png')
        

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
        
        
    # Training Loop
    def train_model(model, dataloaders, datasets, criterion, optimizer, scheduler, device, epochs, loss_steps, val_steps, batch_size):
        model.train().to(device)
        metrics = {
            'train_losses': [],
            'train_accs': [],
            'val_losses': [],
            'val_accs': [],
            'val_precisions': [],
            'val_recalls': []
        }
        best_val_recall = 0.0
        checkpoint_path = './model2/checkpoint.pth'
        best_checkpoint_path = './model2/best_checkpoint.pth'
        

        print(f'Starting Training on Device: {device}...')
        for epoch in tqdm(range(epochs), total=epochs):
            running_loss = 0.0
            running_corrects = 0
            for batch_idx, (inputs, targets) in enumerate(tqdm(dataloaders['train'], desc=f'Training Loop {epoch}')):
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad() # reset gradients for proper backwards pass 
                
                #forward pass
                outputs=model(inputs)
                _, preds = torch.max(outputs,1)
                loss = criterion(outputs,targets)
                
                #backward pass
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                running_corrects += torch.sum(preds == targets.data)
                
                if (batch_idx+1)%loss_steps == 0:
                    step = len(datasets['train'])*epoch + batch_idx
                    acc = running_corrects.double()/(loss_steps*batch_size)
                    loss_avg = running_loss/loss_steps
                    print_metrics(epoch, batch_idx, loss_avg, acc, None, None)
                    
                    metrics['train_losses'].append((step, loss_avg))
                    metrics['train_accs'].append((step, acc))
                    
                    running_loss = 0
                    running_corrects = 0
                    
                    # Update plots after each training loss step
                    update_plots(metrics)
                    
                if (batch_idx+1)%val_steps == 0:
                    print('Starting Validation...')
                    val_loss, val_acc, val_precision, val_recall = evaluate_model(model, dataloaders['val'], criterion, device)
                    step = len(datasets['train']) * epoch + batch_idx

                    metrics['val_losses'].append((step, val_loss))
                    metrics['val_accs'].append((step, val_acc))
                    metrics['val_precisions'].append((step, val_precision))
                    metrics['val_recalls'].append((step, val_recall))
                    
                    print_metrics(epoch, batch_idx, val_loss, val_acc, val_precision, val_recall, phase='Validation')
                    
                    # Save the most recent checkpoint
                    checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': val_loss,
                    }
                    
                    save_checkpoint(checkpoint, checkpoint_path)
                    # Save the best checkpoint based on validation recall
                    if val_recall > best_val_recall:
                        best_val_recall = val_recall
                        save_checkpoint(checkpoint, best_checkpoint_path)
                    
                    # Update plots after each training loss step
                    update_plots(metrics)
                    
                    model.train()
                    
            scheduler.step()

        return metrics
    
    metrics = train_model(model, dataloaders, datasets, criterion, optimizer, scheduler, device, epochs, loss_steps, val_steps, batch_size)


if __name__=='__main__':
    main()