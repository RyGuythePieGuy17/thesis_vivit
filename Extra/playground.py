# Save Model From Checkpoint
# checkpoint = torch.load(checkpoint_path)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
# start_epoch = checkpoint['epoch']
# loss = checkpoint['loss']

out_checkpoint_path = f'./model24/checkpoint_step2.pth'

print(out_checkpoint_path.split('/')[1])