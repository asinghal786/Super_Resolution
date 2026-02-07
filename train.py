import os
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm  # For progress bar
from dataset import get_data_loader  # Ensure this import is correct
from model import MySuperResolutionModel  # Your model definition

# Hyperparameters
num_epochs = 50  # Set the desired number of epochs
learning_rate = 0.001
batch_size = 16  # Adjust as necessary

# Initialize Model
model = MySuperResolutionModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Load Data
train_loader = get_data_loader(batch_size)

# Load checkpoint if available
checkpoint_path = 'checkpoints/latest_checkpoint.pth'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f'Resuming training from epoch {start_epoch}')
else:
    start_epoch = 0
    print('No checkpoint found. Starting training from scratch.')

# Training Loop
try:
    for epoch in range(start_epoch, num_epochs):
        model.train()
        # Use tqdm to show a progress bar
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
            for batch_idx, (lr_images, hr_images) in enumerate(train_loader):
                # Resize lr_images to match the desired input size for your model
                lr_images = nn.functional.interpolate(lr_images, size=(1024, 1024), mode='bilinear', align_corners=False)

                optimizer.zero_grad()
                outputs = model(lr_images)

                # Ensure outputs and hr_images are the same size
                loss = criterion(outputs, hr_images)
                loss.backward()
                optimizer.step()

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

                if batch_idx % 10 == 0:  # Print every 10 batches
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # Save checkpoint after each epoch
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)

except KeyboardInterrupt:
    print("\nTraining interrupted. Saving current state...")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print("Checkpoint saved. You can resume training from the last epoch.")

print('Training finished!')