import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, transform=None, hr_size=(512, 512), lr_size=(256, 256)):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.hr_images = sorted(os.listdir(hr_dir))
        self.lr_images = sorted(os.listdir(lr_dir))
        self.transform = transform
        self.hr_size = hr_size
        self.lr_size = lr_size

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        hr_image_path = os.path.join(self.hr_dir, self.hr_images[idx])
        lr_image_path = os.path.join(self.lr_dir, self.lr_images[idx])

        hr_image = Image.open(hr_image_path).convert("RGB").resize(self.hr_size)
        lr_image = Image.open(lr_image_path).convert("RGB").resize(self.lr_size)

        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)

        return lr_image, hr_image

# Function to create DataLoader
def get_data_loader(hr_dir, lr_dir, batch_size, shuffle=True, num_workers=0):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = ImageDataset(hr_dir=hr_dir, lr_dir=lr_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader
