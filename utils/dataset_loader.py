from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def get_dataloader(data_dir, image_size=64, batch_size=128):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5 for _ in range(3)], [0.5 for _ in range(3)]),
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return loader
