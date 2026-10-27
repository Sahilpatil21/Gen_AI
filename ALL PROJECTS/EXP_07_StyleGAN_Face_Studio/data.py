import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from datasets import load_dataset

import config


def build_transform(img_size=config.IMG_SIZE):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1, 1] to match tanh output
    ])


class CelebADataset(Dataset):
    def __init__(self, hf_ds, transform):
        self.hf_ds = hf_ds
        self.transform = transform

    def __len__(self):
        return len(self.hf_ds)

    def __getitem__(self, idx):
        img = self.hf_ds[idx]["image"].convert("RGB")
        return self.transform(img)


def get_dataloader():
    print("Downloading CelebA faces (first run may take a few minutes)...")
    hf_dataset = load_dataset(config.DATASET_NAME, split="train")
    n = min(config.NUM_IMAGES, len(hf_dataset))
    hf_dataset = hf_dataset.select(range(n))
    print("Loaded", len(hf_dataset), "images")
    dataset = CelebADataset(hf_dataset, build_transform())
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                      num_workers=config.NUM_WORKERS, drop_last=True)
