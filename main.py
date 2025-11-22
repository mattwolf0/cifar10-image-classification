import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from models import *
import random


MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_loaders(data_root="./data", batch_size=128, workers=2):
    tf_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(MEAN, STD),])
    tf_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD),])
    train_ds = datasets.CIFAR10(root=data_root, train=True, download=True, transform=tf_train)
    test_ds = datasets.CIFAR10(root=data_root, train=False, download=True, transform=tf_test)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    test_ld = DataLoader(test_ds, batch_size=max(64, batch_size * 2), shuffle=False, num_workers=workers, pin_memory=True)

    return train_ld, test_ld

def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):

    train_mode = optimizer is not None
    if train_mode:
        model.train()
    else:
        model.eval()

    total = 0
    correct = 0
    loss_sum = 0.0

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            if train_mode:
                optimizer.zero_grad()

            logits = model(x) #[B, 10] logit
            loss = criterion(logits, y) #CE loss

            if train_mode:
                loss.backward()
                optimizer.step()

            loss_sum += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = loss_sum / total
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    set_seed(42)


if __name__ == "__main__":
    main()