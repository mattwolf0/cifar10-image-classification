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


def main():
    return 0


if __name__ == "__main__":
    main()