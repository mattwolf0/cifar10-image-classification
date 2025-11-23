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
    tf_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4), 
         transforms.RandomHorizontalFlip(), 
         transforms.ToTensor(), 
         transforms.Normalize(MEAN, STD),]
         )
    
    tf_test = transforms.Compose(
        [transforms.ToTensor(), 
         transforms.Normalize(MEAN, STD),]
         )
    
    train_ds = datasets.CIFAR10(
        root=data_root, 
        train=True, download=True, 
        transform=tf_train
        )
    
    test_ds = datasets.CIFAR10(
        root=data_root, 
        train=False, 
        download=True, 
        transform=tf_test
        )
    
    train_ld = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=workers, 
        pin_memory=True
        )
    
    test_ld = DataLoader(
        test_ds, 
        batch_size=max(64, batch_size * 2), 
        shuffle=False, 
        num_workers=workers, 
        pin_memory=True
        )

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

def show_samples(model, loader, device, num_images=8):
    model.eval()
    model.to(device)

    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)

    images_shown = 0
    rows = 2
    cols = (num_images + rows - 1) // rows

    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    axes = axes.flatten()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            preds = logits.argmax(dim=1)

            for i in range(x.size(0)):
                if images_shown >= num_images:
                    plt.tight_layout()
                    return

                img = x[i].cpu()

                img = img * std + mean
                img = img.clamp(0.0, 1.0)

                #[C,H,W] -> [H,W,C]
                img_np = img.numpy().transpose(1, 2, 0)

                ax = axes[images_shown]
                ax.imshow(img_np)
                ax.axis("off")

                true_label = CLASSES[y[i].item()]
                pred_label = CLASSES[preds[i].item()]
                ax.set_title(f"pred: {pred_label}\ntrue: {true_label}", fontsize=8)

                images_shown += 1

        plt.tight_layout()


def main():
    set_seed(42)

    device = torch.device("cuda")

    train_loader, test_loader = get_loaders(data_root="./data", batch_size=128, workers=2)

    model = SmallCNN(input_channels=3, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    num_epochs = 40

    for epoch in range(1, num_epochs + 1):

        train_loss, train_acc = run_epoch(
            model, 
            train_loader, 
            criterion, 
            optimizer=optimizer, device=device
            )

        test_loss, test_acc = run_epoch(
            model, 
            test_loader, 
            criterion, 
            optimizer=None, 
            device=device
            )

        print(f"epoch {epoch:02d} train_acc={train_acc:.3f} loss={train_loss:.3f}/{test_loss:.3f}")

        #if train_acc == 0.800:
        #    break

    show_samples(model, test_loader, device, num_images=8)
    plt.show()



if __name__ == "__main__":
    main()