import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from functools import partial
from models import *

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id, base_seed):
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_loaders(data_root="./data", batch_size=128, workers=2, seed=42):
    tf_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0),
        ]
    )

    tf_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )

    train_ds = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=tf_train,
    )

    test_ds = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=tf_test,
    )

    g = torch.Generator()
    g.manual_seed(seed)

    pin = torch.cuda.is_available()
    worker_fn = partial(seed_worker, base_seed=seed) if workers and workers > 0 else None
    persistent = True if workers and workers > 0 else False

    train_ld = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin,
        worker_init_fn=worker_fn,
        generator=g,
        persistent_workers=persistent,
    )

    test_ld = DataLoader(
        test_ds,
        batch_size=max(64, batch_size * 2),
        shuffle=False,
        num_workers=workers,
        pin_memory=pin,
        worker_init_fn=worker_fn,
        generator=g,
        persistent_workers=persistent,
    )

    return train_ld, test_ld

def run_epoch(model, loader, criterion, optimizer=None, device="cpu", scaler=None, scheduler=None):
    train_mode = optimizer is not None

    if train_mode:
        model.train()
    else:
        model.eval()

    total = 0
    correct = 0
    loss_sum = 0.0

    use_amp = scaler is not None and getattr(scaler, "is_enabled", lambda: False)()

    with (torch.enable_grad() if train_mode else torch.no_grad()):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(x)
                    loss = criterion(logits, y)
            else:
                logits = model(x)
                loss = criterion(logits, y)

            if train_mode:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                if scheduler is not None:
                    scheduler.step()

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_loaders(
        data_root="./data",
        batch_size=128,
        workers=2,
        seed=42,
    )

    model = MediumCNN(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.05,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
    )

    num_epochs = 35
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.25,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.15,
        div_factor=10.0,
        final_div_factor=100.0,
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_acc = -1.0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            scheduler=scheduler,
        )

        test_loss, test_acc = run_epoch(
            model,
            test_loader,
            criterion,
            optimizer=None,
            device=device,
            scaler=None,
            scheduler=None,
        )

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"epoch {epoch:02d} train_acc={train_acc:.3f} test_acc={test_acc:.3f} loss={train_loss:.3f}/{test_loss:.3f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    show_samples(model, test_loader, device, num_images=8)
    plt.show()

if __name__ == "__main__":
    main()
