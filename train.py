from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam
import torch.nn as nn
import torch
from data.custom_dataset import MiniImageNetDataset
from models.cnn import CNN
from tqdm import tqdm
from torchvision import models
import matplotlib.pyplot as plt
from torchsummary import summary
import os
import argparse


def plot_images(train_accuracy, val_accuracy, train_loss, val_loss, filename="acc_loss.png"):

    # tensor to numpy
    train_accuracy = [x.cpu().item() if isinstance(x, torch.Tensor) else x for x in train_accuracy]
    val_accuracy = [x.cpu().item() if isinstance(x, torch.Tensor) else x for x in val_accuracy]
    train_loss = [x.cpu().item() if isinstance(x, torch.Tensor) else x for x in train_loss]
    val_loss = [x.cpu().item() if isinstance(x, torch.Tensor) else x for x in val_loss]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracy, label="Train Accuracy")
    plt.plot(val_accuracy, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{filename}")


def save_metrics(train_accuracy, val_accuracy, train_loss, val_loss, filename="log/log.txt"):
    # tensor to numpy
    train_accuracy = [x.cpu().item() if isinstance(x, torch.Tensor) else x for x in train_accuracy]
    val_accuracy = [x.cpu().item() if isinstance(x, torch.Tensor) else x for x in val_accuracy]
    train_loss = [x.cpu().item() if isinstance(x, torch.Tensor) else x for x in train_loss]
    val_loss = [x.cpu().item() if isinstance(x, torch.Tensor) else x for x in val_loss]

    with open(filename, "w") as f:
        f.write("Train Accuracy:\n")
        for item in train_accuracy:
            f.write("%s\n" % item)

        f.write("\nValidation Accuracy:\n")
        for item in val_accuracy:
            f.write("%s\n" % item)

        f.write("\nTrain Loss:\n")
        for item in train_loss:
            f.write("%s\n" % item)

        f.write("\nValidation Loss:\n")
        for item in val_loss:
            f.write("%s\n" % item)


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    n_epochs,
    device,
    learning_rate,
    checkpoints_dir,
    checkpoints_name,
):
    # initialize the optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Train and Val Accuracy/Loss
    train_accuracy, train_loss = [], []
    val_accuracy, val_loss = [], []

    # train the model
    print("Start training...")
    for epoch in range(n_epochs):
        model.train()  # set the model to training mode
        print(f"Epoch {epoch}/{n_epochs}")
        running_accuracy, running_loss = 0, 0
        total = 0
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):

            # move the data to the device
            data = data.to(device=device)
            targets = targets.to(device=device)

            optimizer.zero_grad()
            scores = model(data)
            # print(scores.shape, targets.shape)    # torch.Size([64, 50]) torch.Size([64])
            loss = criterion(scores, targets)
            loss.backward()
            optimizer.step()

            # predict
            _, predictions = scores.max(1)
            correct = (predictions == targets).sum()

            # sum the correct predictions and loss
            running_accuracy += correct
            running_loss += loss.item()
            total += targets.size(0)

            # print the accuracy and loss every 5 iterations
            if batch_idx % 5 == 0:
                print(f"Iteration {batch_idx}/{len(train_loader)}: Accuracy: {correct}/{len(data)} Loss: {loss}/{len(data)}")
        running_accuracy = running_accuracy / total
        running_loss = running_loss / total

        print(f"Training Accuracy: {running_accuracy}")
        print(f"Training Loss: {running_loss}")

        # record the training accuracy and loss
        train_accuracy.append(running_accuracy)
        train_loss.append(running_loss)

        # evaluate the model
        model.eval()
        with torch.no_grad():
            running_accuracy, running_loss = 0, 0
            total = 0
            for data, targets in tqdm(val_loader):

                # move the data to the device
                data = data.to(device=device)
                targets = targets.to(device=device)

                scores = model(data)
                loss = criterion(scores, targets)

                # predict
                _, predictions = scores.max(1)
                correct = (predictions == targets).sum()

                # sum the correct predictions and loss
                running_accuracy += correct
                running_loss += loss.item()
                total += targets.size(0)

            running_accuracy = running_accuracy / total
            running_loss = running_loss / total
            print(f"Validation Accuracy: {running_accuracy}")
            print(f"Validation Loss: {running_loss}")

            # record the validation accuracy and loss
            val_accuracy.append(running_accuracy)
            val_loss.append(running_loss)

        # save the model every 5 epochs
        if epoch % 5 == 0:
            print(f"Saving the model for epoch {epoch}")
            torch.save(model.state_dict(), f"{checkpoints_dir}/{checkpoints_name}_{epoch}.pth")

        # save the best model based on validation loss
        # print(running_loss)
        # print(val_loss)
        if running_loss <= min(val_loss):
            print(f"Saving the best model for epoch {epoch}")
            torch.save(model.state_dict(), f"{checkpoints_dir}/{checkpoints_name}_best.pth")

    return train_accuracy, val_accuracy, train_loss, val_loss


# Designing a Convolution Module for Variable Input Channels
def task1_CNN():

    # checkpoints
    checkpoints_dir = "checkpoints"
    checkpoints_name = "task1_CNN"  # checkpoints name
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    # log history
    log_dir = "log"
    log_name = "task1_CNN_log.log"  # log name
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # plot history
    plot_dir = "plot"
    plot_name = "task1_CNN.png"  # plot name
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # hyperparameters
    n_epochs = 15
    num_classes = 50
    learning_rate = 0.001
    batch_size = 64
    in_channels = 3
    input_size = (3, 256, 256)

    # get the dataset and dataloader
    ## train
    print(f"Preparing the training dataset...")
    train_dataset = MiniImageNetDataset(text_file="train.txt", root_dir="./dataset")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    ## val
    print(f"Preparing the validation dataset...")
    val_dataset = MiniImageNetDataset(text_file="val.txt", root_dir="./dataset")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")

    # initialize the model
    model = CNN(in_channels=in_channels, num_classes=num_classes)

    # move the model to the device
    model = model.to(device=device)

    # print the model summary
    summary(model, input_size=input_size)

    # train model
    train_accuracy, val_accuracy, train_loss, val_loss = train_model(
        model, train_loader, val_loader, n_epochs, device, learning_rate, checkpoints_dir, checkpoints_name
    )

    # save the metrics
    save_metrics(train_accuracy, val_accuracy, train_loss, val_loss, filename=f"{log_dir}/{log_name}")

    # save the training and validation accuracy and loss (plot)
    plot_images(train_accuracy, val_accuracy, train_loss, val_loss, filename=f"{plot_dir}/{plot_name}")


# Designing a Convolution Module for Variable Input Channels
def task1_dynamic():
    pass


# Designing a Two-Layer Network for Image Classification
def task2():
    pass


# Designing a Two-Layer Network for Image Classification
def task2_ResNet34():

    # checkpoints
    checkpoints_dir = "checkpoints"
    checkpoints_name = "task2_ResNet34"  # checkpoints name
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    # log history
    log_dir = "log"
    log_name = "task2_ResNet34_log.log"  # log name
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # plot history
    plot_dir = "plot"
    plot_name = "task2_ResNet34.png"  # plot name
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # hyperparameters
    n_epochs = 15
    num_classes = 50
    learning_rate = 0.001
    batch_size = 64
    in_channels = 3
    input_size = (3, 256, 256)

    # get the dataset and dataloader
    ## train
    print(f"Preparing the training dataset...")
    train_dataset = MiniImageNetDataset(text_file="train.txt", root_dir="./dataset")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    ## val
    print(f"Preparing the validation dataset...")
    val_dataset = MiniImageNetDataset(text_file="val.txt", root_dir="./dataset")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")

    # initialize model
    model = models.resnet34(weights=None, num_classes=num_classes)

    # move the model to the device
    model = model.to(device=device)

    # print the model summary
    summary(model, input_size=input_size)

    # train model
    train_accuracy, val_accuracy, train_loss, val_loss = train_model(
        model, train_loader, val_loader, n_epochs, device, learning_rate, checkpoints_dir, checkpoints_name
    )

    # save the metrics
    save_metrics(train_accuracy, val_accuracy, train_loss, val_loss, filename=f"{log_dir}/{log_name}")

    # save the training and validation accuracy and loss (plot)
    plot_images(train_accuracy, val_accuracy, train_loss, val_loss, filename=f"{plot_dir}/{plot_name}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--task", type=str, default="task1_CNN", help="Task to run [task1_CNN, task1_dynamic, task2_ResNet34, task2]")
    args = parser.parse_args()

    task = args.task

    print(f"Running task: {task}")

    if task == "task1_CNN":
        task1_CNN()
    elif task == "task1_dynamic":
        task1_dynamic()
    elif task == "task2_ResNet34":
        task2_ResNet34()
    elif task == "task2":
        task2()
    else:
        print("Invalid task")
        exit(1)
