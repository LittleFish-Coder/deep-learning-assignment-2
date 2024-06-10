from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam
import torch.nn as nn
import torch
from data.custom_dataset import MiniImageNetDataset
from models.cnn import CNN
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


def plot_images(train_accuracy, val_accuracy, train_loss, val_loss, name="task1.png"):
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
    plt.savefig("task1.png")


# Designing a Convolution Module for Variable Input Channels
def task1():
    # get the dataset and dataloader
    ## train
    print(f"Preparing the training dataset...")
    train_dataset = MiniImageNetDataset(text_file="train.txt", root_dir="./dataset")
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=16)
    ## val
    print(f"Preparing the validation dataset...")
    val_dataset = MiniImageNetDataset(text_file="val.txt", root_dir="./dataset")
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=16)
    # ## test
    # print(f"Preparing the test dataset...")
    # test_dataset = MiniImageNetDataset(text_file="test.txt", root_dir="./dataset")
    # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    # print(f"Number of test samples: {len(test_dataset)}")

    # hyperparameters
    n_epochs = 15
    in_channels = 3
    num_classes = 50
    learning_rate = 0.001
    checkpoints_dir = "checkpoints"
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # initialize the model
    model = CNN(in_channels=in_channels, num_classes=num_classes).to(device=device)

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

            # print the accuracy and loss every 5 iterations
            if batch_idx % 5 == 0:
                print(f"Iteration {batch_idx}/{len(train_loader)}: Accuracy: {correct}, Loss: {loss.item()}")
        running_accuracy = running_accuracy / len(train_dataset)
        running_loss = running_loss / len(train_dataset)

        print(f"Training Accuracy: {running_accuracy}")
        print(f"Training Loss: {running_loss}")

        # record the training accuracy and loss
        train_accuracy.append(running_accuracy)
        train_loss.append(running_loss)

        # evaluate the model
        model.eval()
        with torch.no_grad():
            running_accuracy, running_loss = 0, 0
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

            running_accuracy = running_accuracy / len(val_dataset)
            running_loss = running_loss / len(val_dataset)
            print(f"Validation Accuracy: {running_accuracy}")
            print(f"Validation Loss: {running_loss}")

            # record the validation accuracy and loss
            val_accuracy.append(running_accuracy)
            val_loss.append(running_loss)

        # save the model every 5 epochs
        if epoch % 5 == 0:
            print(f"Saving the model for epoch {epoch}")
            torch.save(model.state_dict(), f"{checkpoints_dir}/task1_{epoch}.pth")

    # save the training and validation accuracy and loss (plot)
    plot_images(train_accuracy, val_accuracy, train_loss, val_loss, name="task1.png")


# Designing a Two-Layer Network for Image Classification
def task2():
    pass


if __name__ == "__main__":

    task = "task1"

    if task == "task1":
        print("Running task1")
        task1()  # do task1
    else:
        print("Running task2")
        task2()  # do task2
