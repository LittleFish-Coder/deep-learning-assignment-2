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


def plot_images(train_accuracy, val_accuracy, train_loss, val_loss, name="task1.png"):

    # tensor to numpy
    train_accuracy = [x.item() for x in train_accuracy]
    val_accuracy = [x.item() for x in val_accuracy]
    train_loss = [x.item() for x in train_loss]
    val_loss = [x.item() for x in val_loss]

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
                print(f"Iteration {batch_idx}/{len(train_loader)}: Accuracy: {correct}/{len(data)} Loss: {loss}/{len(data)}")
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

        # save the best model based on validation loss
        if epoch == 0 or running_loss < min(val_loss):
            print(f"Saving the best model for epoch {epoch}")
            torch.save(model.state_dict(), f"{checkpoints_dir}/task1_best.pth")

    # save the training and validation accuracy and loss (plot)
    plot_images(train_accuracy, val_accuracy, train_loss, val_loss, name="task1.png")


def task1_dynamic():
    pass


# Designing a Two-Layer Network for Image Classification
def task2():
    pass


def task2_ResNet34():

    # get the dataset and dataloader
    ## train
    print(f"Preparing the training dataset...")
    train_dataset = MiniImageNetDataset(text_file="train.txt", root_dir="./dataset")
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=16)
    ## val
    print(f"Preparing the validation dataset...")
    val_dataset = MiniImageNetDataset(text_file="val.txt", root_dir="./dataset")
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=16)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    # print(f"Number of test samples: {len(test_dataset)}")

    # hyperparameters
    n_epochs = 15
    num_classes = 50
    learning_rate = 0.001
    checkpoints_dir = "checkpoints"
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # initialize the model: use the ResNet34 model from torchvision
    model = models.resnet34(pretrained=False)

    # Replace the top layer for fine-tuning
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)  # num_classes is the number of classes in your dataset

    # move the model to the device
    model = model.to(device=device)

    summary(model, input_size=(3, 256, 256))

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
                print(f"Iteration {batch_idx}/{len(train_loader)}: Accuracy: {correct}/{len(data)} Loss: {loss}/{len(data)}")
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

        # save the best model based on validation loss
        if epoch == 0 or running_loss < min(val_loss):
            print(f"Saving the best model for epoch {epoch}")
            torch.save(model.state_dict(), f"{checkpoints_dir}/task1_best.pth")

    # save the training and validation accuracy and loss (plot)
    plot_images(train_accuracy, val_accuracy, train_loss, val_loss, name="task1.png")


if __name__ == "__main__":

    task = "task2_ResNet34"

    if task == "task1":
        print("Running task1")
        task1()  # do task1
    elif task == "task1_dynamivc":
        print("Running task1_dynamic")
        task1_dynamic()
    elif task == "task2_ResNet34":
        print("Running task2_ResNet34")
        task2_ResNet34()
    else:
        print("Running task2")
        task2()  # do task2
