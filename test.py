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


def test(model, test_loader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(tqdm(test_loader)):
            # move the data to device
            data, targets = data.to(device), targets.to(device)

            outputs = model(data)
            _, predicted = torch.max(outputs, 1)

            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    return correct / total


def task1():
    pass


def task1_dynamic():
    pass


def task2():
    pass


def task2_ResNet34():

    # checkpoints
    checkpoints_dir = "checkpoints"
    checkpoints_name = "task2_ResNet34_5.pth"  # checkpoints name
    if not os.path.exists(f"{checkpoints_dir}/{checkpoints_name}"):
        print("Checkpoints not found")
        return

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # hyperparameters
    num_classes = 50
    batch_size = 64
    input_size = (3, 256, 256)

    # get the dataset and dataloader
    ## test
    print(f"Preparing the testing dataset...")
    test_dataset = MiniImageNetDataset(text_file="test.txt", root_dir="./dataset")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
    print(f"Number of test samples: {len(test_dataset)}")

    # initialize model
    model = models.resnet34(weights=None, num_classes=num_classes)

    # load model
    model.load_state_dict(torch.load(f"{checkpoints_dir}/{checkpoints_name}"))
    model.eval()

    # move model to device
    model.to(device)

    # summary
    summary(model, input_size=input_size)

    # test
    accuracy = test(model, test_loader)

    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    print("Test Mode")

    task = "task2_ResNet34"

    if task == "task1":
        print("Running task1")
        task1()  # do task1
    elif task == "task1_dynamic":
        print("Running task1_dynamic")
        task1_dynamic()
    elif task == "task2_ResNet34":
        print("Running task2_ResNet34")
        task2_ResNet34()
    else:
        print("Running task2")
        task2()  # do task2
