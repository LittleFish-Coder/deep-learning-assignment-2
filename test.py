from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam
import torch.nn as nn
import torch
from data.custom_dataset import MiniImageNetDataset
from models.cnn import CNN, DYNAMIC_CNN
from models.attention_cnn import ConvNet
from tqdm import tqdm
from torchvision import models
import matplotlib.pyplot as plt
from torchsummary import summary
import os
import argparse


def save_metrics(test_accuracy, test_loss, filename="log/log.txt"):
    with open(filename, "w") as f:
        f.write("Test Accuracy:\n")
        f.write(f"{test_accuracy}\n")

        f.write("Test Loss:\n")
        f.write(f"{test_loss}\n")


def test_model(model, test_loader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    model.eval()

    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(tqdm(test_loader)):
            # move the data to device
            data, targets = data.to(device), targets.to(device)

            outputs = model(data)

            loss += criterion(outputs, targets).item()

            _, predicted = torch.max(outputs, 1)

            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = correct / total
    loss = loss / total

    return accuracy, loss


def task1_CNN(channels="RGB"):
    # checkpoints
    checkpoints_dir = "checkpoints"
    checkpoints_name = "task1_CNN_best.pth"  # checkpoints name
    if not os.path.exists(f"{checkpoints_dir}/{checkpoints_name}"):
        print("Checkpoints not found")
        return

    # log history
    log_dir = "log"
    log_name = f"task1_CNN_{channels}_test_accuracy.log"  # log name
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # hyperparameters
    num_classes = 50
    batch_size = 64
    input_size = (3, 256, 256)

    # get the dataset and dataloader
    ## test
    print(f"Preparing the testing dataset...")
    test_dataset = MiniImageNetDataset(text_file="test.txt", root_dir="./dataset", channels=channels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
    print(f"Number of test samples: {len(test_dataset)}")

    # initialize model
    model = CNN(num_classes=num_classes)

    # load model
    model.load_state_dict(torch.load(f"{checkpoints_dir}/{checkpoints_name}"))
    model.eval()

    # move model to device
    model.to(device)

    # summary
    summary(model, input_size=input_size)

    # test
    accuracy, loss = test_model(model, test_loader)

    # save metrics
    print(f"Accuracy: {accuracy}")
    print(f"Loss: {loss}")
    save_metrics(accuracy, loss, f"{log_dir}/{log_name}")


def task1_dynamic(channels="RGB"):
    # checkpoints
    checkpoints_dir = "checkpoints"
    checkpoints_name = "task1_DYNAMIC_CNN_best.pth"  # checkpoints name
    if not os.path.exists(f"{checkpoints_dir}/{checkpoints_name}"):
        print("Checkpoints not found")
        return

    # log history
    log_dir = "log"
    log_name = f"task1_DYNAMIC_CNN_{channels}_test_accuracy.log"  # log name
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # hyperparameters
    num_classes = 50
    batch_size = 64
    input_size = (len(channels), 256, 256)

    # get the dataset and dataloader
    ## test
    print(f"Preparing the testing dataset...")
    test_dataset = MiniImageNetDataset(text_file="test.txt", root_dir="./dataset", channels=channels, dynamic=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
    print(f"Number of test samples: {len(test_dataset)}")

    # initialize model
    model = DYNAMIC_CNN(num_classes=num_classes)

    # load model
    model.load_state_dict(torch.load(f"{checkpoints_dir}/{checkpoints_name}"))
    model.eval()

    # move model to device
    model.to(device)

    # summary
    summary(model, input_size=input_size)

    # test
    accuracy, loss = test_model(model, test_loader)

    # save metrics
    print(f"Accuracy: {accuracy}")
    print(f"Loss: {loss}")
    save_metrics(accuracy, loss, f"{log_dir}/{log_name}")


def task2_attention():

    # checkpoints
    checkpoints_dir = "checkpoints"
    checkpoints_name = "task2_attention_best.pth"  # checkpoints name
    if not os.path.exists(f"{checkpoints_dir}/{checkpoints_name}"):
        print("Checkpoints not found")
        return

    # log history
    log_dir = "log"
    log_name = "task2_attention_test_accuracy.log"  # log name
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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
    model = ConvNet(num_classes=num_classes)

    # load model
    model.load_state_dict(torch.load(f"{checkpoints_dir}/{checkpoints_name}"))
    model.eval()

    # move model to device
    model.to(device)

    # summary
    summary(model, input_size=input_size)

    # test
    accuracy, loss = test_model(model, test_loader)

    # save metrics
    print(f"Accuracy: {accuracy}")
    print(f"Loss: {loss}")
    save_metrics(accuracy, loss, f"{log_dir}/{log_name}")


def task2_ResNet34():

    # checkpoints
    checkpoints_dir = "checkpoints"
    checkpoints_name = "task2_ResNet34_best.pth"  # checkpoints name
    if not os.path.exists(f"{checkpoints_dir}/{checkpoints_name}"):
        print("Checkpoints not found")
        return

    # log history
    log_dir = "log"
    log_name = "task2_ResNet34_test_accuracy.log"  # log name
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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
    accuracy, loss = test_model(model, test_loader)

    # save metrics
    print(f"Accuracy: {accuracy}")
    print(f"Loss: {loss}")
    save_metrics(accuracy, loss, f"{log_dir}/{log_name}")


if __name__ == "__main__":
    print("Test Mode")

    parser = argparse.ArgumentParser(description="Testing script")
    parser.add_argument("--task", type=str, default="task1_CNN", help="Task to run [task1_CNN, task1_dynamic, task2_ResNet34, task2]")
    parser.add_argument("--channels", type=str, default="RGB", help="Channels to use [RGB, RG, GB, R, G, B]")
    args = parser.parse_args()

    task = args.task
    channels = args.channels

    print(f"Running task: {task}")

    if task == "task1_CNN":
        task1_CNN(channels=channels)
    elif task == "task1_dynamic":
        task1_dynamic(channels=channels)
    elif task == "task2_ResNet34":
        task2_ResNet34()
    elif task == "task2_attention":
        task2_attention()
    else:
        print("Invalid task")
        exit(1)
