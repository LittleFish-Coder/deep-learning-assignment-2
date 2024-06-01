from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam
import torch.nn as nn
from data.custom_dataset import MiniImageNetDataset
from models.cnn import CNN


if __name__ == "__main__":
    # Example usage

    # get the dataset and dataloader
    ## train
    train_dataset = MiniImageNetDataset(text_file="train.txt", root_dir="./dataset")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    ## val
    val_dataset = MiniImageNetDataset(text_file="val.txt", root_dir="./dataset")
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    ## test
    test_dataset = MiniImageNetDataset(text_file="test.txt", root_dir="./dataset")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")

    # initialize the model
    model = CNN(in_channels=3, num_classes=50)

    # initialize the optimizer
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # train the model
    model.train()

    for epoch in range(10):
        for batch_idx, (data, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            scores = model(data)
            loss = criterion(scores, targets)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} Iteration {batch_idx}/{len(train_loader)}: Loss = {loss.item()}")
