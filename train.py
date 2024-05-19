from torch.utils.data import DataLoader
from data.custom_dataset import MiniImageNetDataset
from torchvision import transforms


if __name__ == "__main__":
    # Example usage

    # get the dataset and dataloader
    ## train
    train_dataset = MiniImageNetDataset(text_file="train.txt", root_dir="./dataset", transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    ## val
    val_dataset = MiniImageNetDataset(text_file="val.txt", root_dir="./dataset", transform=transforms.ToTensor())
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    ## test
    test_dataset = MiniImageNetDataset(text_file="test.txt", root_dir="./dataset", transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")
