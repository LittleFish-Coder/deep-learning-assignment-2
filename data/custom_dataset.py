import os
from PIL import Image
from torch.utils.data import Dataset


class MiniImageNetDataset(Dataset):
    def __init__(self, text_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        # Read the text file and populate the image_paths and labels
        with open(os.path.join(self.root_dir, text_file), "r") as f:
            for line in f:
                image_path, label = line.strip().split(" ")
                self.image_paths.append(os.path.join(self.root_dir, image_path))
                self.labels.append(int(label))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
