import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MiniImageNetDataset(Dataset):
    def __init__(self, text_file, root_dir):
        self.root_dir = root_dir
        self.transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
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
        if image.mode == "L":  # if the image is grayscale
            image = image.convert("RGB")
        label = self.labels[idx]
        image = self.transform(image)  # apply the transformation to the image
        return image, label
