import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch


class MiniImageNetDataset(Dataset):
    def __init__(self, text_file, root_dir, channels="RGB", show_partial_image=False):
        """
        Args:
            text_file (string): Path to the txt file with annotations
            root_dir (string): Directory with all the images
            channels (string): Desired channels to be used ["RGB", "RG", "GB", "R", "G", "B"]
        """
        self.root_dir = root_dir
        self.channels = channels
        self.transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        self.image_paths = []
        self.labels = []
        # Read the text file and populate the image_paths and labels
        with open(os.path.join(self.root_dir, text_file), "r") as f:
            for line in f:
                image_path, label = line.strip().split(" ")
                self.image_paths.append(os.path.join(self.root_dir, image_path))
                self.labels.append(int(label))

        if show_partial_image:
            self._show_partial_image()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        if image.mode == "L":  # if the image is grayscale
            image = image.convert("RGB")
        label = self.labels[idx]
        image = self.transform(image)  # apply the transformation to the image
        image = self._select_channels(image)  # select the specified channels
        return image, label

    def _select_channels(self, image):
        """
        Selects the specified channels from the image.
        """
        if self.channels == "RGB":
            return image

        channel_map = {"R": 0, "G": 1, "B": 2}
        selected_channels = [channel_map[channel] for channel in self.channels]
        selected_image = torch.zeros(3, image.size(1), image.size(2))
        for channel in selected_channels:
            selected_image[channel] = image[channel, :, :]

        return selected_image

    def _show_partial_image(self):
        for i in range(min(5, len(self.labels))):
            img_path = self.image_paths[i]
            img_path = os.path.join(self.root_dir, img_path)
            image = Image.open(img_path).convert("RGB")
            if image.mode == "L":  # if the image is grayscale
                image = image.convert("RGB")
            image = self.transform(image)
            image = self._select_channels(image)
            # tensor to image
            image = transforms.ToPILImage()(image)

            print(f"Sample {i + 1}: Shape={image.size}, Label={self.labels[i]}")

            # save the image
            if not os.path.exists(self.channels):
                os.makedirs(self.channels)
            image.save(f"{self.channels}/sample_{i + 1}_{self.channels}.jpg")


if __name__ == "__main__":

    # select the desired channels
    # channels = ["RGB", "RG", "GB", "R", "G", "B"]

    for channel in ["RGB", "RG", "GB", "R", "G", "B"]:
        print(f"Channels: {channel}")

        dataset = MiniImageNetDataset(text_file="train.txt", root_dir="../dataset", channels=channel, show_partial_image=True)
        print(f"Number of samples in the dataset: {len(dataset)}")
        image, label = dataset[0]
        print(f"Image shape: {image.shape}, Label: {label}")
