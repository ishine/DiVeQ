import os
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image
import numpy as np


class Dataset_Custom(Dataset):
    def __init__(self, args):
        self.root_dir = args.data_path

        self.transform = transforms.Compose(
            [
                transforms.Resize(args.size),
                transforms.CenterCrop(args.size),
            ]
        )

        # Get list of image file paths
        self.image_paths = [
            os.path.join(self.root_dir, fname)
            for fname in os.listdir(self.root_dir)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
        ]
        self.image_paths.sort()  # Optional: sort for reproducibility

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)

        return image  # No label, just return the image