import os
from PIL import Image
import torch
from torch.utils.data import Dataset


# ============================================================
# SQUEEZENET IMAGE DATASET
# ============================================================

class SqueezeNetDataset(Dataset):

    def __init__(self, data, transform=None):

        self.data = data
        self.transform = transform

    # --------------------------------------------------------
    # Number of images
    # --------------------------------------------------------

    def __len__(self):

        return len(self.data)

    # --------------------------------------------------------
    # Get one image and its label
    # --------------------------------------------------------

    def __getitem__(self, index):

        item = self.data[index]

        image_path = item["image"]
        label = item["label"]

        # Check whether image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        # Open image
        image = Image.open(
            image_path
        ).convert("RGB")

        # Apply transformation if provided
        if self.transform is not None:
            image = self.transform(image)

        return image, label


# ============================================================
# SAMPLE DATA
# ============================================================

if __name__ == "__main__":

    data = [

        {
            "image": "image.png",
            "label": 207
        }

    ]

    dataset = SqueezeNetDataset(data)

    print()
    print("========================================")
    print("       SQUEEZENET DATASET TEST")
    print("========================================")

    print(
        "Number of images:",
        len(dataset)
    )

    image, label = dataset[0]

    print(
        "Image size:",
        image.size
    )

    print(
        "Label:",
        label
    )

    print()
    print("Dataset loaded successfully!")

    print("========================================")