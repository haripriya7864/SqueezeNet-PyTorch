import torch
from PIL import Image


def preprocess_image(image_path):

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Resize to 256 x 256
    image = image.resize((256, 256))

    # Center crop to 224 x 224
    width, height = image.size

    left = (width - 224) // 2
    top = (height - 224) // 2

    right = left + 224
    bottom = top + 224

    image = image.crop((left, top, right, bottom))

    # Convert image pixels to tensor
    image_data = list(image.getdata())

    tensor = torch.tensor(
        image_data,
        dtype=torch.float32
    )

    # Convert to 224 x 224 x 3
    tensor = tensor.reshape(224, 224, 3)

    # Change HWC to CHW
    tensor = tensor.permute(2, 0, 1)

    # Convert pixel values from 0-255 to 0-1
    tensor = tensor / 255.0

    # ImageNet mean
    mean = torch.tensor([
        0.485,
        0.456,
        0.406
    ]).reshape(3, 1, 1)

    # ImageNet standard deviation
    std = torch.tensor([
        0.229,
        0.224,
        0.225
    ]).reshape(3, 1, 1)

    # Normalize
    tensor = (tensor - mean) / std

    # Add batch dimension
    tensor = tensor.unsqueeze(0)

    return tensor


if __name__ == "__main__":

    image_path = "image.png"

    image_tensor = preprocess_image(image_path)

    print("Image preprocessing successful!")
    print("Image tensor shape:", image_tensor.shape)