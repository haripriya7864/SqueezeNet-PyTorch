import torch
from PIL import Image

from model import SqueezeNet


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_path):

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Resize image
    image = image.resize((224, 224))

    # Convert image pixels to a list
    image_data = list(image.getdata())

    # Convert pixels to PyTorch tensor
    tensor = torch.tensor(
        image_data,
        dtype=torch.float32
    )

    # Convert H x W x C to C x H x W
    tensor = tensor.reshape(224, 224, 3)
    tensor = tensor.permute(2, 0, 1)

    # Convert pixel values from 0-255 to 0-1
    tensor = tensor / 255.0

    # ImageNet mean
    mean = torch.tensor([
        0.485,
        0.456,
        0.406
    ]).view(3, 1, 1)

    # ImageNet standard deviation
    std = torch.tensor([
        0.229,
        0.224,
        0.225
    ]).view(3, 1, 1)

    # Normalize image
    tensor = (tensor - mean) / std

    # Add batch dimension
    tensor = tensor.unsqueeze(0)

    return tensor


# ============================================================
# LOAD OUR SQUEEZENET MODEL
# ============================================================

def load_model():

    # Create our manually implemented SqueezeNet
    model = SqueezeNet(
        num_classes=1000
    )

    # Path to pretrained weights
    weight_path = (
        "results/pretrained_models/"
        "squeezenet1_0-b66bff10.pth"
    )

    print("Loading pretrained weights...")

    # Load weights manually
    state_dict = torch.load(
        weight_path,
        map_location="cpu",
        weights_only=True
    )

    # Load weights into our model
    model.load_state_dict(state_dict)

    # Evaluation mode
    model.eval()

    print("Weights loaded successfully!")

    return model


# ============================================================
# PREDICTION
# ============================================================

def predict(model, image_tensor):

    with torch.no_grad():

        # Model prediction
        output = model(image_tensor)

        # Convert logits to probabilities
        probabilities = torch.softmax(
            output,
            dim=1
        )

        # Get top 5 predictions
        top5_probabilities, top5_classes = torch.topk(
            probabilities,
            5,
            dim=1
        )

    return (
        top5_probabilities[0],
        top5_classes[0]
    )


# ============================================================
# IMAGE CLASS NAMES
# ============================================================

# Class number 207 corresponds to Golden Retriever
CLASS_NAMES = {
    207: "Golden Retriever"
}


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print()
    print("========================================")
    print("       SQUEEZENET IMAGE CLASSIFIER")
    print("========================================")

    # --------------------------------------------------------
    # Image path
    # --------------------------------------------------------

    image_path = "image.png"

    # --------------------------------------------------------
    # Preprocess image
    # --------------------------------------------------------

    print()
    print("Preprocessing image...")

    image_tensor = preprocess_image(
        image_path
    )

    print(
        "Image preprocessing successful!"
    )

    print(
        "Image tensor shape:",
        image_tensor.shape
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print()

    model = load_model()

    # --------------------------------------------------------
    # Run prediction
    # --------------------------------------------------------

    print()
    print("Running prediction...")

    probabilities, classes = predict(
        model,
        image_tensor
    )

    # --------------------------------------------------------
    # Display TOP 5
    # --------------------------------------------------------

    print()
    print("========================================")
    print("           TOP 5 PREDICTIONS")
    print("========================================")

    for i in range(5):

        class_number = classes[i].item()

        confidence = (
            probabilities[i].item() * 100
        )

        object_name = CLASS_NAMES.get(
            class_number,
            f"Class {class_number}"
        )

        print(
            f"{i + 1}. {object_name}"
            f" | Confidence: {confidence:.2f}%"
        )

    # --------------------------------------------------------
    # Final prediction
    # --------------------------------------------------------

    print()
    print("========================================")
    print("           FINAL PREDICTION")
    print("========================================")

    best_class = classes[0].item()

    best_confidence = (
        probabilities[0].item() * 100
    )

    best_name = CLASS_NAMES.get(
        best_class,
        f"Class {best_class}"
    )

    print("Object:", best_name)

    print(
        f"Confidence: {best_confidence:.2f}%"
    )

    print("========================================")