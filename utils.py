import torch


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def load_weights(model, weight_path):
    """
    Load pretrained SqueezeNet weights.
    """

    print("Loading weights from:")
    print(weight_path)

    state_dict = torch.load(
        weight_path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state_dict)

    print("Weights loaded successfully!")

    return model


def get_prediction(output):
    """
    Get predicted class and confidence.
    """

    probabilities = torch.softmax(
        output,
        dim=1
    )

    confidence, predicted_class = torch.max(
        probabilities,
        dim=1
    )

    return (
        predicted_class.item(),
        confidence.item()
    )


def print_prediction(predicted_class, confidence):
    """
    Display prediction result.
    """

    class_names = {
        207: "Golden Retriever"
    }

    object_name = class_names.get(
        predicted_class,
        f"Class {predicted_class}"
    )

    print()
    print("========================================")
    print("           PREDICTION RESULT")
    print("========================================")
    print("Object:", object_name)
    print(
        f"Confidence: {confidence * 100:.2f}%"
    )
    print("========================================")


def count_parameters(model):
    """
    Count trainable parameters.
    """

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total


# ============================================================
# TEST UTILS
# ============================================================

if __name__ == "__main__":

    print()
    print("========================================")
    print("           UTILS TEST")
    print("========================================")

    print("Utility functions loaded successfully!")

    print("========================================")