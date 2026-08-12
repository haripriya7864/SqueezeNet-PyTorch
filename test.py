import torch

from model import SqueezeNet


# ============================================================
# TEST SQUEEZENET MODEL
# ============================================================

def test_model():

    print()
    print("========================================")
    print("       SQUEEZENET MODEL TEST")
    print("========================================")

    # --------------------------------------------------------
    # 1. Create model
    # --------------------------------------------------------

    print()
    print("Creating SqueezeNet model...")

    model = SqueezeNet(
        num_classes=1000
    )

    print("Model created successfully!")

    # --------------------------------------------------------
    # 2. Load pretrained weights
    # --------------------------------------------------------

    weight_path = (
        "results/pretrained_models/"
        "squeezenet1_0-b66bff10.pth"
    )

    print()
    print("Loading pretrained weights...")

    state_dict = torch.load(
        weight_path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(
        state_dict
    )

    print("Weights loaded successfully!")

    # --------------------------------------------------------
    # 3. Evaluation mode
    # --------------------------------------------------------

    model.eval()

    print()
    print("Model is in evaluation mode.")

    # --------------------------------------------------------
    # 4. Create test input
    # --------------------------------------------------------

    x = torch.randn(
        1,
        3,
        224,
        224
    )

    print()
    print("Input shape:")
    print(x.shape)

    # --------------------------------------------------------
    # 5. Run model
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(x)

    print()
    print("Output shape:")
    print(output.shape)

    # --------------------------------------------------------
    # 6. Check output
    # --------------------------------------------------------

    if output.shape == torch.Size([1, 1000]):

        print()
        print("TEST PASSED!")
        print("SqueezeNet is working correctly.")

    else:

        print()
        print("TEST FAILED!")
        print("Unexpected output shape.")

    print()
    print("========================================")


# ============================================================
# RUN TEST
# ============================================================

if __name__ == "__main__":

    test_model()