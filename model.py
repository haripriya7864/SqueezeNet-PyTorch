import torch
import torch.nn as nn


# ============================================================
# FIRE MODULE
# ============================================================

class Fire(nn.Module):

    def __init__(
        self,
        in_channels,
        squeeze_channels,
        expand1x1_channels,
        expand3x3_channels
    ):
        super().__init__()

        # Squeeze layer
        self.squeeze = nn.Conv2d(
            in_channels,
            squeeze_channels,
            kernel_size=1
        )

        self.squeeze_activation = nn.ReLU(inplace=True)

        # 1x1 expand layer
        self.expand1x1 = nn.Conv2d(
            squeeze_channels,
            expand1x1_channels,
            kernel_size=1
        )

        self.expand1x1_activation = nn.ReLU(inplace=True)

        # 3x3 expand layer
        self.expand3x3 = nn.Conv2d(
            squeeze_channels,
            expand3x3_channels,
            kernel_size=3,
            padding=1
        )

        self.expand3x3_activation = nn.ReLU(inplace=True)


    def forward(self, x):

        x = self.squeeze(x)
        x = self.squeeze_activation(x)

        expand1x1 = self.expand1x1(x)
        expand1x1 = self.expand1x1_activation(expand1x1)

        expand3x3 = self.expand3x3(x)
        expand3x3 = self.expand3x3_activation(expand3x3)

        return torch.cat(
            [expand1x1, expand3x3],
            dim=1
        )


# ============================================================
# SQUEEZENET 1.0
# ============================================================

class SqueezeNet(nn.Module):

    def __init__(self, num_classes=1000):

        super().__init__()

        # ----------------------------------------------------
        # FEATURE EXTRACTION
        # ----------------------------------------------------

        self.features = nn.Sequential(

            # Conv1
            nn.Conv2d(
                3,
                96,
                kernel_size=7,
                stride=2
            ),

            nn.ReLU(inplace=True),

            # MaxPool1
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                ceil_mode=True
            ),

            # Fire2
            Fire(
                in_channels=96,
                squeeze_channels=16,
                expand1x1_channels=64,
                expand3x3_channels=64
            ),

            # Fire3
            Fire(
                in_channels=128,
                squeeze_channels=16,
                expand1x1_channels=64,
                expand3x3_channels=64
            ),

            # Fire4
            Fire(
                in_channels=128,
                squeeze_channels=32,
                expand1x1_channels=128,
                expand3x3_channels=128
            ),

            # MaxPool2
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                ceil_mode=True
            ),

            # Fire5
            Fire(
                in_channels=256,
                squeeze_channels=32,
                expand1x1_channels=128,
                expand3x3_channels=128
            ),

            # Fire6
            Fire(
                in_channels=256,
                squeeze_channels=48,
                expand1x1_channels=192,
                expand3x3_channels=192
            ),

            # Fire7
            Fire(
                in_channels=384,
                squeeze_channels=48,
                expand1x1_channels=192,
                expand3x3_channels=192
            ),

            # Fire8
            Fire(
                in_channels=384,
                squeeze_channels=64,
                expand1x1_channels=256,
                expand3x3_channels=256
            ),

            # MaxPool3
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                ceil_mode=True
            ),

            # Fire9
            Fire(
                in_channels=512,
                squeeze_channels=64,
                expand1x1_channels=256,
                expand3x3_channels=256
            )
        )


        # ----------------------------------------------------
        # CLASSIFIER
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            # IMPORTANT:
            # Dropout must be index 0
            nn.Dropout(p=0.5),

            # Final 1x1 convolution
            # This becomes classifier.1
            nn.Conv2d(
                in_channels=512,
                out_channels=num_classes,
                kernel_size=1
            ),

            nn.ReLU(inplace=True),

            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )


    # --------------------------------------------------------
    # FORWARD PASS
    # --------------------------------------------------------

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        x = torch.flatten(
            x,
            1
        )

        return x


# ============================================================
# LOAD PRETRAINED WEIGHTS
# ============================================================

def load_weights(model, weight_path):

    print()
    print("========================================")
    print("LOADING PRETRAINED WEIGHTS")
    print("========================================")

    print("Weight file:")
    print(weight_path)

    state_dict = torch.load(
        weight_path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state_dict)

    print()
    print("Weights loaded successfully!")
    print("========================================")


# ============================================================
# TEST THE MODEL
# ============================================================

if __name__ == "__main__":

    # Create YOUR SqueezeNet architecture
    model = SqueezeNet(
        num_classes=1000
    )

    # Path to pretrained weights
    weight_path = (
        "results/pretrained_models/"
        "squeezenet1_0-b66bff10.pth"
    )

    # Load pretrained weights
    load_weights(
        model,
        weight_path
    )

    # Evaluation mode
    model.eval()

    # Create a dummy input image
    x = torch.randn(
        1,
        3,
        224,
        224
    )

    # Run inference
    with torch.no_grad():

        output = model(x)

    # Display shapes
    print()
    print("Input shape :", x.shape)
    print("Output shape:", output.shape)