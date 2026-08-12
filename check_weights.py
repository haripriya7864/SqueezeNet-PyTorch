import torch

from model import SqueezeNet


weight_path = "results/pretrained_models/squeezenet1_0-b66bff10.pth"

state_dict = torch.load(
    weight_path,
    map_location="cpu",
    weights_only=True
)

model = SqueezeNet(num_classes=1000)


weight_keys = list(state_dict.keys())
model_keys = list(model.state_dict().keys())


print("\n==============================")
print("NUMBER OF WEIGHT PARAMETERS")
print("==============================")
print(len(weight_keys))


print("\n==============================")
print("NUMBER OF MODEL PARAMETERS")
print("==============================")
print(len(model_keys))


print("\n==============================")
print("WEIGHT FILE KEYS")
print("==============================")

for key in weight_keys:
    print(key)


print("\n==============================")
print("YOUR MODEL KEYS")
print("==============================")

for key in model_keys:
    print(key)


print("\n==============================")
print("KEYS MISSING FROM YOUR MODEL")
print("==============================")

for key in weight_keys:
    if key not in model_keys:
        print(key)


print("\n==============================")
print("EXTRA KEYS IN YOUR MODEL")
print("==============================")

for key in model_keys:
    if key not in weight_keys:
        print(key)