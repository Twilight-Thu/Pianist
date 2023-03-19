import torchvision
import torch
from torchsummary import summary
from PIL import Image
from torch import nn
from torchvision import transforms
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torchvision.models.resnet18(pretrained=True)
model = model.to(device)
# print(model) [Channels, H, W]
# summary(model, input_size=(3, 224, 224))

image = Image.open('test.jpg')

transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]
)

image = transforms(image)
print(f"Image shape before: {image.shape}")
image = image.unsqueeze(0)
print(f"Image shape after: {image.shape}")

image = image.to(device)

# 遍历得到ResNet18所有卷积层及权重
model_weights = []
conv_layers = []

model_children = list(model.children())

count = 0

for i in range(len(model_children)):
    if type(model_children[i]) == nn.Conv2d:
        count = count + 1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])

    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            for child in model_children[i][j].children():
                if type(child) == nn.Conv2d:
                    count = count + 1
                    model_weights.append(child.weight)
                    conv_layers.append(child)

print(f"Total convolution layers: {count}")

outputs = []
names = []

for layer in conv_layers:
    image = layer(image)
    outputs.append(image)
    names.append(str(layer))

# print(len(outputs))
#
# for feature_map in outputs:
#     print(feature_map.shape)
#
# for element in names:
#     print(element)

processed = []
for feature_map in outputs:
    feature_map = feature_map.squeeze(0)
    gray_scale = torch.sum(feature_map, 0)
    gray_scale = gray_scale / feature_map.shape[0]
    processed.append(gray_scale.data.cpu().numpy())

fig = plt.figure(figsize=(30, 50))

for i in range(len(processed)):
    a = fig.add_subplot(5, 4, i + 1)
    img_plot = plt.imshow(processed[i])
    a.axis("off")
    a.set_title(names[i].split("(")[0], fontsize=30)
plt.show()

plt.savefig("resnet18_feature_maps.jpg", bbox_inches='tight')


