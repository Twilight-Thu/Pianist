import os
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transform
import pandas as pd
import numpy as np

from Networks_v2 import PM_Single_Net

def process_input(image):
    image = transform.Resize((256, 256))(image)
    image = transform.ToTensor()(image)[None, ::]
    return image

Daytime_PM_MAX = 262.0
Daytime_PM_MIN = 1.0

test_imgs = r'../imgs'
img_dir = test_imgs + '/'
output_dir = img_dir
print("pred_dir:", output_dir)

exp_dir = r""
model_dir = exp_dir + r"./MobileNetv2.pk"
device = "cuda" if torch.cuda.is_available() else "cpu"
ckp = torch.load(model_dir, map_location=device)
print("Load model: ", model_dir)

net = PM_Single_Net(Body='mobilev2')
net = nn.DataParallel(net)
net.load_state_dict(ckp["model"])
net.eval()

# 开始测试------------------
all_pred = np.empty((0, 1), float)
img_names = []

for im in os.listdir(img_dir):
    if not im.endswith(('jpg', 'jpeg', 'png')):
        continue
    print(f"\r {im}", end='\n', flush=True)
    img_names += [im]

    haze = Image.open(img_dir + im)
    haze = process_input(haze)

    with torch.no_grad():
        pred = net(haze)

    ts = torch.squeeze(pred.clamp(0, 1).cpu())
    ultimate_pred = pred * (Daytime_PM_MAX - Daytime_PM_MIN) + Daytime_PM_MIN
    print("Estimated PM2.5: ", ultimate_pred.flatten())
    all_pred = np.vstack((all_pred, ultimate_pred))

df = pd.DataFrame({"IMG": img_names, "Preds": all_pred.flatten()})
df.to_excel(os.path.join(output_dir, "results.xlsx"), index=False)
