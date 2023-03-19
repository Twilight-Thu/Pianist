import torch
import torch.nn as nn
from torchsummary import summary
from thop import profile

import SingleNet_backbone

class PM_Single_Net(nn.Module):
    def __init__(self, Body=None, pretrained=True):
        super(PM_Single_Net, self).__init__()
        self._get_Body(Body, pretrained)

    def forward(self, x):
        y = self.Body(x)
        y = self.fc(y)
        return y

    def _get_Body(self, name='vgg16', pretrained=True):
        if name == 'vgg16':
            self.Body = SingleNet_backbone.VGG16_15(pretrained)
            self.fc = nn.Linear(4096, 1)
        elif name == 'resnet18':
            self.Body = SingleNet_backbone.Resnet18_17(pretrained)
            self.fc = nn.Linear(512, 1)
        elif name == 'mobilev2':
            self.Body = SingleNet_backbone.MobileNetv2(pretrained)
            self.fc = nn.Linear(1280, 1)

if __name__ == "__main__":
    net = PM_Single_Net(Body='mobilev2')
    summary(net, input_size=[(3, 256, 256), ])

    x1 = torch.rand(1, 3, 256, 256)
    flops, params = profile(net, input=(x1,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')