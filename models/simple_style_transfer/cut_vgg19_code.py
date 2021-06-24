import torch
import torch.nn as nn
import torch.nn.functional as F


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = mean.clone().detach().view(-1, 1, 1) # torch.tensor(mean).view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1) # torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


class CutVgg19(nn.Module):
    def __init__(self):
        super(CutVgg19, self).__init__()
        self.norm = Normalization(torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225]))
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu_1 = nn.ReLU()
        self.conv_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu_2 = nn.ReLU()
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv_3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu_3 = nn.ReLU()
        self.conv_4 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu_4 = nn.ReLU()
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv_5 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    
    def forward(self, x):
        return conv_5(pool_4(relu_4(conv_4(relu_3(conv_3(pool_2(relu_2(conv_2(relu_1(conv_1(norm(x))))))))))))

'''
cut_vgg19_model = nn.ModuleDict([
    ("norm", Normalization(torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225]))),
    ("conv_1", nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
    ("relu_1", nn.ReLU()),
    ("conv_2", nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
    ("relu_2", nn.ReLU()),
    ("pool_2", nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),
    ("conv_3", nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
    ("relu_3", nn.ReLU()),
    ("conv_4", nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
    ("relu_4", nn.ReLU()),
    ("pool_4", nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),
    ("conv_5", nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
])
'''
