import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models


from PIL import Image

import style_transformer as st_tr

content_img = Image.open("images/lisa.jpg")
style_img = Image.open("images/picasso.jpg")

image_transformer = st_tr.ImageTransformer()
content_img, style_img = image_transformer(content_img, style_img)

style_transformer = st_tr.StyleTransformer()
image = transforms.ToPILImage(mode='RGB')(style_transformer(content_img, style_img).squeeze(0))
image.save("output.jpg")
