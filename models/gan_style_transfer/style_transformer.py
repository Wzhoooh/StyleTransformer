import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
import sys

from models.gan_style_transfer.transformer_net import TransformerNet as TransformerNet

class ImageTransformer(object):
    def __init__(self, max_img_side_size):
        self.__max_img_side_size = max_img_side_size

    def __call__(self, content_image, height=None, width=None):
        if height == None: # taking size of content_image
            _, height = content_image.size

        if width == None:
            width, _ = content_image.size

        # compression of image if its side size > max_img_side_size
        max_size = max([height, width])
        compression_coef = 0.0
        if max_size > self.__max_img_side_size:
            compression_coef = self.__max_img_side_size / max_size
            height = int(height * compression_coef)
            width = int(width * compression_coef)

        if height > self.__max_img_side_size or width > self.__max_img_side_size or height <= 0 or width <= 0:
            raise ValueError("uncorrect size of image")

        loader = transforms.Compose([
            transforms.Resize(size=[height, width]), # нормируем размер изображения
            transforms.CenterCrop(size=[height, width]),
            transforms.ToTensor()]) # превращаем в удобный формат
        return loader(content_image).unsqueeze(0).to(torch.float)


def transfer(img_path, style, img_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    img = Image.open(img_path)
    image_transformer = ImageTransformer(img_size)
    img = image_transformer(img)

    model = TransformerNet().to(device)
    model.load_state_dict(torch.load(f"models/gan_style_transfer/{style}.pth"))

    for p in model.parameters():
        p.requires_grad = False

    return model(img)[0]

def tensor_save_rgbimage(tensor, filename, cuda=False):
    if cuda:
        img = tensor.clone().cpu().clamp(0, 255).numpy()
    else:
        img = tensor.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)

def tensor_save_bgrimage(tensor, filename, cuda=False):
    (b, g, r) = torch.chunk(tensor, 3)
    tensor = torch.cat((r, g, b))
    tensor_save_rgbimage(tensor, filename, cuda)

