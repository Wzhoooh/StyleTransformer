import torch
import torchvision.transforms as transforms

from PIL import Image

import style_transformer as st_tr
import database

content_img = Image.open("images/content_image.jpg")
style_img = Image.open("images/style_image.jpg")

prop = database.Properties()

image_transformer = st_tr.ImageTransformer()
content_img, style_img = image_transformer(content_img, style_img)

style_transformer = st_tr.StyleTransformer()
tensor_image = style_transformer(content_img, style_img, num_steps=prop.num_steps).squeeze(0)
image = transforms.ToPILImage(mode='RGB')(tensor_image)
image.save("images/output.jpg")
