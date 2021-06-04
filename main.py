import torch
import torchvision.transforms as transforms

from PIL import Image

import style_transformer as st_tr
import properties as prop
import tokens

from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor

bot = Bot(token=tokens.TOKEN)
dp = Dispatcher(bot)

@dp.message_handler(commands=["start"])
async def process_start_command(msg: types.Message):
    await bot.send_message(msg.from_user.id, prop.START_MSG)

@dp.message_handler(commands=["help"])
async def process_help_command(msg: types.Message):
    await bot.send_message(msg.from_user.id, prop.HELP_MSG)


@dp.message_handler(commands=['photo'])
async def process_photo_command(msg: types.Message):
    photo = types.input_file.InputFile("images/output.jpg")
    await bot.send_photo(msg.from_user.id, photo)

@dp.message_handler()
async def echo_message(msg: types.Message):
    await bot.send_message(msg.from_user.id, msg.text)



if __name__ == '__main__':
    executor.start_polling(dp)

'''
content_img = Image.open("images/content_image.jpg")
style_img = Image.open("images/style_image.jpg")

image_transformer = st_tr.ImageTransformer()
content_img, style_img = image_transformer(content_img, style_img)

style_transformer = st_tr.StyleTransformer()
tensor_image = style_transformer(content_img, style_img, num_steps=prop.NUM_STEPS).squeeze(0)
image = transforms.ToPILImage(mode='RGB')(tensor_image)
image.save("images/output.jpg")
'''
