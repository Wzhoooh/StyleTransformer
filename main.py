import torch
import torchvision.transforms as transforms

from PIL import Image
import os

import aiogram
from aiogram import Bot, types
from aiogram.utils import executor
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import Dispatcher
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup

from aiogram.contrib.middlewares.logging import LoggingMiddleware

import style_transformer as st_tr
import properties as prop
import messages
from utils import PhotosStates 
import tokens

def save_tensor_as_image(t: torch.Tensor, img_name):
    t = t.squeeze(0)
    t = transforms.ToPILImage(mode='RGB')(t)
    t.save(img_name)

    

async def process_images(content_msg: types.Message, style_msg: types.Message):
    # getting files names
    if content_msg.from_user.id != content_msg.from_user.id:
        raise RuntimeError("got messages from different clients")

    user_id = content_msg.from_user.id
    content_img_file_name = f"images/{user_id}_content.jpg"
    style_img_file_name = f"images/{user_id}_style.jpg"
    output_img_file_name = f"images/{user_id}_output.jpg"

    # downloading files
    await content_msg.photo[-1].download(content_img_file_name)
    await style_msg.photo[-1].download(style_img_file_name)

    content_img = Image.open(content_img_file_name)
    style_img = Image.open(style_img_file_name)

    image_transformer = st_tr.ImageTransformer(max_img_side_size=prop.MAX_IMG_SIDE_SIZE)
    content_img, style_img = image_transformer(content_img, style_img)
    '''       
    save_tensor_as_image(content_img, "images/content_intermed.jpg")
    save_tensor_as_image(style_img, "images/style_intermed.jpg")
    '''
    style_transformer = st_tr.StyleTransformer()
    tensor_image = style_transformer(content_img, style_img, num_steps=prop.NUM_STEPS).squeeze(0)
    image = transforms.ToPILImage(mode='RGB')(tensor_image)
    image.save(output_img_file_name)

    os.remove(content_img_file_name)
    os.remove(style_img_file_name)

    return output_img_file_name


async def process_help_command(msg: types.Message):
    await msg.answer(messages.COMMANDS["help"][messages.CUR_LANG])


async def start_getting_images(msg: types.Message):
    await PhotosStates.waiting_for_content_image.set()
    await msg.answer(messages.MESSAGES["SEND_ME_CONTENT_IMAGE"][messages.CUR_LANG])


async def getting_content_image(msg: types.Message, state: FSMContext):
    await state.update_data(content_img=msg)
    if msg.photo[-1]["height"] > prop.MAX_IMG_SIDE_SIZE or msg.photo[-1]["height"] > prop.MAX_IMG_SIDE_SIZE:
        await msg.answer(messages.warn("TOO_BIG_IMAGE_WAS_COMPRESSED", messages.CUR_LANG))

    await PhotosStates.waiting_for_style_image.set()
    await msg.answer(messages.MESSAGES["SEND_ME_STYLE_IMAGE"][messages.CUR_LANG])

    
async def getting_style_image(msg: types.Message, state: FSMContext):
    await state.update_data(style_img=msg)
    await msg.answer(messages.MESSAGES["WAIT_FOR_SEVERAL_MINUTES"][messages.CUR_LANG])

    info = await state.get_data()
    output_img_file_name = await process_images(info["content_img"], info["style_img"])
    output_img = types.input_file.InputFile(output_img_file_name)
    await bot.send_photo(msg.from_user.id, output_img)

    os.remove(output_img_file_name)


def register_handlers(dp: Dispatcher):
    dp.register_message_handler(process_help_command, commands=["help"], state="*")
    dp.register_message_handler(start_getting_images, commands=["start"], state="*")
    dp.register_message_handler(getting_content_image, content_types=types.message.ContentType.PHOTO, state=PhotosStates.waiting_for_content_image)
    dp.register_message_handler(getting_style_image, content_types=types.message.ContentType.PHOTO, state=PhotosStates.waiting_for_style_image)


bot = Bot(token=tokens.TOKEN)
dp = Dispatcher(bot, storage=MemoryStorage())
dp.middleware.setup(LoggingMiddleware())

register_handlers(dp)
'''
@dp.message_handler()
async def echo_message(msg: types.Message):
    await msg.answer(messages.MESSAGES["UNKNOWN_COMMAND"][messages.CUR_LANG])
'''
@dp.message_handler(content_types=types.message.ContentType.ANY, state="*")
async def unknown_message(msg: types.Message):
    await msg.reply(messages.MESSAGES["UNKNOWN_COMMAND"][messages.CUR_LANG])


if __name__ == '__main__':
    executor.start_polling(dp)

