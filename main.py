import torch
import torchvision.transforms as transforms

from PIL import Image

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
from messages import MESSAGES
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

    image_transformer = st_tr.ImageTransformer()
    content_img, style_img = image_transformer(content_img, style_img)

    save_tensor_as_image(content_img, "content_intermed.jpg")
    save_tensor_as_image(style_img, "style_intermed.jpg")

    style_transformer = st_tr.StyleTransformer()
    tensor_image = style_transformer(content_img, style_img, num_steps=prop.NUM_STEPS).squeeze(0)
    image = transforms.ToPILImage(mode='RGB')(tensor_image)
    image.save(output_img_file_name)
'''
    await os.remove(content_img_file_name)
    await os.remove(style_img_file_name)
'''
    return output_img_file_name

async def start_getting_images(msg: types.Message):
    await PhotosStates.waiting_for_content_image.set()
    await msg.answer("Send me content image")


async def getting_content_image(msg: types.Message, state: FSMContext):
    await state.update_data(content_img=msg)
    await PhotosStates.waiting_for_style_image.set()
    await msg.answer("Send me style image")

    
async def getting_style_image(msg: types.Message, state: FSMContext):
    await state.update_data(style_img=msg)
    await msg.answer("Wait several minutes")
    info = await state.get_data()
    output_img_file_name = await process_images(info["content_img"], info["style_img"])
    output_img = types.input_file.InputFile(output_img_file_name)
    await bot.send_photo(msg.from_user.id, output_img)
'''
    await os.remove(output_img_file_name)
'''


def register_handlers(dp: Dispatcher):
    dp.register_message_handler(start_getting_images, commands=["start"], state="*")
    dp.register_message_handler(getting_content_image, content_types=types.message.ContentType.PHOTO, state=PhotosStates.waiting_for_content_image)
    dp.register_message_handler(getting_style_image, content_types=types.message.ContentType.PHOTO, state=PhotosStates.waiting_for_style_image)


bot = Bot(token=tokens.TOKEN)
dp = Dispatcher(bot, storage=MemoryStorage())
dp.middleware.setup(LoggingMiddleware())

register_handlers(dp)
'''
@dp.message_handler(commands=["start"])
async def process_start_command(msg: types.Message):
    await bot.send_message(msg.from_user.id, MESSAGES["start"])

@dp.message_handler(commands=["help"])
async def process_help_command(msg: types.Message):
    await bot.send_message(msg.from_user.id, MESSAGES["help"])

@dp.message_handler(commands=["photo"])
async def process_photo_command(msg: types.Message):
    photo = types.input_file.InputFile("images/output.jpg")
    await bot.send_photo(msg.from_user.id, photo)

@dp.message_handler(content_types=types.message.ContentType.PHOTO)
async def photo_handler(msg: types.Message):
    state = dp.current_state(user=msg.from_user.id)
    
    await bot.send_photo(msg.from_user.id, msg.photo[-1]["file_id"])

@dp.message_handler()
async def echo_message(msg: types.Message):
    await bot.send_message(msg.from_user.id, msg.text)

@dp.message_handler(content_types=types.message.ContentType.ANY)
async def unknown_message(msg: types.Message):
    await msg.reply(MESSAGES["unknown"])
'''

if __name__ == '__main__':
    executor.start_polling(dp)

