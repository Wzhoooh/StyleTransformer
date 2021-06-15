import torch
import torchvision
import torchvision.transforms as transforms

from PIL import Image
import os
from urllib.parse import urljoin

from pathlib import Path

import aiogram
from aiogram import Bot, types
from aiogram.utils import executor
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import Dispatcher
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup

from aiogram.types import ReplyKeyboardRemove, \
    ReplyKeyboardMarkup, KeyboardButton

from aiogram.contrib.middlewares.logging import LoggingMiddleware

import style_transformer as st_tr
import properties as prop
import messages
from utils import States 
import tokens
import users


MAIN_MENU = ReplyKeyboardMarkup(resize_keyboard=True).row("/image", "/style").add(
    KeyboardButton("/make_magic")).add(
    KeyboardButton("/language")).add(
    KeyboardButton("/help"))


def save_tensor_as_image(t: torch.Tensor, img_name):
    t = t.squeeze(0)
    t = transforms.ToPILImage(mode='RGB')(t)
    t.save(img_name)


async def get_field(state: FSMContext, field_name: str):
    all_fields = await state.get_data()
    return all_fields.get(field_name)


async def get_language(state: FSMContext):
    if state == None:
        return prop.DEFAULT_LANGUAGE

    language = await get_field(state, "language")
    if language == None:
        return prop.DEFAULT_LANGUAGE
    else:
        return language

def create_markup(names: list):
    markup = ReplyKeyboardMarkup(resize_keyboard=True)
    for i in names:
        markup.add(KeyboardButton(i))
    
    return markup


async def process_images(content_msg: types.Message, style_msg: types.Message):
    # getting files names
    if content_msg.from_user.id != content_msg.from_user.id:
        raise RuntimeError("got messages from different clients")

    # creting dir "images"
    Path("images").mkdir(parents=True, exist_ok=True)

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
    tensor_image = await style_transformer(content_img, style_img, num_steps=prop.NUM_STEPS)
    tensor_image = tensor_image.squeeze(0)
    image = transforms.ToPILImage(mode='RGB')(tensor_image)
    image.save(output_img_file_name)

    os.remove(content_img_file_name)
    os.remove(style_img_file_name)

    return output_img_file_name


async def process_start_command(msg: types.Message):
    users.add_user(msg.from_user)
    await States.init_state.set()
    await msg.answer(messages.COMMANDS["START"][prop.DEFAULT_LANGUAGE], reply_markup=MAIN_MENU)

async def process_help_command(msg: types.Message, state: FSMContext):
    await msg.answer(messages.COMMANDS["HELP"][await get_language(state)])


async def process_cancel_command(msg: types.Message, state: FSMContext):
    await States.init_state.set()
    await msg.answer(messages.COMMANDS["CANCEL"][await get_language(state)], reply_markup=MAIN_MENU)


async def process_language_command(msg: types.Message, state: FSMContext):
    markup = create_markup(messages.LANGS)
    await States.waiting_for_language.set()
    await msg.answer(messages.COMMANDS["LANGUAGE"][await get_language(state)], reply_markup=markup)


async def choosing_language(msg: types.Message, state: FSMContext):
    new_lang = msg.text.upper()
    if new_lang in messages.LANGS:
        await state.update_data(language=new_lang)
        await msg.answer(messages.MESSAGES["LANGUAGE_CHANGED"][await get_language(state)], reply_markup=MAIN_MENU)
    else:
         await msg.answer(messages.warn("UNKNOWN_LANGUAGE", await get_language(state)), reply_markup=MAIN_MENU)
   
    await States.init_state.set()
    

async def process_image_command(msg: types.Message, state: FSMContext):
    markup = create_markup(["/cancel", "/help"])
    await States.waiting_for_content_image.set()
    await msg.answer(messages.MESSAGES["SEND_ME_CONTENT_IMAGE"][await get_language(state)], reply_markup=markup)


async def getting_content_image(msg: types.Message, state: FSMContext):
    await state.update_data(content_img=msg)
    if msg.photo[-1]["height"] > prop.MAX_IMG_SIDE_SIZE or msg.photo[-1]["height"] > prop.MAX_IMG_SIDE_SIZE:
        await msg.answer(messages.warn("TOO_BIG_IMAGE_WAS_COMPRESSED", await get_language(state)))

    await States.init_state.set()
    await msg.answer(messages.MESSAGES["IMAGE_RECEIVED"][await get_language(state)], reply_markup=MAIN_MENU)


async def process_style_command(msg: types.Message, state: FSMContext):
    markup = create_markup(["/cancel", "/help"])
    await States.waiting_for_style_image.set()
    await msg.answer(messages.MESSAGES["SEND_ME_STYLE_IMAGE"][await get_language(state)], reply_markup=markup)

async def getting_style_image(msg: types.Message, state: FSMContext):
    await state.update_data(style_img=msg)
    await States.init_state.set()
    await msg.answer(messages.MESSAGES["IMAGE_RECEIVED"][await get_language(state)], reply_markup=MAIN_MENU)


async def process_make_magic_command(msg: types.Message, state: FSMContext):
    info = await state.get_data()
    content_img_id = info.get("content_img")
    style_img_id = info.get("style_img")
    if content_img_id == None:
        await msg.answer(messages.error("CONTENT_IMAGE_NOT_RECEIVED", await get_language(state)), reply_markup=MAIN_MENU)
    elif style_img_id == None:
        await msg.answer(messages.error("STYLE_IMAGE_NOT_RECEIVED", await get_language(state)), reply_markup=MAIN_MENU)
    else:
        await msg.answer(messages.MESSAGES["WAIT_FOR_SEVERAL_MINUTES"][await get_language(state)])
        output_img_file_name = await process_images(content_img_id, style_img_id)
        output_img = types.input_file.InputFile(output_img_file_name)
        await bot.send_photo(msg.from_user.id, output_img, reply_markup=MAIN_MENU)
        os.remove(output_img_file_name)

    await States.init_state.set()


async def unknown_message(msg: types.Message, state: FSMContext):
    await msg.reply(messages.error("UNKNOWN_COMMAND", await get_language(state)))


def register_handlers(dp: Dispatcher):
    dp.register_message_handler(process_start_command, commands=["start"], state="*")
    dp.register_message_handler(process_help_command, commands=["help"], state="*")
    dp.register_message_handler(process_cancel_command, commands=["cancel"], state="*")

    dp.register_message_handler(process_language_command, commands=["language"], state="*")
    dp.register_message_handler(choosing_language, content_types=types.message.ContentType.TEXT, state=States.waiting_for_language)

    dp.register_message_handler(process_image_command, commands=["image"], state="*")
    dp.register_message_handler(getting_content_image, content_types=types.message.ContentType.PHOTO, state=States.waiting_for_content_image)

    dp.register_message_handler(process_style_command, commands=["style"], state="*")
    dp.register_message_handler(getting_style_image, content_types=types.message.ContentType.PHOTO, state=States.waiting_for_style_image)
    
    dp.register_message_handler(process_make_magic_command, commands=["make_magic"], state="*")

    dp.register_message_handler(unknown_message, content_types=types.message.ContentType.ANY, state="*")



# webhook settings
WEBHOOK_HOST = "https://afternoon-hollows-87094.herokuapp.com"
WEBHOOK_PATH = f"/webhook/{tokens.TOKEN}"
WEBHOOK_URL = urljoin(WEBHOOK_HOST, WEBHOOK_PATH)
 
# webserver settings
WEBAPP_HOST = "0.0.0.0"
WEBAPP_PORT = int(os.environ.get('PORT', 5000))


bot = Bot(token=tokens.TOKEN)
dp = Dispatcher(bot, storage=MemoryStorage())
dp.middleware.setup(LoggingMiddleware())

register_handlers(dp)


async def on_startup(dp):
    print('Starting...')
    await bot.set_webhook(WEBHOOK_URL)

async def on_shutdown(dp):
    print('Shutting down...')
    print('Bye!')


if __name__ == '__main__':
    print("main started")
    if prop.CONNECTION_TYPE == "POLLING":
        print("connection is polling")
        executor.start_polling(dp)
    elif prop.CONNECTION_TYPE == "WEBHOOKS":
        print("host:", WEBAPP_HOST)
        print("port:", WEBAPP_PORT)
        print("url:", WEBHOOK_URL)
        executor.start_webhook(
            dispatcher=dp,
            webhook_path=WEBHOOK_PATH,
            on_startup=on_startup,
            on_shutdown=on_shutdown,
            skip_updates=True,
            host=WEBAPP_HOST,
            port=WEBAPP_PORT,
        )
    else:
        print("uncorrect CONNECTION_TYPE")

