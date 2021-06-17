import torch
import torchvision
import torchvision.transforms as transforms

from PIL import Image
import os
from pathlib import Path
import asyncio

import aiogram
from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.types import ReplyKeyboardRemove, \
    ReplyKeyboardMarkup, KeyboardButton

import style_transformer as st_tr
import properties as prop
import messages
from utils import States 
import users


def save_tensor_as_image(t: torch.Tensor, img_name):
    t = t.squeeze(0)
    t = transforms.ToPILImage(mode='RGB')(t)
    t.save(img_name)


def is_represents_int(s: str):
    try:
        int(s)
        return True
    except ValueError:
        return False

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
    for i in range(len(names)):
        if isinstance(names[i], list) or isinstance(names[i], tuple):
            markup.row(*names[i])
        else:        
            markup.add(names[i])
    
    return markup

MAIN_MENU = create_markup([["/image", "/style", "/affect"], "/make_magic", "/language", "/help"])


async def process_images(msg: types.Message, content_img, style_img, affect):
    # creting dir "images"
    Path("images").mkdir(parents=True, exist_ok=True)

    user_id = msg.from_user.id
    content_img_file_name = f"images/{user_id}_content.jpg"
    style_img_file_name = f"images/{user_id}_style.jpg"
    output_img_file_name = f"images/{user_id}_output.jpg"

    # downloading files
    await content_img.download(content_img_file_name)
    await style_img.download(style_img_file_name)

    content_img = Image.open(content_img_file_name)
    style_img = Image.open(style_img_file_name)

    image_transformer = st_tr.ImageTransformer(max_img_side_size=prop.MAX_IMG_SIDE_SIZE)
    content_img, style_img = image_transformer(content_img, style_img)
    '''       
    save_tensor_as_image(content_img, "images/content_intermed.jpg")
    save_tensor_as_image(style_img, "images/style_intermed.jpg")
    '''
    cont_w = 1
    st_w = 10 ** affect
    style_transformer = st_tr.StyleTransformer()
    tensor_image = await style_transformer(content_img, style_img, num_steps=prop.NUM_STEPS, 
        style_weight=st_w, content_weight=cont_w)
    tensor_image = tensor_image.squeeze(0)
    image = transforms.ToPILImage(mode='RGB')(tensor_image)
    image.save(output_img_file_name)

    os.remove(content_img_file_name)
    os.remove(style_img_file_name)

    return output_img_file_name


async def background_process_and_send_result(msg, content_img, style_img, affect):
    print("background process started")
    output_img_file_name = await process_images(msg, content_img, style_img, affect)
    output_img = types.input_file.InputFile(output_img_file_name)
    await msg.answer_photo(output_img)
    os.remove(output_img_file_name)


#---handlers---#

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
    markup = create_markup([messages.LANGS, "/cancel"])
    await States.waiting_for_language.set()
    await msg.answer(messages.COMMANDS["LANGUAGE"][await get_language(state)], reply_markup=markup)


async def choosing_language(msg: types.Message, state: FSMContext):
    new_lang = msg.text.upper()
    if not new_lang in messages.LANGS:
        await msg.answer(messages.warn("UNKNOWN_LANGUAGE", await get_language(state)))
    else:
        await state.update_data(language=new_lang)
        await msg.answer(messages.MESSAGES["LANGUAGE_CHANGED"][await get_language(state)], reply_markup=MAIN_MENU)
        await States.init_state.set()
   

async def process_image_command(msg: types.Message, state: FSMContext):
    markup = create_markup(["/cancel"])
    await States.waiting_for_content_image.set()
    await msg.answer(messages.MESSAGES["SEND_ME_CONTENT_IMAGE"][await get_language(state)], reply_markup=markup)


async def getting_content_image(msg: types.Message, state: FSMContext):
    await state.update_data(content_msg=msg)
    if msg.photo[-1]["height"] > prop.MAX_IMG_SIDE_SIZE or msg.photo[-1]["height"] > prop.MAX_IMG_SIDE_SIZE:
        await msg.answer(messages.warn("TOO_BIG_IMAGE_WAS_COMPRESSED", await get_language(state)))

    await States.init_state.set()
    await msg.answer(messages.MESSAGES["IMAGE_RECEIVED"][await get_language(state)], reply_markup=MAIN_MENU)


async def process_style_command(msg: types.Message, state: FSMContext):
    markup = create_markup(["/cancel"])
    await States.waiting_for_style_image.set()
    await msg.answer(messages.MESSAGES["SEND_ME_STYLE_IMAGE"][await get_language(state)], reply_markup=markup)

async def getting_style_image(msg: types.Message, state: FSMContext):
    await state.update_data(style_msg=msg)
    await States.init_state.set()
    await msg.answer(messages.MESSAGES["IMAGE_RECEIVED"][await get_language(state)], reply_markup=MAIN_MENU)


async def process_affect_command(msg: types.Message, state: FSMContext):
    markup = create_markup([list(map(lambda x: str(x), range(1, prop.AFFECT_MAX+1))), "/cancel"])
    await States.waiting_for_affect.set()
    await msg.answer(messages.COMMANDS["AFFECT"][await get_language(state)], reply_markup=markup)


async def getting_affect(msg: types.Message, state: FSMContext):
    affect_str = msg.text
    if not is_represents_int(affect_str):
        await msg.answer(messages.warn("VALUE_MUST_BE_INT", await get_language(state)))
        return
    
    affect_val = int(affect_str)
    if affect_val > prop.AFFECT_MAX:
        await msg.answer(messages.warn("TOO_BIG_VALUE", await get_language(state)))
        return
    if affect_val == 0:
        await msg.answer(messages.warn("VALUE_MUST_BE_NON_ZERO", await get_language(state)))
        return
    if affect_val < 0:
        await msg.answer(messages.warn("VALUE_MUST_BE_POSITIVE", await get_language(state)))
        return

    await state.update_data(affect=affect_val)
    await msg.answer(messages.MESSAGES["AFFECT_CHANGED"][await get_language(state)], reply_markup=MAIN_MENU)
    await States.init_state.set()


async def process_make_magic_command(msg: types.Message, state: FSMContext):
    info = await state.get_data()
    content_msg = info.get("content_msg")
    style_msg = info.get("style_msg")
    affect = info.get("affect")

    if affect == None:
        affect = prop.AFFECT_DEFAULT

    if content_msg == None:
        await msg.answer(messages.error("CONTENT_IMAGE_NOT_RECEIVED", await get_language(state)), reply_markup=MAIN_MENU)
    elif style_msg == None:
        await msg.answer(messages.error("STYLE_IMAGE_NOT_RECEIVED", await get_language(state)), reply_markup=MAIN_MENU)
    else:
        await msg.answer(messages.MESSAGES["WAIT_FOR_SEVERAL_MINUTES"][await get_language(state)])
        asyncio.create_task(background_process_and_send_result(msg, content_msg.photo[-1],  style_msg.photo[-1], affect))

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
   
    dp.register_message_handler(process_affect_command, commands=["affect"], state="*")
    dp.register_message_handler(getting_affect, content_types=types.message.ContentType.TEXT, state=States.waiting_for_affect)
 
    dp.register_message_handler(process_make_magic_command, commands=["make_magic"], state="*")

    dp.register_message_handler(unknown_message, content_types=types.message.ContentType.ANY, state="*")


