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

import models.simple_style_transfer.style_transformer as simple_st_tr
import models.gan_style_transfer.style_transformer as gan_st_tr
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


GANS = {
    1: "candy",
    2: "mosaic",
    3: "starry-night",
    4: "udnie"
}

async def gan_style_transfer(msg: types.Message, content_img, style_img_number):
    '''
    if content_img["height"] > prop.MAX_IMG_SIDE_SIZE_FOR_GAN_ST_TR or content_img["height"] > prop.MAX_IMG_SIDE_SIZE_FOR_GAN_ST_TR:
        await msg.answer(messages.warn("TOO_BIG_IMAGE_WAS_COMPRESSED", await get_language(state)))
    '''

    # creating dir "images"
    Path("images").mkdir(parents=True, exist_ok=True)

    user_id = msg.from_user.id
    content_img_file_name = f"images/{user_id}_content.jpg"
    output_img_file_name = f"images/{user_id}_output.jpg"

    # downloading file
    await content_img.download(content_img_file_name)

    model_name = GANS.get(style_img_number)
    if model_name == None:
        raise RuntimeError(f"unknown gan model number({style_img_numger})")

    img = gan_st_tr.transfer(content_img_file_name, model_name, prop.MAX_IMG_SIDE_SIZE_FOR_GAN_ST_TR)
    gan_st_tr.tensor_save_bgrimage(img, output_img_file_name)

    os.remove(content_img_file_name)
    return output_img_file_name


async def simple_style_transfer(msg: types.Message, content_img, style_img, affect):
    '''
    if content_img["height"] > prop.MAX_IMG_SIDE_SIZE_FOR_SIMPLE_ST_TR or content_img["height"] > prop.MAX_IMG_SIDE_SIZE_FOR_SIMPLE_ST_TR:
        await msg.answer(messages.warn("TOO_BIG_IMAGE_WAS_COMPRESSED", await get_language(state)))
    '''

    # creating dir "images"
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

    image_transformer = simple_st_tr.ImageTransformer(max_img_side_size=prop.MAX_IMG_SIDE_SIZE_FOR_SIMPLE_ST_TR)
    content_img, style_img = image_transformer(content_img, style_img)
    '''       
    save_tensor_as_image(content_img, "images/content_intermed.jpg")
    save_tensor_as_image(style_img, "images/style_intermed.jpg")
    '''
    cont_w = 1
    st_w = 10 ** affect
    style_transformer = simple_st_tr.StyleTransformer()
    tensor_image = await style_transformer(content_img, style_img, num_steps=prop.NUM_STEPS, 
        style_weight=st_w, content_weight=cont_w)
    tensor_image = tensor_image.squeeze(0)
    image = transforms.ToPILImage(mode='RGB')(tensor_image)
    image.save(output_img_file_name)

    os.remove(content_img_file_name)
    os.remove(style_img_file_name)

    return output_img_file_name


async def make_background_gan_style_transfer_and_send_result(msg, state: FSMContext, content_img, style_img_number):
    print("background gan style transfer started")
    output_img_file_name = await gan_style_transfer(msg, content_img, style_img_number)
    output_img = types.input_file.InputFile(output_img_file_name)
    await msg.answer_photo(output_img)
    os.remove(output_img_file_name)
    await state.update_data(is_running=False)


async def make_background_simple_style_transfer_and_send_result(msg, state: FSMContext, content_img, style_img, affect):
    print("background simple style transfer started")
    output_img_file_name = await simple_style_transfer(msg, content_img, style_img, affect)
    output_img = types.input_file.InputFile(output_img_file_name)
    await msg.answer_photo(output_img)
    os.remove(output_img_file_name)
    await state.update_data(is_running=False)


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
        await msg.answer(messages.error("UNKNOWN_LANGUAGE", await get_language(state)))
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
    await States.init_state.set()
    await msg.answer(messages.MESSAGES["IMAGE_RECEIVED"][await get_language(state)], reply_markup=MAIN_MENU)


async def process_style_command(msg: types.Message, state: FSMContext):
    markup = create_markup(["/get_all_gan_images", list(map(lambda x: str(x), range(1, len(GANS)+1))), "/cancel"])
    await States.waiting_for_style_image.set()
    await msg.answer(messages.MESSAGES["SEND_ME_STYLE_IMAGE"][await get_language(state)], reply_markup=markup)

async def getting_style_image(msg: types.Message, state: FSMContext):
    await state.update_data(style_msg=msg)
    await States.init_state.set()
    await msg.answer(messages.MESSAGES["IMAGE_RECEIVED"][await get_language(state)], reply_markup=MAIN_MENU)

async def get_all_gan_images(msg: types.Message):
    for i in range(1, len(GANS)+1):
        await msg.answer(i)
        img = types.input_file.InputFile(f"models/gan_style_transfer/images/{GANS[i]}.jpg")
        await msg.answer_photo(img)

async def getting_number_of_gan_image(msg: types.Message, state: FSMContext):
    num_text = msg.text
    if not is_represents_int(num_text):
        await msg.answer(messages.error("VALUE_MUST_BE_INT", await get_language(state)))
        return

    num = int(num_text)
    if num > len(GANS) + 1:
        await msg.answer(messages.error("TOO_BIG_VALUE", await get_language(state)))
        return
    if num == 0:
        await msg.answer(messages.error("VALUE_MUST_BE_NON_ZERO", await get_language(state)))
        return
    if num < 0:
        await msg.answer(messages.error("VALUE_MUST_BE_POSITIVE", await get_language(state)))
        return

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
        await msg.answer(messages.error("VALUE_MUST_BE_INT", await get_language(state)))
        return
    
    affect_val = int(affect_str)
    if affect_val > prop.AFFECT_MAX:
        await msg.answer(messages.error("TOO_BIG_VALUE", await get_language(state)))
        return
    if affect_val == 0:
        await msg.answer(messages.error("VALUE_MUST_BE_NON_ZERO", await get_language(state)))
        return
    if affect_val < 0:
        await msg.answer(messages.error("VALUE_MUST_BE_POSITIVE", await get_language(state)))
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

    is_running = info.get("is_running")
    if is_running != None and is_running == True:
        await msg.answer(messages.error("STYLE_TRANSFERRING_ALREADY_RUNNING", await get_language(state)), reply_markup=MAIN_MENU)
    elif content_msg == None:
        await msg.answer(messages.error("CONTENT_IMAGE_NOT_RECEIVED", await get_language(state)), reply_markup=MAIN_MENU)
    elif style_msg == None:
        await msg.answer(messages.error("STYLE_IMAGE_NOT_RECEIVED", await get_language(state)), reply_markup=MAIN_MENU)
    else:
        await state.update_data(is_running=True)
        await msg.answer(messages.MESSAGES["WAIT_FOR_SEVERAL_MINUTES"][await get_language(state)])

        if style_msg.text != None:
            # making gan style transfer
            asyncio.create_task(make_background_gan_style_transfer_and_send_result(msg, state, content_msg.photo[-1], int(style_msg.text)))
        else:
            # making simple style transfer
            asyncio.create_task(make_background_simple_style_transfer_and_send_result(msg, state, content_msg.photo[-1], style_msg.photo[-1], affect))

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
    dp.register_message_handler(get_all_gan_images, commands=["get_all_gan_images"], state="*")
    dp.register_message_handler(getting_number_of_gan_image, content_types=types.message.ContentType.TEXT, state=States.waiting_for_style_image)
    dp.register_message_handler(getting_style_image, content_types=types.message.ContentType.PHOTO, state=States.waiting_for_style_image)
 
    dp.register_message_handler(process_affect_command, commands=["affect"], state="*")
    dp.register_message_handler(getting_affect, content_types=types.message.ContentType.TEXT, state=States.waiting_for_affect)
 
    dp.register_message_handler(process_make_magic_command, commands=["make_magic"], state="*")

    dp.register_message_handler(unknown_message, content_types=types.message.ContentType.ANY, state="*")


