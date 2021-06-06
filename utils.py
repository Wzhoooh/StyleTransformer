from aiogram import Dispatcher, types
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup

class PhotosStates(StatesGroup):
    waiting_for_content_image = State()
    waiting_for_style_image = State()

